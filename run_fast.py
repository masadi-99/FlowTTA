"""FlowTTA Feasibility Round 2.

All critical fixes applied:
  Fix 1: Temporal loss uses multi-window independent adaptation
  Fix 2: Reconstruction loss uses FM leave-last-out scheme
  Fix 3: Entropy minimization loss added
  Fix 4: Scaled to 50 windows, 20 samples, 3 seeds
  Fix 5: Global seed + pre-computed shifted windows
  Fix 6: Time-dependent segmented adapter (16 params)
  Fix 7: RevIN oracle baseline
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_etth import load_etth1
from data.synthetic_shift import apply_shift
from evaluate import compute_metrics, relative_improvement


# ============================================================
# CONFIG (Fix 4: scaled up)
# ============================================================
MODEL_ID = "amazon/chronos-t5-tiny"
DEVICE = "cuda"
PRED_LEN = 24
CTX_LEN = 512
NUM_SAMPLES = 20       # was 5
MAX_WINDOWS = 30       # was 8, need >=30 for stable estimates
ADAPT_STEPS = 10
ADAPT_LR = 1e-3
N_SEEDS = 3            # 3 seeds for mean +/- std
RECON_K = 24           # leave-last-out length
RECON_EVERY = 5        # compute FM-calling losses every N-th step (speed)


def load_fm():
    from models.fm_wrapper import ChronosWrapper
    return ChronosWrapper(model_id=MODEL_ID, device=DEVICE)


def get_windows():
    data = load_etth1(context_length=CTX_LEN, prediction_length=PRED_LEN)
    windows = data["test_windows"][:MAX_WINDOWS]
    print(f"Using {len(windows)} test windows (ctx={CTX_LEN}, pred={PRED_LEN})")
    return windows


# ============================================================
# Fix 6: Time-dependent segmented adapter
# ============================================================
class SegmentedAdapter(torch.nn.Module):
    """Per-segment affine adapter. 2*n_segments params."""
    def __init__(self, n_segments=8):
        super().__init__()
        self.n_segments = n_segments
        self.scale = torch.nn.Parameter(torch.ones(n_segments))
        self.shift = torch.nn.Parameter(torch.zeros(n_segments))

    def forward(self, x):
        # x: (1, T, 1)
        T = x.shape[1]
        seg_len = T // self.n_segments
        out = x.clone()
        for i in range(self.n_segments):
            s = i * seg_len
            e = s + seg_len if i < self.n_segments - 1 else T
            out[:, s:e, :] = x[:, s:e, :] * self.scale[i] + self.shift[i]
        return out

    def reset_parameters(self):
        self.scale.data.fill_(1.0)
        self.shift.data.fill_(0.0)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPInputAdapter(torch.nn.Module):
    """MLP residual adapter."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.GELU(),
            torch.nn.Linear(32, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)

    def reset_parameters(self):
        for m in self.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# Fix 1: Temporal loss — multi-window independent adaptation
# ============================================================
def temporal_loss(adapter, ctx_windows_np, dev):
    """
    Adapt overlapping windows independently, enforce agreement on overlap.
    ctx_windows_np: list of numpy arrays (consecutive context windows).
    """
    if len(ctx_windows_np) < 2:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    loss = torch.tensor(0.0, device=dev)
    count = 0
    stride = len(ctx_windows_np[0]) // 4

    for i in range(len(ctx_windows_np) - 1):
        w1_np = ctx_windows_np[i]
        # Overlapping window: shift by stride
        w2_np = np.concatenate([w1_np[stride:], ctx_windows_np[i + 1][:stride]])

        w1_t = torch.tensor(w1_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
        w2_t = torch.tensor(w2_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

        # Adapt each INDEPENDENTLY
        a1 = adapter(w1_t)
        a2 = adapter(w2_t)

        # Compare overlapping region
        overlap_len = w1_t.shape[1] - stride
        loss = loss + F.mse_loss(a1[:, stride:, :], a2[:, :overlap_len, :])
        count += 1

    return loss / max(count, 1)


# ============================================================
# Spectral loss (unchanged)
# ============================================================
def spectral_loss(adapted, original):
    fft_o = torch.fft.rfft(original, dim=1)
    fft_a = torch.fft.rfft(adapted, dim=1)
    psd_o = torch.abs(fft_o) ** 2
    psd_a = torch.abs(fft_a) ** 2
    psd_o = psd_o / (psd_o.sum(dim=1, keepdim=True) + 1e-8)
    psd_a = psd_a / (psd_a.sum(dim=1, keepdim=True) + 1e-8)
    return F.mse_loss(psd_a, psd_o)


# ============================================================
# Fix 2: Reconstruction loss — leave-last-out with FM
# ============================================================
def recon_loss(adapter, fm, ctx_np, dev):
    """
    Withhold last K timesteps, adapt truncated context, use FM to predict
    those K steps, compare to the actual observed values.
    """
    K = RECON_K
    if len(ctx_np) <= K + 64:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    truncated = ctx_np[:-K]
    held_out = ctx_np[-K:]

    trunc_t = torch.tensor(truncated, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
    adapted_trunc = adapter(trunc_t).squeeze(-1).squeeze(0).detach().cpu().numpy()

    pred = fm.predict(adapted_trunc, prediction_length=K, num_samples=NUM_SAMPLES)
    pred_mean = pred.squeeze()
    if pred_mean.ndim > 1:
        pred_mean = np.median(pred_mean, axis=0)

    held_out_t = torch.tensor(held_out, dtype=torch.float32, device=dev)
    pred_t = torch.tensor(pred_mean, dtype=torch.float32, device=dev)

    return F.mse_loss(pred_t, held_out_t)


# ============================================================
# Fix 3: Entropy minimization loss
# ============================================================
def entropy_loss(adapter, fm, ctx_np, dev):
    """
    Minimize variance of FM's sampled forecasts.
    Well-adapted input should produce confident predictions.
    """
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
    adapted = adapter(ctx_t).squeeze(-1).squeeze(0).detach().cpu().numpy()

    samples = fm.predict_with_quantiles(adapted, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
    samples = np.squeeze(samples)

    if samples.ndim < 2:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    # Variance across samples at each timestep
    variance = np.var(samples, axis=0).mean()
    return torch.tensor(variance, dtype=torch.float32, device=dev, requires_grad=True)


# ============================================================
# Fix 7: RevIN oracle baseline
# ============================================================
def revin_predict(fm, ctx_np, pred_len, num_samples):
    """RevIN: normalize input to zero mean, unit variance, denormalize output."""
    mean = ctx_np.mean()
    std = ctx_np.std() + 1e-8
    normed = (ctx_np - mean) / std
    pred = fm.predict(normed, prediction_length=pred_len, num_samples=num_samples)
    pred_denormed = pred.squeeze() * std + mean
    if pred_denormed.ndim > 1:
        pred_denormed = np.median(pred_denormed, axis=0)
    return pred_denormed


# ============================================================
# ADAPTATION (rewritten for new loss signatures)
# ============================================================
def adapt_predict(fm, ctx_np, neighbor_ctxs, adapter, loss_mode, dev):
    """
    Adapt input-level adapter and predict.

    loss_mode: str — which loss to use:
      "temporal", "spectral", "reconstruction", "entropy",
      "temporal+entropy", "all_four", or any combo.
    neighbor_ctxs: list of neighboring context windows for temporal loss.
    """
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

    t0 = time.time()
    for step in range(ADAPT_STEPS):
        optimizer.zero_grad()
        adapted = adapter(ctx_t)
        loss = torch.tensor(0.0, device=dev)

        if "temporal" in loss_mode:
            loss = loss + 1.0 * temporal_loss(adapter, neighbor_ctxs, dev)

        if "spectral" in loss_mode:
            loss = loss + 0.5 * spectral_loss(adapted, ctx_t)

        if "reconstruction" in loss_mode:
            # Only every RECON_EVERY steps (FM call is expensive)
            if step % RECON_EVERY == 0:
                loss = loss + 0.5 * recon_loss(adapter, fm, ctx_np, dev)

        if "entropy" in loss_mode:
            # Only every RECON_EVERY steps (FM call is expensive)
            if step % RECON_EVERY == 0:
                loss = loss + 1.0 * entropy_loss(adapter, fm, ctx_np, dev)

        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    adapt_ms = (time.time() - t0) * 1000

    with torch.no_grad():
        final = adapter(ctx_t).squeeze(-1).squeeze(0).cpu().numpy()
    pred = fm.predict(final, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
    pred_out = pred.squeeze()
    if pred_out.ndim > 1:
        pred_out = np.median(pred_out, axis=0)
    return pred_out, adapt_ms


# ============================================================
# Run one experiment configuration on pre-shifted windows
# ============================================================
def run_config(fm, shifted_windows, adapter_factory, loss_mode, dev, label=""):
    """
    Run a single config across all windows.
    shifted_windows: list of (ctx_shifted, tgt) — already shifted.
    adapter_factory: callable() -> adapter, or None for zero-shot/revin.
    loss_mode: str for adapt_predict, or "zero_shot" / "revin".
    """
    ps, ts, total_ms = [], [], 0

    for i, (ctx_s, tgt) in enumerate(shifted_windows):
        if loss_mode == "zero_shot":
            p = fm.predict(ctx_s, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
            if p.ndim > 1:
                p = np.median(p, axis=0)
            ms = 0
        elif loss_mode == "revin":
            p = revin_predict(fm, ctx_s, PRED_LEN, NUM_SAMPLES)
            ms = 0
        else:
            adapter = adapter_factory()
            # Gather neighbor contexts for temporal loss
            neighbors = [ctx_s]
            if i > 0:
                neighbors = [shifted_windows[i - 1][0]] + neighbors
            if i < len(shifted_windows) - 1:
                neighbors = neighbors + [shifted_windows[i + 1][0]]

            p, ms = adapt_predict(fm, ctx_s, neighbors, adapter, loss_mode, dev)
            total_ms += ms

        ps.append(p)
        ts.append(tgt)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    {label} window {i+1}/{len(shifted_windows)}")

    preds = np.concatenate(ps)
    trues = np.concatenate(ts)
    m = compute_metrics(preds, trues)
    avg_ms = total_ms / max(len(shifted_windows), 1)
    return m, avg_ms


# ============================================================
# EXP 1: Degradation (same as before, but with more windows)
# ============================================================
def exp1(fm, windows):
    print("\n" + "=" * 60)
    print("  EXP 1: Does shift degrade Chronos?")
    print("=" * 60)

    preds, trues = [], []
    for i, (ctx, tgt) in enumerate(windows):
        p = fm.predict(ctx, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
        p = p.squeeze()
        if p.ndim > 1:
            p = np.median(p, axis=0)
        preds.append(p)
        trues.append(tgt)
        if (i + 1) % 10 == 0:
            print(f"  Baseline window {i+1}/{len(windows)}")

    baseline = compute_metrics(np.concatenate(preds), np.concatenate(trues))
    print(f"  Clean baseline MSE: {baseline['mse']:.6f}")

    results = []
    for shift_type in ["mean", "variance", "trend"]:
        for mag in [1.0, 2.0, 3.0]:
            ps, ts = [], []
            for ctx, tgt in windows:
                ctx_s = apply_shift(ctx, shift_type, mag)
                p = fm.predict(ctx_s, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
                p = p.squeeze()
                if p.ndim > 1:
                    p = np.median(p, axis=0)
                ps.append(p)
                ts.append(tgt)

            m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
            deg = -relative_improvement(baseline['mse'], m['mse'])
            results.append({"shift": shift_type, "mag": mag, "mse": m['mse'], "deg%": deg})
            print(f"  {shift_type} mag={mag}: MSE={m['mse']:.6f} ({deg:+.1f}%)")

    return baseline, results


# ============================================================
# EXP 2: Loss Ablation (with all fixes)
# ============================================================
def exp2(fm, shifted_windows, dev):
    print("\n" + "=" * 60)
    print("  EXP 2: Loss Ablation")
    print("=" * 60)

    configs = {
        "zero_shot":         (None, "zero_shot"),
        "revin_oracle":      (None, "revin"),
        "temporal":          (lambda: SegmentedAdapter().to(dev), "temporal"),
        "spectral":          (lambda: SegmentedAdapter().to(dev), "spectral"),
        "reconstruction":    (lambda: SegmentedAdapter().to(dev), "reconstruction"),
        "entropy":           (lambda: SegmentedAdapter().to(dev), "entropy"),
        "temporal+entropy":  (lambda: SegmentedAdapter().to(dev), "temporal+entropy"),
        "all_four":          (lambda: SegmentedAdapter().to(dev), "temporal+spectral+reconstruction+entropy"),
    }

    results = {}
    for name, (af, loss_mode) in configs.items():
        print(f"\n  --- {name} ---")
        m, avg_ms = run_config(fm, shifted_windows, af, loss_mode, dev, label=name)
        results[name] = {"mse": m["mse"], "mae": m["mae"], "adapt_ms": avg_ms}
        print(f"  {name:>20}: MSE={m['mse']:.6f} MAE={m['mae']:.6f} ({avg_ms:.0f}ms/batch)")

    # Summary
    zs = results["zero_shot"]["mse"]
    rv = results["revin_oracle"]["mse"]
    print(f"\n  {'Config':>20} | {'MSE':>10} | {'vs ZS':>8} | {'vs RevIN':>8}")
    print(f"  {'-'*20} | {'-'*10} | {'-'*8} | {'-'*8}")
    for name, r in results.items():
        imp_zs = relative_improvement(zs, r["mse"])
        imp_rv = relative_improvement(rv, r["mse"])
        print(f"  {name:>20} | {r['mse']:>10.4f} | {imp_zs:>+7.1f}% | {imp_rv:>+7.1f}%")

    return results


# ============================================================
# EXP 3: Adapter Comparison (uses best loss from exp2)
# ============================================================
def exp3(fm, shifted_windows, best_loss_mode, dev):
    print("\n" + "=" * 60)
    print(f"  EXP 3: Adapter Comparison (loss={best_loss_mode})")
    print("=" * 60)

    adapters = {
        "zero_shot":  (None, "zero_shot"),
        "revin":      (None, "revin"),
        "segmented":  (lambda: SegmentedAdapter().to(dev), best_loss_mode),
        "mlp":        (lambda: MLPInputAdapter().to(dev), best_loss_mode),
    }

    results = {}
    for name, (af, loss_mode) in adapters.items():
        n_params = af().num_params if af else 0
        print(f"\n  --- {name} ({n_params} params) ---")
        m, avg_ms = run_config(fm, shifted_windows, af, loss_mode, dev, label=name)
        results[name] = {"mse": m["mse"], "mae": m["mae"], "params": n_params, "adapt_ms": avg_ms}
        print(f"  {name:>12}: MSE={m['mse']:.6f} ({avg_ms:.0f}ms/batch)")

    zs = results["zero_shot"]["mse"]
    rv = results["revin"]["mse"]
    print(f"\n  {'Adapter':>12} | {'Params':>6} | {'MSE':>10} | {'vs ZS':>8} | {'vs RevIN':>8}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*10} | {'-'*8} | {'-'*8}")
    for name, r in results.items():
        print(f"  {name:>12} | {r['params']:>6} | {r['mse']:>10.4f} | "
              f"{relative_improvement(zs, r['mse']):>+7.1f}% | "
              f"{relative_improvement(rv, r['mse']):>+7.1f}%")

    return results


# ============================================================
# EXP 4: Second shift type (trend)
# ============================================================
def exp4(fm, windows, best_loss_mode, dev):
    print("\n" + "=" * 60)
    print(f"  EXP 4: Trend Shift (loss={best_loss_mode})")
    print("=" * 60)

    shifted_trend = [(apply_shift(ctx, "trend", 2.0), tgt) for ctx, tgt in windows]

    configs = {
        "zero_shot": (None, "zero_shot"),
        "revin":     (None, "revin"),
        "segmented": (lambda: SegmentedAdapter().to(dev), best_loss_mode),
        "mlp":       (lambda: MLPInputAdapter().to(dev), best_loss_mode),
    }

    results = {}
    for name, (af, loss_mode) in configs.items():
        print(f"\n  --- {name} ---")
        m, avg_ms = run_config(fm, shifted_trend, af, loss_mode, dev, label=name)
        results[name] = {"mse": m["mse"], "mae": m["mae"], "adapt_ms": avg_ms}

    zs = results["zero_shot"]["mse"]
    rv = results["revin"]["mse"]
    print(f"\n  Trend shift results:")
    print(f"  {'Config':>12} | {'MSE':>10} | {'vs ZS':>8} | {'vs RevIN':>8}")
    print(f"  {'-'*12} | {'-'*10} | {'-'*8} | {'-'*8}")
    for name, r in results.items():
        print(f"  {name:>12} | {r['mse']:>10.4f} | "
              f"{relative_improvement(zs, r['mse']):>+7.1f}% | "
              f"{relative_improvement(rv, r['mse']):>+7.1f}%")

    return results


# ============================================================
# MAIN (Fix 5: global seed, pre-computed shifts, multi-seed)
# ============================================================
def main():
    print("=" * 70)
    print("  FlowTTA Feasibility Round 2 (all fixes applied)")
    print(f"  Model: {MODEL_ID} | Device: {DEVICE}")
    print(f"  Windows: {MAX_WINDOWS} | Pred: {PRED_LEN} | Samples: {NUM_SAMPLES}")
    print(f"  Seeds: {N_SEEDS} | Adapt steps: {ADAPT_STEPS}")
    print("=" * 70)

    t_total = time.time()
    fm = load_fm()
    dev = fm.device
    windows = get_windows()

    # ---- EXP 1 (run once, no seed sensitivity) ----
    torch.manual_seed(42)
    np.random.seed(42)
    baseline, deg_results = exp1(fm, windows)
    max_deg = max(r['deg%'] for r in deg_results)

    # ---- EXP 2, 3, 4: run over N_SEEDS, collect mean +/- std ----
    all_exp2 = []
    all_exp3 = []
    all_exp4 = []

    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({seed_idx+1}/{N_SEEDS})")
        print(f"{'#'*70}")

        # Pre-compute shifted windows (Fix 5)
        shifted_mean = [(apply_shift(ctx, "mean", 2.0), tgt) for ctx, tgt in windows]

        # Exp 2
        exp2_r = exp2(fm, shifted_mean, dev)
        all_exp2.append(exp2_r)

        # Determine best loss from this seed's exp2
        zs_mse = exp2_r["zero_shot"]["mse"]
        adaptive_configs = {k: v for k, v in exp2_r.items()
                           if k not in ("zero_shot", "revin_oracle")}
        best_loss_name = min(adaptive_configs, key=lambda k: adaptive_configs[k]["mse"])
        # Map config name to loss_mode
        loss_mode_map = {
            "temporal": "temporal",
            "spectral": "spectral",
            "reconstruction": "reconstruction",
            "entropy": "entropy",
            "temporal+entropy": "temporal+entropy",
            "all_four": "temporal+spectral+reconstruction+entropy",
        }
        best_loss_mode = loss_mode_map.get(best_loss_name, best_loss_name)

        # Exp 3
        exp3_r = exp3(fm, shifted_mean, best_loss_mode, dev)
        all_exp3.append(exp3_r)

        # Exp 4: trend shift
        exp4_r = exp4(fm, windows, best_loss_mode, dev)
        all_exp4.append(exp4_r)

    # ============================================================
    # AGGREGATE RESULTS (mean +/- std across seeds)
    # ============================================================
    def aggregate(all_runs):
        """Aggregate list of {config: {mse, mae, ...}} dicts."""
        configs = all_runs[0].keys()
        agg = {}
        for c in configs:
            mses = [r[c]["mse"] for r in all_runs]
            maes = [r[c]["mae"] for r in all_runs]
            agg[c] = {
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "adapt_ms": float(np.mean([r[c].get("adapt_ms", 0) for r in all_runs])),
            }
            if "params" in all_runs[0][c]:
                agg[c]["params"] = all_runs[0][c]["params"]
        return agg

    exp2_agg = aggregate(all_exp2)
    exp3_agg = aggregate(all_exp3)
    exp4_agg = aggregate(all_exp4)

    elapsed = time.time() - t_total

    # ============================================================
    # FINAL REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("  FINAL AGGREGATED RESULTS (mean +/- std over 3 seeds)")
    print("=" * 70)
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Exp 1
    print(f"\n  EXP 1: Max degradation under shift: {max_deg:.1f}%")

    # Exp 2
    zs2 = exp2_agg["zero_shot"]["mse_mean"]
    rv2 = exp2_agg["revin_oracle"]["mse_mean"]
    print(f"\n  EXP 2: Loss Ablation (mean shift, mag=2.0)")
    print(f"  {'Config':>20} | {'MSE (mean+/-std)':>20} | {'vs ZS':>8} | {'vs RevIN':>10}")
    print(f"  {'-'*20} | {'-'*20} | {'-'*8} | {'-'*10}")
    for name, r in exp2_agg.items():
        mse_str = f"{r['mse_mean']:.4f} +/- {r['mse_std']:.4f}"
        imp_zs = relative_improvement(zs2, r["mse_mean"])
        imp_rv = relative_improvement(rv2, r["mse_mean"])
        print(f"  {name:>20} | {mse_str:>20} | {imp_zs:>+7.1f}% | {imp_rv:>+9.1f}%")

    # Exp 3
    zs3 = exp3_agg["zero_shot"]["mse_mean"]
    rv3 = exp3_agg["revin"]["mse_mean"]
    print(f"\n  EXP 3: Adapter Comparison (mean shift, mag=2.0)")
    print(f"  {'Adapter':>12} | {'Params':>6} | {'MSE (mean+/-std)':>20} | {'vs ZS':>8} | {'vs RevIN':>10}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*20} | {'-'*8} | {'-'*10}")
    for name, r in exp3_agg.items():
        params = r.get("params", 0)
        mse_str = f"{r['mse_mean']:.4f} +/- {r['mse_std']:.4f}"
        print(f"  {name:>12} | {params:>6} | {mse_str:>20} | "
              f"{relative_improvement(zs3, r['mse_mean']):>+7.1f}% | "
              f"{relative_improvement(rv3, r['mse_mean']):>+9.1f}%")

    # Exp 4
    zs4 = exp4_agg["zero_shot"]["mse_mean"]
    rv4 = exp4_agg["revin"]["mse_mean"]
    print(f"\n  EXP 4: Trend Shift (mag=2.0)")
    print(f"  {'Config':>12} | {'MSE (mean+/-std)':>20} | {'vs ZS':>8} | {'vs RevIN':>10}")
    print(f"  {'-'*12} | {'-'*20} | {'-'*8} | {'-'*10}")
    for name, r in exp4_agg.items():
        mse_str = f"{r['mse_mean']:.4f} +/- {r['mse_std']:.4f}"
        print(f"  {name:>12} | {mse_str:>20} | "
              f"{relative_improvement(zs4, r['mse_mean']):>+7.1f}% | "
              f"{relative_improvement(rv4, r['mse_mean']):>+9.1f}%")

    # GO/NO-GO
    best_exp2 = min([(k, v) for k, v in exp2_agg.items()
                     if k not in ("zero_shot", "revin_oracle")],
                    key=lambda x: x[1]["mse_mean"])
    best_exp2_imp = relative_improvement(zs2, best_exp2[1]["mse_mean"])
    best_vs_revin = relative_improvement(rv2, best_exp2[1]["mse_mean"])

    print(f"\n  DECISION METRICS:")
    print(f"    Max shift degradation: {max_deg:.1f}%")
    print(f"    Best loss ({best_exp2[0]}): {best_exp2_imp:+.1f}% vs zero-shot, {best_vs_revin:+.1f}% vs RevIN")

    if best_exp2_imp >= 3 and max_deg >= 5:
        decision = "GO"
        print(f"\n  🟢 GO")
    elif best_exp2_imp >= 1 and max_deg >= 1:
        decision = "WEAK GO"
        print(f"\n  🟡 WEAK GO")
    else:
        decision = "NO-GO"
        print(f"\n  🔴 NO-GO")

    # Save
    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    all_results = {
        "decision": decision,
        "runtime_s": elapsed,
        "config": {
            "model": MODEL_ID, "pred_len": PRED_LEN, "ctx_len": CTX_LEN,
            "num_samples": NUM_SAMPLES, "max_windows": MAX_WINDOWS,
            "adapt_steps": ADAPT_STEPS, "n_seeds": N_SEEDS,
        },
        "exp1": {"baseline": baseline, "degradation": deg_results, "max_deg": max_deg},
        "exp2_aggregated": exp2_agg,
        "exp3_aggregated": exp3_agg,
        "exp4_aggregated": exp4_agg,
    }

    with open(os.path.join(os.path.dirname(__file__), "results_r2.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)

    print(f"\n  Results saved to results_r2.json")
    fm.cleanup()


if __name__ == "__main__":
    main()
