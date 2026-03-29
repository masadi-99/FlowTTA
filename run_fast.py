"""Fast FlowTTA feasibility experiment.

Reduced settings for quick go/no-go on GPU with limited memory.
Uses chronos-t5-tiny, pred_len=24, 8 windows, 5 samples.
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.load_etth import load_etth1
from data.synthetic_shift import apply_shift
from evaluate import compute_metrics, relative_improvement


# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "amazon/chronos-t5-tiny"
DEVICE = "cuda"
PRED_LEN = 24
CTX_LEN = 512
NUM_SAMPLES = 5
MAX_WINDOWS = 8  # Limit for speed
ADAPT_STEPS = 10
ADAPT_LR = 1e-3


def load_fm():
    from models.fm_wrapper import ChronosWrapper
    return ChronosWrapper(model_id=MODEL_ID, device=DEVICE)


def get_windows():
    data = load_etth1(context_length=CTX_LEN, prediction_length=PRED_LEN)
    windows = data["test_windows"][:MAX_WINDOWS]
    print(f"Using {len(windows)} test windows (ctx={CTX_LEN}, pred={PRED_LEN})")
    return windows


# ============================================================
# INPUT ADAPTER
# ============================================================
class InputAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.shift = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale + self.shift

    def reset_parameters(self):
        self.scale.data.fill_(1.0)
        self.shift.data.fill_(0.0)


class MLPInputAdapter(torch.nn.Module):
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


# ============================================================
# LOSSES
# ============================================================
def temporal_loss(adapted, original):
    T = adapted.shape[1]
    stride = T // 4
    w1 = adapted[:, :T - stride, :]
    w2 = adapted[:, stride:, :]
    o1 = w1[:, stride:, :]
    o2 = w2[:, :T - 2 * stride, :]
    return F.mse_loss(o1, o2)


def spectral_loss(adapted, original):
    fft_o = torch.fft.rfft(original, dim=1)
    fft_a = torch.fft.rfft(adapted, dim=1)
    psd_o = torch.abs(fft_o) ** 2
    psd_a = torch.abs(fft_a) ** 2
    psd_o = psd_o / (psd_o.sum(dim=1, keepdim=True) + 1e-8)
    psd_a = psd_a / (psd_a.sum(dim=1, keepdim=True) + 1e-8)
    return F.mse_loss(psd_a, psd_o)


def recon_loss(adapted, original):
    T = adapted.shape[1]
    ml = max(1, T // 8)
    ms = torch.randint(0, T - ml, (1,)).item()
    unmasked = torch.cat([adapted[:, :ms], adapted[:, ms+ml:]], dim=1)
    orig_unmasked = torch.cat([original[:, :ms], original[:, ms+ml:]], dim=1)
    return F.mse_loss(unmasked, orig_unmasked)


def combined_loss(adapted, original):
    return 1.0 * temporal_loss(adapted, original) + 0.5 * spectral_loss(adapted, original) + 0.5 * recon_loss(adapted, original)


# ============================================================
# ADAPTATION FUNCTION
# ============================================================
def adapt_predict(fm, ctx_np, adapter, loss_fn, dev="cpu"):
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

    t0 = time.time()
    for _ in range(ADAPT_STEPS):
        optimizer.zero_grad()
        a = adapter(ctx_t)
        loss = loss_fn(a, ctx_t)
        loss.backward()
        optimizer.step()
    adapt_ms = (time.time() - t0) * 1000

    with torch.no_grad():
        final = adapter(ctx_t).squeeze(-1).squeeze(0).cpu().numpy()
    pred = fm.predict(final, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
    return pred.squeeze(), adapt_ms


# ============================================================
# EXPERIMENT 1: Degradation
# ============================================================
def exp1(fm, windows):
    print("\n" + "=" * 60)
    print("  EXP 1: Does shift degrade Chronos?")
    print("=" * 60)

    # Clean baseline
    preds, trues = [], []
    for i, (ctx, tgt) in enumerate(windows):
        p = fm.predict(ctx, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
        preds.append(p.squeeze())
        trues.append(tgt)
        print(f"  Baseline window {i+1}/{len(windows)}")

    baseline = compute_metrics(np.concatenate(preds), np.concatenate(trues))
    print(f"  Clean baseline MSE: {baseline['mse']:.6f}")

    # Shifted
    results = []
    for shift_type in ["mean", "variance", "trend"]:
        for mag in [1.0, 2.0, 3.0]:
            ps, ts = [], []
            for ctx, tgt in windows:
                ctx_s = apply_shift(ctx, shift_type, mag)
                p = fm.predict(ctx_s, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES)
                ps.append(p.squeeze())
                ts.append(tgt)

            m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
            deg = -relative_improvement(baseline['mse'], m['mse'])
            results.append({"shift": shift_type, "mag": mag, "mse": m['mse'], "deg%": deg})
            print(f"  {shift_type} mag={mag}: MSE={m['mse']:.6f} ({deg:+.1f}% degradation)")

    print(f"\n  Max degradation: {max(r['deg%'] for r in results):.1f}%")
    return baseline, results


# ============================================================
# EXPERIMENT 2: Loss Ablation
# ============================================================
def exp2(fm, windows, shift_type="mean", shift_mag=2.0):
    print("\n" + "=" * 60)
    print(f"  EXP 2: Loss Ablation (shift={shift_type}, mag={shift_mag})")
    print("=" * 60)

    dev = fm.device
    configs = {
        "zero_shot": (None, None),
        "temporal": (InputAdapter().to(dev), temporal_loss),
        "spectral": (InputAdapter().to(dev), spectral_loss),
        "reconstruction": (InputAdapter().to(dev), recon_loss),
        "all_three": (InputAdapter().to(dev), combined_loss),
    }

    results = {}
    for name, (adapter, loss_fn) in configs.items():
        ps, ts, total_ms = [], [], 0
        for i, (ctx, tgt) in enumerate(windows):
            ctx_s = apply_shift(ctx, shift_type, shift_mag)
            if adapter is None:
                p = fm.predict(ctx_s, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
                ms = 0
            else:
                p, ms = adapt_predict(fm, ctx_s, adapter, loss_fn, dev)
                total_ms += ms
            ps.append(p)
            ts.append(tgt)

        m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
        results[name] = {"mse": m["mse"], "mae": m["mae"], "adapt_ms": total_ms / max(len(windows), 1)}
        print(f"  {name:>15}: MSE={m['mse']:.6f} MAE={m['mae']:.6f} ({results[name]['adapt_ms']:.0f}ms/batch)")

    zs = results["zero_shot"]["mse"]
    print(f"\n  Zero-shot MSE: {zs:.6f}")
    for name, r in results.items():
        if name != "zero_shot":
            imp = relative_improvement(zs, r["mse"])
            print(f"  {name:>15}: {imp:+.1f}% improvement")

    return results


# ============================================================
# EXPERIMENT 3: Adapter Comparison
# ============================================================
def exp3(fm, windows, shift_type="mean", shift_mag=2.0):
    print("\n" + "=" * 60)
    print(f"  EXP 3: Adapter Comparison (shift={shift_type}, mag={shift_mag})")
    print("=" * 60)

    dev = fm.device
    adapters = {
        "zero_shot": None,
        "affine": InputAdapter().to(dev),
        "mlp": MLPInputAdapter().to(dev),
    }

    results = {}
    for name, adapter in adapters.items():
        n_params = sum(p.numel() for p in adapter.parameters()) if adapter else 0
        ps, ts, total_ms = [], [], 0
        for ctx, tgt in windows:
            ctx_s = apply_shift(ctx, shift_type, shift_mag)
            if adapter is None:
                p = fm.predict(ctx_s, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
                ms = 0
            else:
                p, ms = adapt_predict(fm, ctx_s, adapter, combined_loss, dev)
                total_ms += ms
            ps.append(p)
            ts.append(tgt)

        m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
        results[name] = {"mse": m["mse"], "mae": m["mae"], "params": n_params,
                         "adapt_ms": total_ms / max(len(windows), 1)}
        print(f"  {name:>10} ({n_params:>5} params): MSE={m['mse']:.6f} ({results[name]['adapt_ms']:.0f}ms/batch)")

    zs = results["zero_shot"]["mse"]
    for name, r in results.items():
        if name != "zero_shot":
            print(f"  {name}: {relative_improvement(zs, r['mse']):+.1f}% vs zero-shot")

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  FlowTTA Fast Feasibility Experiment")
    print(f"  Model: {MODEL_ID} | Device: {DEVICE}")
    print(f"  Windows: {MAX_WINDOWS} | Pred: {PRED_LEN} | Samples: {NUM_SAMPLES}")
    print("=" * 60)

    t0 = time.time()
    fm = load_fm()
    windows = get_windows()

    # Exp 1
    baseline, deg_results = exp1(fm, windows)
    max_deg = max(r['deg%'] for r in deg_results)

    # Exp 2
    exp2_results = exp2(fm, windows)

    # Exp 3
    exp3_results = exp3(fm, windows)

    # ============================================================
    # FINAL DECISION
    # ============================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  FINAL GO/NO-GO DECISION")
    print("=" * 60)
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Degradation check
    print(f"\n  1. Degradation under shift: {max_deg:.1f}%")
    if max_deg >= 5:
        print("     ✓ Problem exists - FMs degrade under shift")
    elif max_deg >= 1:
        print("     ~ Marginal degradation")
    else:
        print("     ✗ No degradation - problem doesn't exist for this FM")

    # Best adaptation result
    zs_mse = exp2_results["zero_shot"]["mse"]
    best = min([(k, v) for k, v in exp2_results.items() if k != "zero_shot"],
               key=lambda x: x[1]["mse"])
    best_imp = relative_improvement(zs_mse, best[1]["mse"])
    print(f"\n  2. Best loss ({best[0]}): {best_imp:+.1f}% improvement")

    # Adapter comparison
    zs3 = exp3_results["zero_shot"]["mse"]
    best3 = min([(k, v) for k, v in exp3_results.items() if k != "zero_shot"],
                key=lambda x: x[1]["mse"])
    best3_imp = relative_improvement(zs3, best3[1]["mse"])
    print(f"  3. Best adapter ({best3[0]}): {best3_imp:+.1f}% improvement")

    # Decision
    overall_imp = max(best_imp, best3_imp)
    print(f"\n  Overall best improvement: {overall_imp:+.1f}%")
    print()

    if overall_imp >= 3 and max_deg >= 5:
        decision = "GO"
        print("  🟢 GO: Strong results. Proceed to full NeurIPS paper.")
    elif overall_imp >= 1 and max_deg >= 1:
        decision = "WEAK GO"
        print("  🟡 WEAK GO: Some signal. Try embedding-level adaptation,")
        print("     entropy loss, or different FM. Worth exploring further.")
    elif max_deg < 1:
        decision = "NO-GO (no degradation)"
        print("  🔴 NO-GO: FMs don't degrade under shift for this setup.")
        print("     Try TTFBench datasets or different shift types.")
    else:
        decision = "NO-GO"
        print("  🔴 NO-GO: Self-supervised losses don't improve predictions.")
        print("     Try entropy minimization or output calibration (fallbacks).")

    # Save results
    all_results = {
        "decision": decision,
        "exp1_max_degradation": float(max_deg),
        "exp2_best_loss": best[0],
        "exp2_best_improvement": float(best_imp),
        "exp3_best_adapter": best3[0],
        "exp3_best_improvement": float(best3_imp),
        "config": {
            "model": MODEL_ID,
            "pred_len": PRED_LEN,
            "ctx_len": CTX_LEN,
            "num_samples": NUM_SAMPLES,
            "max_windows": MAX_WINDOWS,
            "adapt_steps": ADAPT_STEPS,
        },
        "exp1": {"baseline": baseline, "degradation": deg_results},
        "exp2": {k: v for k, v in exp2_results.items()},
        "exp3": {k: v for k, v in exp3_results.items()},
    }

    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, cls=NE)

    print(f"\n  Results saved to results.json")
    fm.cleanup()


if __name__ == "__main__":
    main()
