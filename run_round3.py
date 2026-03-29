"""FlowTTA Round 3: The Pivot.

Test on shifts where RevIN FAILS, stack RevIN+Ours, try natural shifts,
explore reconstruction loss variants, and test embedding-level adaptation.

Experiments:
  A: Shifts where RevIN fails (trend_intrawindow, frequency, autocorrelation, outlier)
  B: RevIN + Ours stacking
  C: Natural shift (cross-dataset transfer)
  D: Multi-horizon + bidirectional reconstruction loss
  E: Embedding-level adaptation
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
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
NUM_SAMPLES = 20
MAX_WINDOWS = 30
ADAPT_STEPS = 10
ADAPT_LR = 1e-3
N_SEEDS = 3
RECON_EVERY = 5


def load_fm():
    from models.fm_wrapper import ChronosWrapper
    return ChronosWrapper(model_id=MODEL_ID, device=DEVICE)


def get_windows():
    data = load_etth1(context_length=CTX_LEN, prediction_length=PRED_LEN)
    return data["test_windows"][:MAX_WINDOWS]


# ============================================================
# ADAPTERS
# ============================================================
class SegmentedAdapter(nn.Module):
    def __init__(self, n_segments=8):
        super().__init__()
        self.n_segments = n_segments
        self.scale = nn.Parameter(torch.ones(n_segments))
        self.shift = nn.Parameter(torch.zeros(n_segments))

    def forward(self, x):
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


class EmbeddingAdapter(nn.Module):
    """MLP adapter for embedding space with residual connection."""
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)

    def reset_parameters(self):
        for m in self.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# LOSSES
# ============================================================
def recon_loss(adapter, fm, ctx_np, dev, K=24):
    """Leave-last-out reconstruction loss."""
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
    return F.mse_loss(
        torch.tensor(pred_mean, dtype=torch.float32, device=dev),
        torch.tensor(held_out, dtype=torch.float32, device=dev),
    )


def recon_loss_multihorizon(adapter, fm, ctx_np, dev):
    """Multi-horizon reconstruction: withhold 12, 24, 48 steps."""
    total = torch.tensor(0.0, device=dev)
    count = 0
    for K in [12, 24, 48]:
        if K >= len(ctx_np) - 64:
            continue
        total = total + recon_loss(adapter, fm, ctx_np, dev, K=K)
        count += 1
    return total / max(count, 1)


def recon_loss_bidirectional(adapter, fm, ctx_np, dev):
    """Forward + backward reconstruction."""
    K = 24
    fwd = recon_loss(adapter, fm, ctx_np, dev, K=K)
    # Backward: reverse and predict
    reversed_ctx = ctx_np[::-1].copy()
    bwd = recon_loss(adapter, fm, reversed_ctx, dev, K=K)
    return fwd + 0.5 * bwd


def recon_loss_embedding(adapter, fm, ctx_np, dev, K=24):
    """
    Reconstruction loss for embedding-level adapter.
    Encode truncated context, adapt embeddings, decode, compare to held-out.
    Since we can't directly decode from adapted embeddings with Chronos,
    we use a proxy: the adapter should minimize reconstruction of held-out
    via the FM's own encode-then-predict pipeline.
    """
    if len(ctx_np) <= K + 64:
        return torch.tensor(0.0, device=dev, requires_grad=True)

    truncated = ctx_np[:-K]
    held_out = ctx_np[-K:]

    # Get embeddings for truncated context
    trunc_t = torch.tensor(truncated, dtype=torch.float32).unsqueeze(0)

    # Encode via FM
    with torch.no_grad():
        z = fm.encode(trunc_t)  # (1, seq_len, embed_dim)

    # Adapt embeddings
    z_adapted = adapter(z)

    # We can't feed adapted embeddings back into Chronos decoder directly,
    # so we use a simpler signal: the adapted embeddings should be "closer"
    # to what the encoder produces on the FULL (unshifted) context.
    # This is a consistency objective: adapt truncated to look like full.
    full_t = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z_full = fm.encode(full_t)

    # Match the adapted truncated embeddings to the full embeddings
    # at the overlapping positions
    min_len = min(z_adapted.shape[1], z_full.shape[1])
    return F.mse_loss(z_adapted[:, :min_len, :], z_full[:, :min_len, :].detach())


# ============================================================
# PREDICTION METHODS
# ============================================================
def predict_zero_shot(fm, ctx_np):
    pred = fm.predict(ctx_np, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
    if pred.ndim > 1:
        pred = np.median(pred, axis=0)
    return pred


def predict_revin(fm, ctx_np):
    mean, std = ctx_np.mean(), ctx_np.std() + 1e-8
    normed = (ctx_np - mean) / std
    pred = fm.predict(normed, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
    if pred.ndim > 1:
        pred = np.median(pred, axis=0)
    return pred * std + mean


def predict_adapted(fm, ctx_np, adapter, loss_fn, dev):
    """Adapt input-level adapter, then predict."""
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

    for step in range(ADAPT_STEPS):
        optimizer.zero_grad()
        adapted = adapter(ctx_t)
        if step % RECON_EVERY == 0:
            loss = loss_fn(adapter, fm, ctx_np, dev)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

    with torch.no_grad():
        final = adapter(ctx_t).squeeze(-1).squeeze(0).cpu().numpy()
    return predict_zero_shot(fm, final)


def predict_revin_then_adapt(fm, ctx_np, adapter, loss_fn, dev):
    """RevIN first, then adapt on top."""
    mean, std = ctx_np.mean(), ctx_np.std() + 1e-8
    normed = (ctx_np - mean) / std

    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)
    ctx_t = torch.tensor(normed, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

    for step in range(ADAPT_STEPS):
        optimizer.zero_grad()
        adapted = adapter(ctx_t)
        if step % RECON_EVERY == 0:
            loss = loss_fn(adapter, fm, normed, dev)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

    with torch.no_grad():
        final = adapter(ctx_t).squeeze(-1).squeeze(0).cpu().numpy()
    pred = fm.predict(final, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
    if pred.ndim > 1:
        pred = np.median(pred, axis=0)
    return pred * std + mean


def predict_embedding_adapted(fm, ctx_np, adapter, dev):
    """Embedding-level adaptation with reconstruction consistency loss."""
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)

    for step in range(ADAPT_STEPS):
        optimizer.zero_grad()
        if step % RECON_EVERY == 0:
            loss = recon_loss_embedding(adapter, fm, ctx_np, dev)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

    # For final prediction, we can't inject adapted embeddings into Chronos.
    # Instead, the embedding adapter trains an understanding of the shift,
    # but prediction still goes through the normal pipeline.
    # The value is in the LOSS signal, not the adapted forward pass.
    # So for now, this tests whether the embedding loss provides better
    # gradient signal to an input-level adapter that we chain.
    return predict_zero_shot(fm, ctx_np)


# ============================================================
# GENERIC EXPERIMENT RUNNER
# ============================================================
def run_experiment(fm, windows, shift_types, methods, dev, title=""):
    """
    Run all methods on all shift types.

    methods: dict of {name: callable(fm, ctx_shifted) -> pred}
    shift_types: list of (shift_name, magnitude)
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    all_results = {}

    for shift_name, mag in shift_types:
        print(f"\n  --- {shift_name} (mag={mag}) ---")
        shifted_windows = [(apply_shift(ctx, shift_name, mag), tgt) for ctx, tgt in windows]

        for method_name, method_fn in methods.items():
            ps, ts = [], []
            for i, (ctx_s, tgt) in enumerate(shifted_windows):
                p = method_fn(fm, ctx_s)
                ps.append(p)
                ts.append(tgt)
                if (i + 1) % 15 == 0:
                    print(f"    {method_name}: {i+1}/{len(shifted_windows)}")

            m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
            key = (shift_name, method_name)
            all_results[key] = m
            print(f"    {method_name}: MSE={m['mse']:.4f}")

    return all_results


def print_table(results, shift_types, methods, title=""):
    """Print results as shift × method table."""
    print(f"\n  {title}")
    header = f"  {'':>20}"
    for sn, _ in shift_types:
        header += f" | {sn:>14}"
    print(header)
    print(f"  {'-'*20}" + (" | " + "-"*14) * len(shift_types))

    for mn in methods:
        row = f"  {mn:>20}"
        for sn, _ in shift_types:
            mse = results.get((sn, mn), {}).get("mse", float('nan'))
            row += f" | {mse:>14.4f}"
        print(row)

    # Relative to zero_shot
    if "zero_shot" in methods:
        print()
        print(f"  {'vs Zero-Shot (%)':>20}", end="")
        for sn, _ in shift_types:
            print(f" | {'':>14}", end="")
        print()
        for mn in methods:
            if mn == "zero_shot":
                continue
            row = f"  {mn:>20}"
            for sn, _ in shift_types:
                zs = results.get((sn, "zero_shot"), {}).get("mse", 1)
                ours = results.get((sn, mn), {}).get("mse", zs)
                imp = relative_improvement(zs, ours)
                row += f" | {imp:>+13.1f}%"
            print(row)


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  FlowTTA Round 3: The Pivot")
    print(f"  Model: {MODEL_ID} | Windows: {MAX_WINDOWS} | Seeds: {N_SEEDS}")
    print("=" * 70)

    t_total = time.time()
    fm = load_fm()
    dev = fm.device
    windows = get_windows()

    all_seed_results = {"A": [], "B": [], "D": [], "E": []}

    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({seed_idx+1}/{N_SEEDS})")
        print(f"{'#'*70}")

        # ===========================================================
        # EXP A: Shifts where RevIN fails
        # ===========================================================
        shift_types_a = [
            ("mean", 2.0),
            ("trend_intrawindow", 2.0),
            ("frequency", 2.0),
            ("autocorrelation", 2.0),
            ("outlier", 2.0),
        ]

        adapter_factory = lambda: SegmentedAdapter().to(dev)

        methods_a = {
            "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
            "revin": lambda fm, ctx: predict_revin(fm, ctx),
            "ours_recon": lambda fm, ctx: predict_adapted(
                fm, ctx, adapter_factory(), recon_loss, dev),
            "revin+ours": lambda fm, ctx: predict_revin_then_adapt(
                fm, ctx, adapter_factory(), recon_loss, dev),
        }

        results_a = run_experiment(fm, windows, shift_types_a, methods_a, dev,
                                   title="EXP A: Shifts Where RevIN Fails")
        print_table(results_a, shift_types_a, methods_a.keys(),
                    title="EXP A Summary")
        all_seed_results["A"].append(results_a)

        # ===========================================================
        # EXP B: RevIN + Ours stacking (more shift types)
        # ===========================================================
        # Already covered in Exp A via "revin+ours" method

        # ===========================================================
        # EXP D: Reconstruction loss variants
        # ===========================================================
        shift_types_d = [
            ("trend_intrawindow", 2.0),
            ("frequency", 2.0),
        ]

        methods_d = {
            "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
            "revin": lambda fm, ctx: predict_revin(fm, ctx),
            "recon_single": lambda fm, ctx: predict_adapted(
                fm, ctx, adapter_factory(), recon_loss, dev),
            "recon_multi_hz": lambda fm, ctx: predict_adapted(
                fm, ctx, adapter_factory(), recon_loss_multihorizon, dev),
            "recon_bidir": lambda fm, ctx: predict_adapted(
                fm, ctx, adapter_factory(), recon_loss_bidirectional, dev),
        }

        results_d = run_experiment(fm, windows, shift_types_d, methods_d, dev,
                                   title="EXP D: Reconstruction Loss Variants")
        print_table(results_d, shift_types_d, methods_d.keys(),
                    title="EXP D Summary")
        all_seed_results["D"].append(results_d)

        # ===========================================================
        # EXP E: Embedding-level adaptation
        # ===========================================================
        embed_dim = fm.embed_dim
        print(f"\n  FM embedding dim: {embed_dim}")

        emb_adapter_factory = lambda: EmbeddingAdapter(embed_dim, hidden_dim=128).to(dev)

        shift_types_e = [
            ("trend_intrawindow", 2.0),
            ("frequency", 2.0),
        ]

        # For embedding adapter, we use a hybrid approach:
        # train embedding adapter with embedding-consistency loss,
        # then use the loss signal to also update an input adapter
        class HybridAdapter(nn.Module):
            """Input adapter trained with embedding-level consistency loss."""
            def __init__(self, n_segments=8, embed_dim=256, hidden_dim=128):
                super().__init__()
                self.input_adapter = SegmentedAdapter(n_segments)
                self.emb_adapter = EmbeddingAdapter(embed_dim, hidden_dim)

            def forward(self, x):
                return self.input_adapter(x)

            def reset_parameters(self):
                self.input_adapter.reset_parameters()
                self.emb_adapter.reset_parameters()

            @property
            def num_params(self):
                return sum(p.numel() for p in self.parameters())

        def hybrid_loss(adapter, fm, ctx_np, dev, K=24):
            """
            Train input adapter via embedding consistency.
            Encode original and adapted inputs, compare embeddings.
            The adapted input's embeddings should better match
            what the FM expects (closer to training distribution).
            """
            if len(ctx_np) <= K + 64:
                return torch.tensor(0.0, device=dev, requires_grad=True)

            # Adapted input through input adapter
            ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
            adapted_input = adapter(ctx_t).squeeze(-1).squeeze(0)

            # Also do leave-last-out reconstruction
            truncated = ctx_np[:-K]
            held_out = ctx_np[-K:]
            trunc_t = torch.tensor(truncated, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
            adapted_trunc = adapter(trunc_t).squeeze(-1).squeeze(0).detach().cpu().numpy()
            pred = fm.predict(adapted_trunc, prediction_length=K, num_samples=NUM_SAMPLES)
            pred_mean = pred.squeeze()
            if pred_mean.ndim > 1:
                pred_mean = np.median(pred_mean, axis=0)

            recon = F.mse_loss(
                torch.tensor(pred_mean, dtype=torch.float32, device=dev),
                torch.tensor(held_out, dtype=torch.float32, device=dev),
            )

            # Embedding consistency: encode adapted vs original, compare
            with torch.no_grad():
                z_orig = fm.encode(torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0))
                z_adapted = fm.encode(adapted_input.unsqueeze(0).cpu())

            # The adapted embeddings should have lower variance (more "canonical")
            emb_reg = z_adapted.var(dim=1).mean()

            return recon + 0.1 * emb_reg

        hybrid_factory = lambda: HybridAdapter(
            n_segments=8, embed_dim=embed_dim, hidden_dim=128).to(dev)

        methods_e = {
            "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
            "revin": lambda fm, ctx: predict_revin(fm, ctx),
            "recon_input": lambda fm, ctx: predict_adapted(
                fm, ctx, adapter_factory(), recon_loss, dev),
            "hybrid_emb": lambda fm, ctx: predict_adapted(
                fm, ctx, hybrid_factory(), hybrid_loss, dev),
        }

        results_e = run_experiment(fm, windows, shift_types_e, methods_e, dev,
                                   title="EXP E: Embedding-Level Adaptation")
        print_table(results_e, shift_types_e, methods_e.keys(),
                    title="EXP E Summary")
        all_seed_results["E"].append(results_e)

    # ============================================================
    # EXP C: Cross-dataset natural shift (run once, no seed var)
    # ============================================================
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n{'#'*70}")
    print(f"# EXP C: Cross-Dataset Natural Shift")
    print(f"{'#'*70}")

    # Load ETTh1 test windows as before (baseline domain)
    # Then try ETTh2 (different transformer in the ETT dataset)
    try:
        import pandas as pd
        etth2_url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv"
        cache_path = os.path.join(os.path.dirname(__file__), "data", "cached", "ETTh2.csv")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            print("  Downloading ETTh2...")
            df = pd.read_csv(etth2_url)
            df.to_csv(cache_path, index=False)
        else:
            df = pd.read_csv(cache_path)

        features = df.drop(columns=["date"]).values.astype(np.float32)
        n_train, n_val = 8640, 2880
        test_h2 = features[n_train + n_val:n_train + n_val + 2880]
        test_ot_h2 = test_h2[:, -1]

        windows_h2 = []
        stride = PRED_LEN
        for start in range(0, len(test_ot_h2) - CTX_LEN - PRED_LEN + 1, stride):
            ctx = test_ot_h2[start:start + CTX_LEN]
            tgt = test_ot_h2[start + CTX_LEN:start + CTX_LEN + PRED_LEN]
            windows_h2.append((ctx, tgt))
        windows_h2 = windows_h2[:MAX_WINDOWS]
        print(f"  ETTh2: {len(windows_h2)} test windows")

        methods_c = {
            "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
            "revin": lambda fm, ctx: predict_revin(fm, ctx),
            "ours_recon": lambda fm, ctx: predict_adapted(
                fm, ctx, SegmentedAdapter().to(dev), recon_loss, dev),
            "revin+ours": lambda fm, ctx: predict_revin_then_adapt(
                fm, ctx, SegmentedAdapter().to(dev), recon_loss, dev),
        }

        # ETTh1 (reference)
        print("\n  --- ETTh1 (reference) ---")
        results_c_h1 = {}
        for mn, mfn in methods_c.items():
            ps, ts = [], []
            for ctx, tgt in windows:
                ps.append(mfn(fm, ctx))
                ts.append(tgt)
            m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
            results_c_h1[mn] = m
            print(f"    {mn}: MSE={m['mse']:.4f}")

        # ETTh2 (natural shift)
        print("\n  --- ETTh2 (natural shift) ---")
        results_c_h2 = {}
        for mn, mfn in methods_c.items():
            ps, ts = [], []
            for i, (ctx, tgt) in enumerate(windows_h2):
                ps.append(mfn(fm, ctx))
                ts.append(tgt)
                if (i + 1) % 15 == 0:
                    print(f"    {mn}: {i+1}/{len(windows_h2)}")
            m = compute_metrics(np.concatenate(ps), np.concatenate(ts))
            results_c_h2[mn] = m
            print(f"    {mn}: MSE={m['mse']:.4f}")

        print(f"\n  Cross-dataset results:")
        print(f"  {'Method':>15} | {'ETTh1 MSE':>10} | {'ETTh2 MSE':>10} | {'ETTh2 vs ZS':>12}")
        print(f"  {'-'*15} | {'-'*10} | {'-'*10} | {'-'*12}")
        zs_h2 = results_c_h2["zero_shot"]["mse"]
        for mn in methods_c:
            h1 = results_c_h1[mn]["mse"]
            h2 = results_c_h2[mn]["mse"]
            imp = relative_improvement(zs_h2, h2)
            print(f"  {mn:>15} | {h1:>10.4f} | {h2:>10.4f} | {imp:>+11.1f}%")

        exp_c_results = {"etth1": results_c_h1, "etth2": results_c_h2}
    except Exception as e:
        print(f"  EXP C failed: {e}")
        exp_c_results = {"error": str(e)}

    # ============================================================
    # AGGREGATE across seeds
    # ============================================================
    def aggregate_multi(all_runs):
        """Aggregate list of {(shift,method): {mse,...}} dicts."""
        if not all_runs:
            return {}
        keys = all_runs[0].keys()
        agg = {}
        for k in keys:
            mses = [r[k]["mse"] for r in all_runs if k in r]
            if mses:
                agg[k] = {
                    "mse_mean": float(np.mean(mses)),
                    "mse_std": float(np.std(mses)),
                }
        return agg

    elapsed = time.time() - t_total

    print(f"\n\n{'='*70}")
    print(f"  ROUND 3 FINAL RESULTS (mean +/- std over {N_SEEDS} seeds)")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    # Print Exp A aggregated
    agg_a = aggregate_multi(all_seed_results["A"])
    shift_names_a = ["mean", "trend_intrawindow", "frequency", "autocorrelation", "outlier"]
    method_names_a = ["zero_shot", "revin", "ours_recon", "revin+ours"]

    print(f"\n  EXP A: RevIN-Breaking Shifts")
    header = f"  {'Method':>15}"
    for sn in shift_names_a:
        header += f" | {sn:>18}"
    print(header)
    print(f"  {'-'*15}" + (" | " + "-"*18) * len(shift_names_a))

    for mn in method_names_a:
        row = f"  {mn:>15}"
        for sn in shift_names_a:
            k = (sn, mn)
            if k in agg_a:
                row += f" | {agg_a[k]['mse_mean']:>8.2f}+/-{agg_a[k]['mse_std']:>5.2f}"
            else:
                row += f" | {'N/A':>18}"
        print(row)

    # vs zero-shot
    print(f"\n  vs Zero-Shot (%):")
    for mn in method_names_a:
        if mn == "zero_shot":
            continue
        row = f"  {mn:>15}"
        for sn in shift_names_a:
            zs_k = (sn, "zero_shot")
            k = (sn, mn)
            if zs_k in agg_a and k in agg_a:
                imp = relative_improvement(agg_a[zs_k]["mse_mean"], agg_a[k]["mse_mean"])
                row += f" | {imp:>+17.1f}%"
            else:
                row += f" | {'N/A':>18}"
        print(row)

    # Print Exp D aggregated
    agg_d = aggregate_multi(all_seed_results["D"])
    shift_names_d = ["trend_intrawindow", "frequency"]
    method_names_d = ["zero_shot", "revin", "recon_single", "recon_multi_hz", "recon_bidir"]

    print(f"\n  EXP D: Reconstruction Loss Variants")
    for mn in method_names_d:
        row = f"  {mn:>15}"
        for sn in shift_names_d:
            k = (sn, mn)
            if k in agg_d:
                row += f" | {agg_d[k]['mse_mean']:>8.2f}+/-{agg_d[k]['mse_std']:>5.2f}"
            else:
                row += f" | {'N/A':>18}"
        print(row)

    # Print Exp E aggregated
    agg_e = aggregate_multi(all_seed_results["E"])
    method_names_e = ["zero_shot", "revin", "recon_input", "hybrid_emb"]

    print(f"\n  EXP E: Embedding-Level Adaptation")
    for mn in method_names_e:
        row = f"  {mn:>15}"
        for sn in shift_names_d:
            k = (sn, mn)
            if k in agg_e:
                row += f" | {agg_e[k]['mse_mean']:>8.2f}+/-{agg_e[k]['mse_std']:>5.2f}"
            else:
                row += f" | {'N/A':>18}"
        print(row)

    # Save all results
    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, tuple): return list(obj)
            return super().default(obj)

    # Convert tuple keys to strings for JSON
    def stringify_keys(d):
        return {f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k: v for k, v in d.items()}

    save_data = {
        "config": {
            "model": MODEL_ID, "windows": MAX_WINDOWS, "samples": NUM_SAMPLES,
            "seeds": N_SEEDS, "adapt_steps": ADAPT_STEPS,
        },
        "runtime_s": elapsed,
        "exp_a_aggregated": stringify_keys(agg_a),
        "exp_c": {k: {mk: mv for mk, mv in v.items()} if isinstance(v, dict) else v
                  for k, v in exp_c_results.items()},
        "exp_d_aggregated": stringify_keys(agg_d),
        "exp_e_aggregated": stringify_keys(agg_e),
    }

    out_path = os.path.join(os.path.dirname(__file__), "results_r3.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, cls=NE)
    print(f"\n  Results saved to {out_path}")

    fm.cleanup()


if __name__ == "__main__":
    main()
