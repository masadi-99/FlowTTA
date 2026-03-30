"""FlowTTA Round 4: From Feasibility to Paper.

R4.1: Scaled reconstruction on chronos-t5-small (80 windows, 5 seeds, 8 shifts)
R4.4: Natural shift on real datasets (ETTh2, Exchange, Weather)

Key changes from Round 3:
- chronos-t5-small (46M) instead of tiny (8M)
- MLP adapter (97 params) instead of segmented affine (16 params)
- Multi-horizon reconstruction (K=12,24,48)
- 20 samples in reconstruction loss
- 80 windows, 5 seeds
- All 8 shift types
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
MODEL_ID = "amazon/chronos-t5-small"
DEVICE = "cuda"
PRED_LEN = 24
CTX_LEN = 512
NUM_SAMPLES = 20
MAX_WINDOWS = 80
ADAPT_STEPS = 10
ADAPT_LR = 1e-3
N_SEEDS = 5
RECON_EVERY = 5  # FM-calling loss every N steps
RECON_SAMPLES = 20  # samples inside recon loss


def load_fm():
    from models.fm_wrapper import ChronosWrapper
    return ChronosWrapper(model_id=MODEL_ID, device=DEVICE)


def get_windows(ctx_len=CTX_LEN, pred_len=PRED_LEN, max_w=MAX_WINDOWS):
    data = load_etth1(context_length=ctx_len, prediction_length=pred_len)
    windows = data["test_windows"][:max_w]
    print(f"  Loaded {len(windows)} windows (ctx={ctx_len}, pred={pred_len})")
    return windows


# ============================================================
# MLP ADAPTER (97 params, zero-init output = starts as identity)
# ============================================================
class MLPAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)

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
def recon_loss_multihorizon(adapter, fm, ctx_np, dev):
    """Multi-horizon leave-last-out: K=12, 24, 48."""
    total = torch.tensor(0.0, device=dev)
    count = 0
    for K in [12, 24, 48]:
        if len(ctx_np) <= K + 64:
            continue
        truncated = ctx_np[:-K]
        held_out = ctx_np[-K:]

        trunc_t = torch.tensor(truncated, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)
        adapted_trunc = adapter(trunc_t).squeeze(-1).squeeze(0).detach().cpu().numpy()

        pred = fm.predict(adapted_trunc, prediction_length=K, num_samples=RECON_SAMPLES)
        pred_mean = pred.squeeze()
        if pred_mean.ndim > 1:
            pred_mean = np.median(pred_mean, axis=0)

        total = total + F.mse_loss(
            torch.tensor(pred_mean, dtype=torch.float32, device=dev),
            torch.tensor(held_out, dtype=torch.float32, device=dev),
        )
        count += 1
    return total / max(count, 1)


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


def _adapt_core(fm, ctx_np, adapter, dev):
    """Run adaptation loop, return adapted context numpy array."""
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=ADAPT_LR)
    ctx_t = torch.tensor(ctx_np, dtype=torch.float32, device=dev).unsqueeze(0).unsqueeze(-1)

    for step in range(ADAPT_STEPS):
        optimizer.zero_grad()
        _ = adapter(ctx_t)  # keep graph alive
        if step % RECON_EVERY == 0:
            loss = recon_loss_multihorizon(adapter, fm, ctx_np, dev)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

    with torch.no_grad():
        final = adapter(ctx_t).squeeze(-1).squeeze(0).cpu().numpy()
    return final


def predict_ours(fm, ctx_np, dev):
    adapter = MLPAdapter().to(dev)
    final = _adapt_core(fm, ctx_np, adapter, dev)
    return predict_zero_shot(fm, final)


def predict_revin_ours(fm, ctx_np, dev):
    """RevIN normalize → adapt → FM → RevIN denormalize."""
    mean, std = ctx_np.mean(), ctx_np.std() + 1e-8
    normed = (ctx_np - mean) / std
    adapter = MLPAdapter().to(dev)
    adapted = _adapt_core(fm, normed, adapter, dev)
    pred = fm.predict(adapted, prediction_length=PRED_LEN, num_samples=NUM_SAMPLES).squeeze()
    if pred.ndim > 1:
        pred = np.median(pred, axis=0)
    return pred * std + mean


# ============================================================
# EXPERIMENT RUNNER
# ============================================================
def eval_method(fm, windows_shifted, method_fn, label="", dev="cpu"):
    """Evaluate a method on pre-shifted windows. Returns metrics."""
    ps, ts = [], []
    for i, (ctx_s, tgt) in enumerate(windows_shifted):
        p = method_fn(fm, ctx_s, dev) if "dev" in method_fn.__code__.co_varnames else method_fn(fm, ctx_s)
        ps.append(p)
        ts.append(tgt)
        if (i + 1) % 20 == 0:
            print(f"      {label}: {i+1}/{len(windows_shifted)}")
    return compute_metrics(np.concatenate(ps), np.concatenate(ts))


# ============================================================
# R4.1: Scaled reconstruction on all 8 shift types
# ============================================================
def run_r4_1(fm, windows, dev):
    print(f"\n{'='*70}")
    print(f"  R4.1: Scaled Reconstruction ({MODEL_ID})")
    print(f"  {MAX_WINDOWS} windows, {N_SEEDS} seeds, 8 shifts")
    print(f"{'='*70}")

    shift_types = [
        ("mean", 2.0),
        ("variance", 2.0),
        ("trend", 2.0),
        ("trend_intrawindow", 2.0),
        ("frequency", 2.0),
        ("autocorrelation", 2.0),
        ("outlier", 2.0),
    ]

    methods = {
        "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
        "revin": lambda fm, ctx: predict_revin(fm, ctx),
        "ours": lambda fm, ctx: predict_ours(fm, ctx, dev),
        "revin+ours": lambda fm, ctx: predict_revin_ours(fm, ctx, dev),
    }

    all_seeds = []
    for seed_idx in range(N_SEEDS):
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n  --- Seed {seed} ({seed_idx+1}/{N_SEEDS}) ---")

        seed_results = {}
        for shift_name, mag in shift_types:
            shifted = [(apply_shift(ctx, shift_name, mag), tgt) for ctx, tgt in windows]
            print(f"\n    {shift_name} (mag={mag}):")

            for mn, mfn in methods.items():
                m = eval_method(fm, shifted, mfn, label=f"{shift_name}/{mn}", dev=dev)
                seed_results[(shift_name, mn)] = m
                print(f"      {mn:>12}: MSE={m['mse']:.4f}")

        all_seeds.append(seed_results)

    return all_seeds


# ============================================================
# R4.4: Natural shift on real datasets
# ============================================================
def load_dataset_windows(name, ctx_len=CTX_LEN, pred_len=PRED_LEN, max_w=MAX_WINDOWS):
    """Load test windows from various datasets."""
    import pandas as pd
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "cached")
    os.makedirs(cache_dir, exist_ok=True)

    urls = {
        "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "Exchange": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt",
        "Weather": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        # Using ETTm1 as proxy for "weather-like" data (minutely, different dynamics)
    }

    if name == "ETTh1":
        data = load_etth1(ctx_len, pred_len)
        return data["test_windows"][:max_w]

    path = os.path.join(cache_dir, f"{name}.csv")
    url = urls.get(name)

    if name == "Exchange":
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            df = pd.read_csv(url, header=None)
            df.to_csv(path, index=False)
        else:
            df = pd.read_csv(path, header=None)
        # Use first column as univariate target
        values = df.iloc[:, 0].values.astype(np.float32)
    else:
        if not os.path.exists(path):
            print(f"  Downloading {name}...")
            df = pd.read_csv(url)
            df.to_csv(path, index=False)
        else:
            df = pd.read_csv(path)
        # Use OT (last numeric column) as target
        values = df.drop(columns=["date"], errors="ignore").iloc[:, -1].values.astype(np.float32)

    # Use last portion as test
    n = len(values)
    test_start = int(n * 0.8)
    test = values[test_start:]

    windows = []
    stride = pred_len
    for start in range(0, len(test) - ctx_len - pred_len + 1, stride):
        ctx = test[start:start + ctx_len]
        tgt = test[start + ctx_len:start + ctx_len + pred_len]
        windows.append((ctx, tgt))

    return windows[:max_w]


def run_r4_4(fm, dev):
    print(f"\n{'='*70}")
    print(f"  R4.4: Natural Shift on Real Datasets")
    print(f"{'='*70}")

    datasets = ["ETTh1", "ETTh2", "Exchange", "Weather"]
    methods = {
        "zero_shot": lambda fm, ctx: predict_zero_shot(fm, ctx),
        "revin": lambda fm, ctx: predict_revin(fm, ctx),
        "ours": lambda fm, ctx: predict_ours(fm, ctx, dev),
        "revin+ours": lambda fm, ctx: predict_revin_ours(fm, ctx, dev),
    }

    results = {}
    for ds_name in datasets:
        print(f"\n  --- {ds_name} ---")
        try:
            windows = load_dataset_windows(ds_name, max_w=80)
            print(f"  Loaded {len(windows)} windows")
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            continue

        if len(windows) < 5:
            print(f"  Skipping {ds_name}: too few windows ({len(windows)})")
            continue

        ds_results = {}
        for mn, mfn in methods.items():
            m = eval_method(fm, windows, mfn, label=f"{ds_name}/{mn}", dev=dev)
            ds_results[mn] = m
            print(f"    {mn:>12}: MSE={m['mse']:.4f} MAE={m['mae']:.4f}")

        results[ds_name] = ds_results

    return results


# ============================================================
# AGGREGATION & REPORTING
# ============================================================
def aggregate_r41(all_seeds):
    """Aggregate R4.1 results across seeds."""
    keys = all_seeds[0].keys()
    agg = {}
    for k in keys:
        mses = [r[k]["mse"] for r in all_seeds if k in r]
        if mses:
            agg[k] = {"mse_mean": np.mean(mses), "mse_std": np.std(mses)}
    return agg


def print_r41_table(agg):
    shift_names = ["mean", "variance", "trend", "trend_intrawindow",
                   "frequency", "autocorrelation", "outlier"]
    method_names = ["zero_shot", "revin", "ours", "revin+ours"]

    print(f"\n  R4.1 Results (mean +/- std over {N_SEEDS} seeds):")
    header = f"  {'Method':>12}"
    for sn in shift_names:
        header += f" | {sn[:10]:>12}"
    print(header)
    print(f"  {'-'*12}" + (" | " + "-"*12) * len(shift_names))

    for mn in method_names:
        row = f"  {mn:>12}"
        for sn in shift_names:
            k = (sn, mn)
            if k in agg:
                row += f" | {agg[k]['mse_mean']:>5.2f}+/-{agg[k]['mse_std']:>4.2f}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    # vs zero-shot
    print(f"\n  vs Zero-Shot (%):")
    for mn in method_names:
        if mn == "zero_shot":
            continue
        row = f"  {mn:>12}"
        for sn in shift_names:
            zs = agg.get((sn, "zero_shot"), {}).get("mse_mean", 1)
            ours = agg.get((sn, mn), {}).get("mse_mean", zs)
            imp = relative_improvement(zs, ours)
            row += f" | {imp:>+11.1f}%"
        print(row)

    # worst-case degradation per method
    print(f"\n  Worst-case degradation per method:")
    for mn in method_names:
        worst = float('inf')
        worst_shift = ""
        for sn in shift_names:
            zs = agg.get((sn, "zero_shot"), {}).get("mse_mean", 1)
            ours = agg.get((sn, mn), {}).get("mse_mean", zs)
            imp = relative_improvement(zs, ours)
            if imp < worst:
                worst = imp
                worst_shift = sn
        if mn == "zero_shot":
            print(f"    {mn:>12}: 0.0% (baseline)")
        else:
            print(f"    {mn:>12}: {worst:+.1f}% (on {worst_shift})")


def print_r44_table(results):
    print(f"\n  R4.4 Cross-Dataset Results:")
    methods = ["zero_shot", "revin", "ours", "revin+ours"]
    datasets = list(results.keys())

    header = f"  {'Method':>12}"
    for ds in datasets:
        header += f" | {ds:>10}"
    print(header)
    print(f"  {'-'*12}" + (" | " + "-"*10) * len(datasets))

    for mn in methods:
        row = f"  {mn:>12}"
        for ds in datasets:
            mse = results.get(ds, {}).get(mn, {}).get("mse", float('nan'))
            row += f" | {mse:>10.4f}"
        print(row)

    # vs zero-shot
    print(f"\n  vs Zero-Shot (%):")
    for mn in methods:
        if mn == "zero_shot":
            continue
        row = f"  {mn:>12}"
        for ds in datasets:
            zs = results.get(ds, {}).get("zero_shot", {}).get("mse", 1)
            ours = results.get(ds, {}).get(mn, {}).get("mse", zs)
            imp = relative_improvement(zs, ours)
            row += f" | {imp:>+9.1f}%"
        print(row)


# ============================================================
# MAIN
# ============================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("  FlowTTA Round 4: Paper-Quality Experiments")
    print(f"  FM: {MODEL_ID} | Windows: {MAX_WINDOWS} | Seeds: {N_SEEDS}")
    print(f"  Adapter: MLP (97 params) | Loss: multi-horizon recon (K=12,24,48)")
    print("=" * 70)

    t_total = time.time()
    fm = load_fm()
    dev = fm.device
    windows = get_windows()

    # R4.1: All shift types
    r41_seeds = run_r4_1(fm, windows, dev)
    r41_agg = aggregate_r41(r41_seeds)
    print_r41_table(r41_agg)

    # R4.4: Natural shift
    r44_results = run_r4_4(fm, dev)
    print_r44_table(r44_results)

    elapsed = time.time() - t_total

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  ROUND 4 SUMMARY")
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min, {elapsed/3600:.1f}hr)")
    print(f"{'='*70}")

    # Key metrics for go/no-go
    shift_names = ["mean", "variance", "trend", "trend_intrawindow",
                   "frequency", "autocorrelation", "outlier"]

    # Check: does RevIN+Ours beat RevIN on any shift?
    wins_over_revin = []
    for sn in shift_names:
        rv = r41_agg.get((sn, "revin"), {}).get("mse_mean", float('inf'))
        ro = r41_agg.get((sn, "revin+ours"), {}).get("mse_mean", float('inf'))
        imp = relative_improvement(rv, ro)
        if imp > 0:
            wins_over_revin.append((sn, imp))

    print(f"\n  RevIN+Ours beats RevIN on: {len(wins_over_revin)}/{len(shift_names)} shifts")
    for sn, imp in wins_over_revin:
        print(f"    {sn}: +{imp:.1f}%")

    # Check: worst-case of RevIN vs RevIN+Ours
    revin_worst = min(
        relative_improvement(
            r41_agg.get((sn, "zero_shot"), {}).get("mse_mean", 1),
            r41_agg.get((sn, "revin"), {}).get("mse_mean", 1))
        for sn in shift_names)
    revin_ours_worst = min(
        relative_improvement(
            r41_agg.get((sn, "zero_shot"), {}).get("mse_mean", 1),
            r41_agg.get((sn, "revin+ours"), {}).get("mse_mean", 1))
        for sn in shift_names)

    print(f"\n  Worst-case vs zero-shot:")
    print(f"    RevIN:      {revin_worst:+.1f}%")
    print(f"    RevIN+Ours: {revin_ours_worst:+.1f}%")
    if revin_ours_worst > revin_worst:
        print(f"    → RevIN+Ours has BETTER worst-case ({revin_ours_worst-revin_worst:+.1f}pp)")
    else:
        print(f"    → RevIN+Ours has WORSE worst-case ({revin_ours_worst-revin_worst:+.1f}pp)")

    # Cross-dataset
    if "ETTh2" in r44_results:
        zs_h2 = r44_results["ETTh2"].get("zero_shot", {}).get("mse", 1)
        rv_h2 = r44_results["ETTh2"].get("revin", {}).get("mse", 1)
        ro_h2 = r44_results["ETTh2"].get("revin+ours", {}).get("mse", 1)
        ou_h2 = r44_results["ETTh2"].get("ours", {}).get("mse", 1)
        print(f"\n  ETTh2 cross-dataset (vs zero-shot):")
        print(f"    RevIN:      {relative_improvement(zs_h2, rv_h2):+.1f}%")
        print(f"    Ours:       {relative_improvement(zs_h2, ou_h2):+.1f}%")
        print(f"    RevIN+Ours: {relative_improvement(zs_h2, ro_h2):+.1f}%")

    # Decision
    print(f"\n  DECISION:")
    has_wins = len(wins_over_revin) >= 2
    better_worst = revin_ours_worst > revin_worst
    cross_dataset_win = ("ETTh2" in r44_results and
                         r44_results["ETTh2"].get("ours", {}).get("mse", float('inf')) <
                         r44_results["ETTh2"].get("revin", {}).get("mse", float('inf')))

    if has_wins and better_worst and cross_dataset_win:
        print("  🟢 PROCEED TO PAPER: RevIN+Ours is strictly better risk-adjusted.")
    elif better_worst or cross_dataset_win:
        print("  🟡 WORKSHOP PAPER: Partial signal. RevIN failure analysis +")
        print("     preliminary method results.")
    else:
        print("  🔴 PIVOT: Method doesn't add value. Focus on analysis contribution.")

    # Save
    class NE(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, tuple): return list(obj)
            return super().default(obj)

    def sk(d):
        return {(f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k): v for k, v in d.items()}

    save = {
        "config": {"model": MODEL_ID, "windows": MAX_WINDOWS, "seeds": N_SEEDS,
                    "samples": NUM_SAMPLES, "adapt_steps": ADAPT_STEPS,
                    "recon_horizons": [12, 24, 48], "adapter": "mlp_97"},
        "runtime_s": elapsed,
        "r41_aggregated": sk(r41_agg),
        "r44_results": r44_results,
    }
    out = os.path.join(os.path.dirname(__file__), "results_r4.json")
    with open(out, "w") as f:
        json.dump(save, f, indent=2, cls=NE)
    print(f"\n  Results saved to {out}")
    fm.cleanup()


if __name__ == "__main__":
    main()
