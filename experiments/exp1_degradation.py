"""Experiment 1: Does distribution shift actually degrade Chronos-2?

This must be true or the whole project is pointless.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from data.load_etth import load_etth1
from data.synthetic_shift import apply_shift
from models.fm_wrapper import ChronosWrapper
from evaluate import compute_metrics, relative_improvement


def run_exp1():
    print("=" * 60)
    print("  EXPERIMENT 1: Does shift degrade Chronos-2?")
    print("=" * 60)

    # Load data
    data = load_etth1(context_length=512, prediction_length=64)
    windows = data["test_windows"]
    print(f"Loaded ETTh1: {len(windows)} test windows")

    # Load model
    fm = ChronosWrapper(model_id="amazon/chronos-t5-small", device="cuda")

    # Baseline: clean (unshifted) test
    print("\n--- Baseline (no shift) ---")
    all_preds, all_trues = [], []
    for ctx, tgt in windows:
        pred = fm.predict(ctx, prediction_length=64, num_samples=20)
        all_preds.append(pred.squeeze())
        all_trues.append(tgt)

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    baseline = compute_metrics(preds, trues)
    print(f"  Clean MSE: {baseline['mse']:.6f}, MAE: {baseline['mae']:.6f}")

    # Shifted tests
    results = []
    for shift_type in ["mean", "variance", "trend"]:
        for magnitude in [1.0, 2.0, 3.0]:
            all_preds_s, all_trues_s = [], []

            for ctx, tgt in windows:
                # Apply shift to context only (simulates deployment shift)
                ctx_shifted = apply_shift(ctx, shift_type, magnitude)
                pred = fm.predict(ctx_shifted, prediction_length=64, num_samples=20)
                all_preds_s.append(pred.squeeze())
                all_trues_s.append(tgt)  # Ground truth is unshifted targets

            preds_s = np.concatenate(all_preds_s)
            trues_s = np.concatenate(all_trues_s)
            metrics = compute_metrics(preds_s, trues_s)

            degradation = relative_improvement(baseline['mse'], metrics['mse'])
            # Note: negative "improvement" = degradation
            results.append({
                "shift": shift_type,
                "magnitude": magnitude,
                "mse": metrics['mse'],
                "mae": metrics['mae'],
                "degradation_%": -degradation,
            })
            print(f"  {shift_type} mag={magnitude}: MSE={metrics['mse']:.6f} "
                  f"({-degradation:+.1f}% vs clean)")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Clean baseline MSE: {baseline['mse']:.6f}")
    print()
    print(f"  {'Shift':>10} | {'Mag':>5} | {'MSE':>10} | {'Degradation':>12}")
    print(f"  {'-'*10} | {'-'*5} | {'-'*10} | {'-'*12}")
    for r in results:
        print(f"  {r['shift']:>10} | {r['magnitude']:>5.1f} | {r['mse']:>10.6f} | {r['degradation_%']:>+11.1f}%")

    # Decision
    max_deg = max(r['degradation_%'] for r in results)
    print(f"\n  Max degradation: {max_deg:.1f}%")
    if max_deg >= 5:
        print("  ✓ PASS: FMs degrade significantly under shift. Problem exists.")
    elif max_deg >= 1:
        print("  ~ MARGINAL: Some degradation exists but modest.")
    else:
        print("  ✗ FAIL: FMs are robust to these shifts. Try TTFBench or different FM.")

    fm.cleanup()
    return baseline, results


if __name__ == "__main__":
    run_exp1()
