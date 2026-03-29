"""Run all FlowTTA experiments sequentially."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("  FlowTTA: Full Experiment Pipeline")
    print("  Testing label-free TTA for time series foundation models")
    print("=" * 70)

    results = {}
    t_start = time.time()

    # ============================================================
    # EXPERIMENT 1: Does shift degrade FMs?
    # ============================================================
    print("\n\n" + "#" * 70)
    print("# EXPERIMENT 1: Degradation Analysis")
    print("#" * 70)
    from experiments.exp1_degradation import run_exp1
    baseline, deg_results = run_exp1()
    results["exp1"] = {
        "baseline": baseline,
        "degradation": deg_results,
    }

    # Check if we should continue
    max_deg = max(r['degradation_%'] for r in deg_results)
    if max_deg < 1:
        print("\n⚠ FMs don't degrade enough under shift. Project may not be viable.")
        print("  Continuing experiments anyway for completeness...")

    # ============================================================
    # EXPERIMENT 2: Loss ablation
    # ============================================================
    print("\n\n" + "#" * 70)
    print("# EXPERIMENT 2: Loss Ablation")
    print("#" * 70)
    from experiments.exp2_loss_ablation import run_exp2

    # Test on mean shift (most common) at magnitude 2.0
    exp2_results = run_exp2(shift_type="mean", shift_magnitude=2.0)
    results["exp2"] = exp2_results

    # ============================================================
    # EXPERIMENT 3: Adapter comparison
    # ============================================================
    print("\n\n" + "#" * 70)
    print("# EXPERIMENT 3: Adapter Comparison")
    print("#" * 70)
    from experiments.exp3_adapter_ablation import run_exp3
    exp3_results = run_exp3(shift_type="mean", shift_magnitude=2.0)
    results["exp3"] = exp3_results

    # ============================================================
    # FINAL SUMMARY & GO/NO-GO
    # ============================================================
    elapsed = time.time() - t_start
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY & GO/NO-GO DECISION")
    print("=" * 70)
    print(f"  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Exp 1 summary
    print(f"\n  Exp 1 - Max degradation under shift: {max_deg:.1f}%")
    if max_deg >= 5:
        print("    → Problem exists ✓")
    else:
        print("    → Weak or no degradation ✗")

    # Exp 2 summary
    if "zero_shot" in exp2_results:
        zs_mse = exp2_results["zero_shot"]["mse"]
        best_loss = min(
            [(k, v) for k, v in exp2_results.items() if k != "zero_shot"],
            key=lambda x: x[1]["mse"]
        )
        best_imp = (zs_mse - best_loss[1]["mse"]) / zs_mse * 100
        print(f"\n  Exp 2 - Best loss config: {best_loss[0]} ({best_imp:+.1f}%)")
        if best_imp >= 3:
            print("    → Strong signal from self-supervised losses ✓")
        elif best_imp >= 1:
            print("    → Weak signal, needs better adapter/losses ~")
        else:
            print("    → No signal from self-supervised losses ✗")

    # Exp 3 summary
    if "zero_shot" in exp3_results:
        zs_mse3 = exp3_results["zero_shot"]["mse"]
        best_adapter = min(
            [(k, v) for k, v in exp3_results.items() if k != "zero_shot"],
            key=lambda x: x[1]["mse"]
        )
        best_imp3 = (zs_mse3 - best_adapter[1]["mse"]) / zs_mse3 * 100
        print(f"\n  Exp 3 - Best adapter: {best_adapter[0]} ({best_imp3:+.1f}%)")

    # Final decision
    print("\n" + "-" * 70)
    overall_best_imp = max(best_imp, best_imp3) if 'best_imp3' in dir() else best_imp
    if overall_best_imp >= 3 and max_deg >= 5:
        print("  🟢 GO: Strong results. Proceed to full paper experiments.")
        print("     Next: test on 2nd FM (Moirai), more datasets, write paper.")
    elif overall_best_imp >= 1 and max_deg >= 1:
        print("  🟡 WEAK GO: Some signal exists. Try:")
        print("     - Embedding-level adaptation (if input-level used)")
        print("     - Entropy minimization loss (fallback)")
        print("     - Different FMs or TTFBench datasets")
    else:
        print("  🔴 NO-GO: Insufficient improvement.")
        print("     Consider: entropy loss, output calibration, or abandon.")
    print("-" * 70)

    # Save results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
