"""Experiment 2: Does each self-supervised loss help independently?

Test each loss alone with the Affine adapter on shifted ETTh1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import time
from data.load_etth import load_etth1
from data.synthetic_shift import apply_shift
from models.fm_wrapper import ChronosWrapper
from models.adapters import InputAdapter
from evaluate import compute_metrics, relative_improvement


def adapt_with_input_adapter(fm, context, prediction_length, adapter, loss_fn,
                              adapt_steps=10, adapt_lr=1e-3):
    """
    Adapt using input-level adapter (more robust than embedding-level).

    Args:
        fm: ChronosWrapper
        context: numpy array (T,)
        prediction_length: int
        adapter: InputAdapter module
        loss_fn: function(adapted_context_tensor, original_context_tensor) -> loss
        adapt_steps: number of optimization steps
        adapt_lr: learning rate

    Returns:
        prediction: numpy array (prediction_length,)
        adapt_time_ms: adaptation time in milliseconds
    """
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=adapt_lr)

    ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    # (1, T, 1)

    t_start = time.time()

    for step in range(adapt_steps):
        optimizer.zero_grad()
        adapted_ctx = adapter(ctx_tensor)  # (1, T, 1)
        loss = loss_fn(adapted_ctx, ctx_tensor)
        loss.backward()
        optimizer.step()

    adapt_time_ms = (time.time() - t_start) * 1000

    # Predict with adapted input
    with torch.no_grad():
        adapted_ctx_final = adapter(ctx_tensor).squeeze(-1).squeeze(0)
        pred = fm.predict(adapted_ctx_final.cpu().numpy(), prediction_length=prediction_length, num_samples=20)

    return pred.squeeze(), adapt_time_ms


def temporal_loss_fn(adapted_ctx, original_ctx):
    """
    Temporal consistency: overlapping sub-windows should have consistent adapted outputs.
    Split context into two overlapping halves and enforce agreement.
    """
    T = adapted_ctx.shape[1]
    stride = T // 4
    w1 = adapted_ctx[:, :T - stride, :]
    w2 = adapted_ctx[:, stride:, :]
    # After adaptation, the overlapping region should be self-consistent
    # We enforce that the adapter produces smooth transformations
    overlap1 = w1[:, stride:, :]
    overlap2 = w2[:, :T - 2 * stride, :]
    return F.mse_loss(overlap1, overlap2)


def spectral_loss_fn(adapted_ctx, original_ctx):
    """Spectral consistency: preserve frequency structure."""
    fft_orig = torch.fft.rfft(original_ctx, dim=1)
    fft_adapt = torch.fft.rfft(adapted_ctx, dim=1)

    psd_orig = torch.abs(fft_orig) ** 2
    psd_adapt = torch.abs(fft_adapt) ** 2

    psd_orig = psd_orig / (psd_orig.sum(dim=1, keepdim=True) + 1e-8)
    psd_adapt = psd_adapt / (psd_adapt.sum(dim=1, keepdim=True) + 1e-8)

    return F.mse_loss(psd_adapt, psd_orig)


def reconstruction_loss_fn(adapted_ctx, original_ctx):
    """
    Masked reconstruction: mask part of input, check if adapter preserves it.
    The adapter should map close to the original for unmasked parts,
    while fixing the distribution for masked parts.
    """
    T = adapted_ctx.shape[1]
    mask_len = max(1, T // 8)
    mask_start = torch.randint(0, T - mask_len, (1,)).item()

    # Masked version
    masked = original_ctx.clone()
    masked[:, mask_start:mask_start + mask_len, :] = 0

    # The adapted FULL context at masked positions should differ from zero
    # (adapter should "fill in" reasonable values)
    # But also stay close to original at unmasked positions
    loss_unmasked = F.mse_loss(
        torch.cat([adapted_ctx[:, :mask_start], adapted_ctx[:, mask_start + mask_len:]], dim=1),
        torch.cat([original_ctx[:, :mask_start], original_ctx[:, mask_start + mask_len:]], dim=1),
    )
    return loss_unmasked


def entropy_loss_fn_factory(fm, prediction_length=64):
    """Factory for entropy-based loss (fallback)."""
    def entropy_loss(adapted_ctx, original_ctx):
        # Get forecast samples
        with torch.no_grad():
            ctx_np = adapted_ctx.squeeze(-1).squeeze(0).detach().cpu().numpy()
            samples = fm.predict_with_quantiles(ctx_np, prediction_length=prediction_length, num_samples=20)
        # Minimize spread
        spread = np.std(samples, axis=1).mean()
        return torch.tensor(spread, requires_grad=True, device=adapted_ctx.device)
    return entropy_loss


def combined_loss_fn(adapted_ctx, original_ctx, weights=(1.0, 0.5, 0.5)):
    """All three losses combined."""
    l_temp = temporal_loss_fn(adapted_ctx, original_ctx)
    l_spec = spectral_loss_fn(adapted_ctx, original_ctx)
    l_recon = reconstruction_loss_fn(adapted_ctx, original_ctx)
    return weights[0] * l_temp + weights[1] * l_spec + weights[2] * l_recon


def run_exp2(shift_type="mean", shift_magnitude=2.0):
    print("=" * 60)
    print("  EXPERIMENT 2: Loss Ablation Study")
    print(f"  Shift: {shift_type}, Magnitude: {shift_magnitude}")
    print("=" * 60)

    data = load_etth1(context_length=512, prediction_length=64)
    windows = data["test_windows"]
    print(f"Loaded ETTh1: {len(windows)} test windows")

    fm = ChronosWrapper(model_id="amazon/chronos-t5-small", device="cuda")

    # Test configurations
    configs = {
        "zero_shot": None,
        "temporal_only": temporal_loss_fn,
        "spectral_only": spectral_loss_fn,
        "reconstruction_only": reconstruction_loss_fn,
        "all_three": lambda a, o: combined_loss_fn(a, o),
    }

    results = {}

    for config_name, loss_fn in configs.items():
        print(f"\n--- {config_name} ---")
        all_preds, all_trues = [], []
        total_adapt_time = 0

        adapter = InputAdapter(n_features=1).to(fm.device) if loss_fn is not None else None

        for i, (ctx, tgt) in enumerate(windows):
            # Apply shift to context
            ctx_shifted = apply_shift(ctx, shift_type, shift_magnitude)

            if loss_fn is None:
                # Zero-shot baseline
                pred = fm.predict(ctx_shifted, prediction_length=64, num_samples=20).squeeze()
                adapt_time = 0
            else:
                pred, adapt_time = adapt_with_input_adapter(
                    fm, ctx_shifted, 96, adapter, loss_fn,
                    adapt_steps=10, adapt_lr=1e-3,
                )
                total_adapt_time += adapt_time

            all_preds.append(pred)
            all_trues.append(tgt)

            if (i + 1) % 5 == 0:
                print(f"  Window {i+1}/{len(windows)}")

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        metrics = compute_metrics(preds, trues)
        avg_time = total_adapt_time / max(len(windows), 1)

        results[config_name] = {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "avg_adapt_ms": avg_time,
        }
        print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}, Avg adapt: {avg_time:.1f}ms")

    # Summary
    baseline_mse = results["zero_shot"]["mse"]
    print("\n" + "=" * 60)
    print("  LOSS ABLATION SUMMARY")
    print("=" * 60)
    print(f"  {'Config':>25} | {'MSE':>10} | {'MAE':>10} | {'Improv%':>8} | {'ms/batch':>8}")
    print(f"  {'-'*25} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
    for name, r in results.items():
        imp = relative_improvement(baseline_mse, r["mse"])
        print(f"  {name:>25} | {r['mse']:>10.6f} | {r['mae']:>10.6f} | {imp:>+7.1f}% | {r['avg_adapt_ms']:>7.1f}")

    # Decision
    best_name = min(
        [(k, v) for k, v in results.items() if k != "zero_shot"],
        key=lambda x: x[1]["mse"]
    )[0]
    best_imp = relative_improvement(baseline_mse, results[best_name]["mse"])

    print(f"\n  Best config: {best_name} ({best_imp:+.1f}% improvement)")
    if best_imp >= 3:
        print("  ✓ GO: ≥3% improvement with self-supervised losses")
    elif best_imp >= 1:
        print("  ~ WEAK GO: 1-3% improvement, try MLP/flow adapter")
    else:
        print("  ✗ FAIL: <1% improvement, try fallback plans")

    fm.cleanup()
    return results


if __name__ == "__main__":
    run_exp2(shift_type="mean", shift_magnitude=2.0)
