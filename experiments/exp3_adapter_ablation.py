"""Experiment 3: Does the adapter type matter?

Compare Affine (InputAdapter), MLP, and Flow adapters
with the best loss configuration from Exp 2.
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
from models.adapters import InputAdapter, AffineAdapter, MLPAdapter, FlowAdapter
from evaluate import compute_metrics, relative_improvement


def combined_loss_fn(adapted_ctx, original_ctx):
    """Combined loss (temporal + spectral + reconstruction)."""
    T = adapted_ctx.shape[1]
    stride = T // 4

    # Temporal consistency
    w1 = adapted_ctx[:, :T - stride, :]
    w2 = adapted_ctx[:, stride:, :]
    overlap1 = w1[:, stride:, :]
    overlap2 = w2[:, :T - 2 * stride, :]
    l_temp = F.mse_loss(overlap1, overlap2)

    # Spectral consistency
    fft_orig = torch.fft.rfft(original_ctx, dim=1)
    fft_adapt = torch.fft.rfft(adapted_ctx, dim=1)
    psd_orig = torch.abs(fft_orig) ** 2
    psd_adapt = torch.abs(fft_adapt) ** 2
    psd_orig = psd_orig / (psd_orig.sum(dim=1, keepdim=True) + 1e-8)
    psd_adapt = psd_adapt / (psd_adapt.sum(dim=1, keepdim=True) + 1e-8)
    l_spec = F.mse_loss(psd_adapt, psd_orig)

    # Reconstruction (keep close to original at unmasked parts)
    mask_len = max(1, T // 8)
    mask_start = torch.randint(0, T - mask_len, (1,)).item()
    loss_unmasked = F.mse_loss(
        torch.cat([adapted_ctx[:, :mask_start], adapted_ctx[:, mask_start + mask_len:]], dim=1),
        torch.cat([original_ctx[:, :mask_start], original_ctx[:, mask_start + mask_len:]], dim=1),
    )

    return 1.0 * l_temp + 0.5 * l_spec + 0.5 * loss_unmasked


def adapt_with_adapter(fm, context, prediction_length, adapter, loss_fn,
                       adapt_steps=10, adapt_lr=1e-3, is_input_level=True):
    """Run adaptation and return prediction."""
    adapter.reset_parameters()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=adapt_lr)

    dev = next(adapter.parameters()).device
    ctx_tensor = torch.tensor(context, dtype=torch.float32, device=dev)
    if is_input_level:
        ctx_tensor = ctx_tensor.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    else:
        ctx_tensor = ctx_tensor.unsqueeze(0)  # (1, T)

    t_start = time.time()

    for step in range(adapt_steps):
        optimizer.zero_grad()
        if is_input_level:
            adapted = adapter(ctx_tensor)
            loss = loss_fn(adapted, ctx_tensor)
        else:
            # Embedding-level: need to encode first
            with torch.no_grad():
                z = fm.encode(ctx_tensor)
            z_adapted = adapter(z)
            loss = loss_fn(z_adapted, z)
        loss.backward()
        optimizer.step()

    adapt_time_ms = (time.time() - t_start) * 1000

    # Predict
    with torch.no_grad():
        if is_input_level:
            adapted_final = adapter(ctx_tensor).squeeze(-1).squeeze(0)
            pred = fm.predict(adapted_final.cpu().numpy(), prediction_length=prediction_length, num_samples=20)
        else:
            pred = fm.predict(context, prediction_length=prediction_length, num_samples=20)

    return pred.squeeze(), adapt_time_ms


def run_exp3(shift_type="mean", shift_magnitude=2.0):
    print("=" * 60)
    print("  EXPERIMENT 3: Adapter Type Comparison")
    print(f"  Shift: {shift_type}, Magnitude: {shift_magnitude}")
    print("=" * 60)

    data = load_etth1(context_length=512, prediction_length=64)
    windows = data["test_windows"]
    print(f"Loaded ETTh1: {len(windows)} test windows")

    fm = ChronosWrapper(model_id="amazon/chronos-t5-small", device="cuda")

    dev = fm.device

    # Input-level adapters (most practical)
    adapter_configs = {
        "zero_shot": None,
        "affine_input": InputAdapter(n_features=1).to(dev),
        "mlp_input": torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 1),
        ).to(dev),
        "flow_input": FlowAdapter(embed_dim=1, cond_dim=4, hidden_dim=64, steps=3).to(dev),
    }

    # Initialize MLP with residual
    class MLPInputAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, 64),
                torch.nn.GELU(),
                torch.nn.Linear(64, 1),
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

    adapter_configs["mlp_input"] = MLPInputAdapter().to(dev)

    results = {}

    for name, adapter in adapter_configs.items():
        print(f"\n--- {name} ---")
        if adapter is not None:
            n_params = sum(p.numel() for p in adapter.parameters())
            print(f"  Parameters: {n_params}")

        all_preds, all_trues = [], []
        total_adapt_time = 0

        for i, (ctx, tgt) in enumerate(windows):
            ctx_shifted = apply_shift(ctx, shift_type, shift_magnitude)

            if adapter is None:
                pred = fm.predict(ctx_shifted, prediction_length=64, num_samples=20).squeeze()
                adapt_time = 0
            else:
                pred, adapt_time = adapt_with_adapter(
                    fm, ctx_shifted, 96, adapter, combined_loss_fn,
                    adapt_steps=10, adapt_lr=1e-3, is_input_level=True,
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

        n_params = sum(p.numel() for p in adapter.parameters()) if adapter else 0
        results[name] = {
            "mse": metrics["mse"],
            "mae": metrics["mae"],
            "params": n_params,
            "avg_adapt_ms": avg_time,
        }
        print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")

    # Summary
    baseline_mse = results["zero_shot"]["mse"]
    print("\n" + "=" * 60)
    print("  ADAPTER COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Adapter':>15} | {'Params':>8} | {'MSE':>10} | {'MAE':>10} | {'Improv%':>8} | {'ms/batch':>8}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*8}")
    for name, r in results.items():
        imp = relative_improvement(baseline_mse, r["mse"])
        print(f"  {name:>15} | {r['params']:>8} | {r['mse']:>10.6f} | {r['mae']:>10.6f} | {imp:>+7.1f}% | {r['avg_adapt_ms']:>7.1f}")

    fm.cleanup()
    return results


if __name__ == "__main__":
    run_exp3(shift_type="mean", shift_magnitude=2.0)
