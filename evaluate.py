"""Evaluation utilities for FlowTTA."""

import numpy as np


def compute_metrics(pred, true):
    """
    Compute MSE and MAE.

    Args:
        pred: (B, prediction_length) or (prediction_length,)
        true: (B, prediction_length) or (prediction_length,)

    Returns:
        dict with mse, mae
    """
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()

    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))

    return {"mse": float(mse), "mae": float(mae)}


def relative_improvement(metric_base, metric_ours):
    """
    Compute relative improvement (positive = we improved).

    Returns percentage improvement.
    """
    if metric_base == 0:
        return 0.0
    return (metric_base - metric_ours) / metric_base * 100


def format_results_table(results, title="Results"):
    """Format results as a markdown-style table."""
    lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]

    if not results:
        lines.append("  No results.")
        return "\n".join(lines)

    # Get all keys from first result
    keys = list(results[0].keys())
    header = " | ".join(f"{k:>15}" for k in keys)
    lines.append(header)
    lines.append("-" * len(header))

    for row in results:
        line = " | ".join(f"{str(row.get(k, '')):>15}" for k in keys)
        lines.append(line)

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)
