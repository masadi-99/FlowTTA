"""Synthetic distribution shift functions for FlowTTA experiments."""

import numpy as np


def apply_shift(series, shift_type="mean", magnitude=2.0):
    """
    Apply controlled distribution shift to a 1D time series.

    Args:
        series: 1D numpy array
        shift_type: one of "mean", "variance", "trend"
        magnitude: shift strength (multiples of std)

    Returns:
        shifted series (same shape)
    """
    shifted = series.copy()
    std = series.std()

    if shift_type == "mean":
        shifted += magnitude * std
    elif shift_type == "variance":
        # Scale variance while preserving mean
        mean = series.mean()
        shifted = mean + (shifted - mean) * magnitude
    elif shift_type == "trend":
        trend = np.linspace(0, magnitude * std, len(series))
        shifted += trend
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")

    return shifted


def apply_shift_to_windows(windows, shift_type="mean", magnitude=2.0):
    """
    Apply shift to a list of (context, target) window pairs.
    Shifts both context and target consistently.
    """
    shifted_windows = []
    for ctx, tgt in windows:
        # Compute shift params from context (what we'd see at test time)
        full = np.concatenate([ctx, tgt])
        full_shifted = apply_shift(full, shift_type, magnitude)
        ctx_shifted = full_shifted[:len(ctx)]
        tgt_shifted = full_shifted[len(ctx):]
        shifted_windows.append((ctx_shifted, tgt_shifted))
    return shifted_windows
