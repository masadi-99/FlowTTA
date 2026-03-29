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

    elif shift_type == "trend_intrawindow":
        # Linear trend within window — RevIN subtracts mean but slope remains
        trend = np.linspace(-magnitude * std, magnitude * std, len(series))
        shifted += trend

    elif shift_type == "frequency":
        # Shift dominant frequency by compressing/stretching time axis
        from scipy.interpolate import interp1d
        t_orig = np.linspace(0, 1, len(series))
        t_new = np.linspace(0, 1, int(len(series) * (1 + 0.3 * magnitude)))
        f = interp1d(t_orig, series, kind='cubic', fill_value='extrapolate')
        resampled = f(t_new)
        shifted = resampled[:len(series)]
        # Preserve original mean/std so RevIN gets no advantage
        shifted = (shifted - shifted.mean()) / (shifted.std() + 1e-8) * std + series.mean()

    elif shift_type == "autocorrelation":
        # Break temporal smoothness — add high-frequency perturbation
        noise_freq = magnitude * 2
        t = np.arange(len(series))
        perturbation = magnitude * std * 0.3 * np.sin(2 * np.pi * noise_freq * t / len(series))
        shifted += perturbation
        # Keep same mean/std as original (neutralizes RevIN advantage)
        shifted = (shifted - shifted.mean()) / (shifted.std() + 1e-8) * std + series.mean()

    elif shift_type == "outlier":
        # Inject random outliers that corrupt RevIN's mean/std computation
        n_outliers = max(1, int(len(series) * 0.02 * magnitude))
        outlier_idx = np.random.choice(len(series), n_outliers, replace=False)
        shifted[outlier_idx] = series.mean() + magnitude * 5 * std * np.random.choice([-1, 1], n_outliers)

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
