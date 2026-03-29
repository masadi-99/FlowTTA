"""ETTh1 dataset loader for FlowTTA experiments."""

import os
import numpy as np
import pandas as pd


ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cached")


def download_etth1():
    """Download ETTh1 CSV if not cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "ETTh1.csv")
    if not os.path.exists(path):
        print(f"Downloading ETTh1 from {ETT_URL}...")
        df = pd.read_csv(ETT_URL)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
    return path


def load_etth1(context_length=512, prediction_length=64):
    """
    Load ETTh1 and return train/val/test splits as numpy arrays.

    Standard split:
      Train: months 1-12 (8640 rows)
      Val:   months 13-16 (2880 rows)
      Test:  months 17-20 (2880 rows)

    Returns dict with keys: train, val, test, test_windows
    Each is (N, T, C) where C=7 features (excluding date and OT target, or using all 7).
    test_windows: list of (context, target) tuples for evaluation.
    """
    csv_path = download_etth1()
    df = pd.read_csv(csv_path)

    # Drop date column, use all 7 numeric features
    features = df.drop(columns=["date"]).values.astype(np.float32)

    # Standard ETT splits
    n_train = 8640
    n_val = 2880
    n_test = 2880

    train = features[:n_train]
    val = features[n_train:n_train + n_val]
    test = features[n_train + n_val:n_train + n_val + n_test]

    # Create rolling windows for test evaluation
    # For Chronos (univariate), we use OT (last column) as the target
    test_ot = test[:, -1]  # OT column (target)

    windows = []
    stride = prediction_length
    for start in range(0, len(test_ot) - context_length - prediction_length + 1, stride):
        ctx = test_ot[start:start + context_length]
        tgt = test_ot[start + context_length:start + context_length + prediction_length]
        windows.append((ctx, tgt))

    # Compute normalization stats from train
    train_ot = train[:, -1]
    train_mean = train_ot.mean()
    train_std = train_ot.std()

    return {
        "train": train,
        "val": val,
        "test": test,
        "test_ot": test_ot,
        "test_windows": windows,
        "train_mean": train_mean,
        "train_std": train_std,
        "n_features": features.shape[1],
    }
