"""Self-supervised losses for FlowTTA test-time adaptation."""

import torch
import torch.nn.functional as F


def temporal_consistency_loss(z_adapted_1, z_adapted_2, overlap_len):
    """
    Adjacent/overlapping windows should have consistent adapted embeddings.

    Args:
        z_adapted_1: (B, seq_len_1, embed_dim) - adapted embeddings for window 1
        z_adapted_2: (B, seq_len_2, embed_dim) - adapted embeddings for window 2
        overlap_len: number of overlapping positions

    Returns:
        scalar loss
    """
    if overlap_len <= 0:
        return torch.tensor(0.0, device=z_adapted_1.device)

    # The tail of window 1 should match the head of window 2
    overlap_z1 = z_adapted_1[:, -overlap_len:, :]
    overlap_z2 = z_adapted_2[:, :overlap_len, :]

    return F.mse_loss(overlap_z1, overlap_z2)


def spectral_consistency_loss(z_adapted, z_original):
    """
    Adaptation should preserve frequency structure of embeddings.

    Args:
        z_adapted: (B, seq_len, embed_dim)
        z_original: (B, seq_len, embed_dim)

    Returns:
        scalar loss
    """
    # Compute power spectral density along sequence dimension
    fft_original = torch.fft.rfft(z_original, dim=1)
    fft_adapted = torch.fft.rfft(z_adapted, dim=1)

    psd_original = torch.abs(fft_original) ** 2
    psd_adapted = torch.abs(fft_adapted) ** 2

    # Normalize to compare shape (distribution), not scale
    psd_original = psd_original / (psd_original.sum(dim=1, keepdim=True) + 1e-8)
    psd_adapted = psd_adapted / (psd_adapted.sum(dim=1, keepdim=True) + 1e-8)

    # Use MSE instead of KL-div (more stable for optimization)
    return F.mse_loss(psd_adapted, psd_original)


def masked_reconstruction_loss(z_full_adapted, z_masked_adapted, mask_positions):
    """
    The adapted embedding of the full input should be able to
    reconstruct what the masked version cannot see.

    Simplified version: compare adapted embeddings at masked positions.

    Args:
        z_full_adapted: (B, seq_len, embed_dim) - adapted embeddings of full input
        z_masked_adapted: (B, seq_len, embed_dim) - adapted embeddings of masked input
        mask_positions: list of (start, end) tuples for each batch element

    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=z_full_adapted.device)
    B = z_full_adapted.shape[0]

    for i in range(B):
        s, e = mask_positions[i]
        if s < z_full_adapted.shape[1] and e <= z_full_adapted.shape[1]:
            loss += F.mse_loss(z_masked_adapted[i, s:e], z_full_adapted[i, s:e].detach())

    return loss / max(B, 1)


def entropy_loss(forecast_samples):
    """
    Fallback loss: minimize entropy/spread of FM's predictive distribution.

    Args:
        forecast_samples: (B, num_samples, prediction_length) - sampled forecasts

    Returns:
        scalar loss (mean spread across quantiles)
    """
    # Minimize the spread (IQR or std) of the predictive distribution
    spread = forecast_samples.std(dim=1)  # (B, prediction_length)
    return spread.mean()
