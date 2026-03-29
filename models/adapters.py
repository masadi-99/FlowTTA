"""Adapter modules for FlowTTA test-time adaptation."""

import torch
import torch.nn as nn


class AffineAdapter(nn.Module):
    """Learnable shift + scale on embeddings. ~2*embed_dim params."""

    def __init__(self, embed_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, z):
        return z * self.scale + self.shift

    def reset_parameters(self):
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.shift)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class MLPAdapter(nn.Module):
    """2-layer MLP adapter with residual connection."""

    def __init__(self, embed_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._zero_init()

    def _zero_init(self):
        """Zero-init last layer so adapter starts as identity."""
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)

    def reset_parameters(self):
        for m in self.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self._zero_init()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class FlowAdapter(nn.Module):
    """
    Conditional flow matching adapter.
    Learns velocity field v(z, t, c) that transports shifted embeddings.
    c = temporal context features (mean, std, dominant_freq of test window).
    """

    def __init__(self, embed_dim, cond_dim=16, hidden_dim=256, steps=3):
        super().__init__()
        self.steps = steps
        self.cond_dim = cond_dim
        self.time_embed = nn.Linear(1, hidden_dim)
        self.cond_embed = nn.Linear(cond_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Zero-init output for identity start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def velocity(self, z_t, t, cond):
        t_emb = self.time_embed(t.unsqueeze(-1))
        c_emb = self.cond_embed(cond)
        # Broadcast t_emb and c_emb to match z_t dimensions
        if z_t.dim() == 3:  # (B, seq_len, embed_dim)
            t_emb = t_emb.unsqueeze(1).expand(-1, z_t.shape[1], -1)
            c_emb = c_emb.unsqueeze(1).expand(-1, z_t.shape[1], -1)
        h = torch.cat([z_t, t_emb, c_emb], dim=-1)
        return self.net(h)

    def forward(self, z_source, cond=None):
        if cond is None:
            # Default conditioning: zeros
            B = z_source.shape[0]
            cond = torch.zeros(B, self.cond_dim, device=z_source.device)

        dt = 1.0 / self.steps
        z = z_source
        for i in range(self.steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            v = self.velocity(z, t, cond)
            z = z + v * dt
        return z

    def reset_parameters(self):
        for m in [self.time_embed, self.cond_embed]:
            m.reset_parameters()
        for m in self.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class InputAdapter(nn.Module):
    """Adapt raw input instead of embeddings (fallback approach)."""

    def __init__(self, n_features=1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_features))
        self.shift = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        return x * self.scale + self.shift

    def reset_parameters(self):
        nn.init.ones_(self.scale)
        nn.init.zeros_(self.shift)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_adapter(adapter_type, embed_dim, **kwargs):
    """Factory function for adapters."""
    if adapter_type == "affine":
        return AffineAdapter(embed_dim)
    elif adapter_type == "mlp":
        return MLPAdapter(embed_dim, kwargs.get("hidden_dim", 256))
    elif adapter_type == "flow":
        return FlowAdapter(
            embed_dim,
            cond_dim=kwargs.get("cond_dim", 16),
            hidden_dim=kwargs.get("hidden_dim", 256),
            steps=kwargs.get("flow_steps", 3),
        )
    elif adapter_type == "input":
        return InputAdapter(kwargs.get("n_features", 1))
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
