"""Unified Foundation Model wrapper for FlowTTA.

Provides encode/decode/predict interface for Chronos-2.
Designed to be extended to other FMs later.
"""

import torch
import numpy as np


class ChronosWrapper:
    """
    Wrapper around Chronos-2 that exposes:
      - predict(): standard zero-shot forecasting
      - encode(): extract encoder embeddings
      - decode(): generate predictions from embeddings
      - embed_dim: embedding dimension
    """

    def __init__(self, model_id="amazon/chronos-t5-small", device="cuda", dtype=torch.float32):
        from chronos import ChronosPipeline

        self.device = device
        self.dtype = dtype
        self.model_id = model_id

        print(f"Loading Chronos model: {model_id}...")
        # Try GPU first, fall back to CPU if OOM
        try:
            self.pipeline = ChronosPipeline.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError, Exception) as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"  GPU OOM, falling back to CPU...")
                device = "cpu"
                self.device = device
                self.pipeline = ChronosPipeline.from_pretrained(
                    model_id,
                    device_map="cpu",
                    dtype=dtype,
                )
            else:
                raise
        self.model = self.pipeline.model
        print(f"Model loaded on {device}.")

        # Register forward hook to capture encoder embeddings
        self._embeddings = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture intermediate embeddings."""
        # Chronos-Bolt uses a different architecture than T5-based Chronos
        # Try to find the encoder/backbone
        model = self.model

        # For Chronos-Bolt (patched encoder model)
        if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            self._encoder_module = model.model.encoder
        elif hasattr(model, 'encoder'):
            self._encoder_module = model.encoder
        else:
            # Fallback: just hook into the model itself
            self._encoder_module = model

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._embeddings['encoder'] = output[0]
            else:
                self._embeddings['encoder'] = output

        self._hook = self._encoder_module.register_forward_hook(hook_fn)

    @property
    def embed_dim(self):
        """Get embedding dimension by running a dummy forward pass if needed."""
        if 'encoder' in self._embeddings:
            return self._embeddings['encoder'].shape[-1]
        # Run dummy pass
        dummy = torch.randn(1, 64, dtype=self.dtype)
        with torch.no_grad():
            self.pipeline.predict(dummy, prediction_length=1, num_samples=1)
        return self._embeddings['encoder'].shape[-1]

    def predict(self, context, prediction_length=64, num_samples=20):
        """
        Standard zero-shot prediction.

        Args:
            context: numpy array (T,) or (B, T) or torch tensor
            prediction_length: forecast horizon

        Returns:
            median prediction as numpy array (B, prediction_length)
        """
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=self.dtype)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        # Chronos pipeline handles device mapping internally; pass CPU tensors
        context = context.cpu()

        with torch.no_grad():
            forecast = self.pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
            # forecast shape: (B, num_samples, prediction_length)
            median = forecast.median(dim=1).values  # (B, prediction_length)

        return median.cpu().numpy()

    def predict_with_quantiles(self, context, prediction_length=64, num_samples=20):
        """Return full sample distribution for entropy-based losses."""
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=self.dtype)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        context = context.cpu()

        with torch.no_grad():
            forecast = self.pipeline.predict(
                context,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
        return forecast.cpu().numpy()  # (B, num_samples, prediction_length)

    def encode(self, context):
        """
        Run encoder and return embeddings.

        Args:
            context: torch tensor (B, T)

        Returns:
            embeddings tensor (B, seq_len, embed_dim) on device
        """
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=self.dtype)
        if context.dim() == 1:
            context = context.unsqueeze(0)

        context = context.cpu()

        # Trigger forward pass to capture embeddings via hook
        with torch.no_grad():
            self.pipeline.predict(context, prediction_length=1, num_samples=1)

        return self._embeddings['encoder'].clone()

    def cleanup(self):
        """Remove hooks."""
        if hasattr(self, '_hook'):
            self._hook.remove()
