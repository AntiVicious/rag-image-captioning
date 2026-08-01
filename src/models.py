"""
Model loading and management.

Models are loaded lazily — on first call to a `load_*` method, never at
import time or at ModelManager construction time — so importing this
module or constructing a ModelManager never triggers a weights download.
"""
from typing import Callable, Tuple

import torch

from .config import Config


class ModelManager:
    """Owns the CLIP model (and, later, generation models) for a Config."""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.clip_model = None
        self.clip_preprocess = None

    def load_clip_model(self) -> Tuple[torch.nn.Module, Callable]:
        """Load and cache the CLIP model plus its preprocessing transform."""
        if self.clip_model is not None:
            return self.clip_model, self.clip_preprocess

        import open_clip

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=self.config.clip_pretrained,
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        return self.clip_model, self.clip_preprocess

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed image tensor into a normalised CLIP embedding."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding
