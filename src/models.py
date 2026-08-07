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
        self.clip_tokenizer = None
        self.blip2_model = None
        self.blip2_processor = None

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

    def load_blip2_model(self):
        """Load and cache the caption-generation model plus its processor.

        Despite the name (kept for API stability -- see Config.blip2_model_name),
        this loads BLIP (v1) base by default, not BLIP-2: BLIP-2 (2.7B params) is
        impractical on a CPU-only machine, whereas BLIP base (~247M params) runs
        real inference in seconds on CPU and supports the same text-conditioned
        captioning call shape (image + text prompt -> caption)."""
        if self.blip2_model is not None:
            return self.blip2_model, self.blip2_processor

        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.blip2_processor = BlipProcessor.from_pretrained(self.config.blip2_model_name)
        # use_safetensors=True: refuse to fall back to pickle-based torch.load
        # (CVE-2025-32434) regardless of installed torch version.
        self.blip2_model = BlipForConditionalGeneration.from_pretrained(
            self.config.blip2_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        self.blip2_model = self.blip2_model.to(self.device)
        self.blip2_model.eval()
        return self.blip2_model, self.blip2_processor

    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed image tensor into a normalised CLIP embedding."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding

    def encode_text(self, texts) -> torch.Tensor:
        """Encode a list of strings into normalised CLIP text embeddings, using
        the same backbone/checkpoint already loaded for image encoding -- this
        keeps text and image embeddings in one consistent space, which is what
        CLIPScore (Hessel et al. 2021, arXiv:2104.08718) needs."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")
        import open_clip

        if self.clip_tokenizer is None:
            self.clip_tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)
        tokens = self.clip_tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return text_features
