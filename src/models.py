"""
Model loading and management.

Models are loaded on the first call to a `load_*` method, so importing this
module or creating a ModelManager doesn't download weights.
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

        Despite the name, this loads BLIP base by default (see
        Config.blip2_model_name), which runs on CPU in seconds."""
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
        """Encode strings into normalised CLIP text embeddings with the same
        model used for images, so both share one embedding space (as CLIPScore,
        Hessel et al. 2021, arXiv:2104.08718, requires)."""
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
