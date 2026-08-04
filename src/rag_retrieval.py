"""
Retrieval-augmented generation: turns an image into an aggregated block of
retrieved captions, using CLIP embeddings + ChromaDB and, optionally,
segmentation/object-detection crops for richer per-region context.

RAGRetriever takes its collaborators (ModelManager, DatabaseManager,
ImagePreprocessor) as constructor arguments rather than importing global
singletons, so it can be unit tested against fakes without touching any
real model or database.
"""

from typing import Dict, Union

import torch
from PIL import Image

from .caption_aggregation import aggregate_captions
from .config import Config
from .database import DatabaseManager
from .image_preprocessing import ImagePreprocessor
from .models import ModelManager


class RAGRetriever:
    """Retrieves and aggregates captions for an image."""

    def __init__(
        self,
        config: Config,
        model_manager: ModelManager,
        db_manager: DatabaseManager,
        preprocessor: ImagePreprocessor,
    ):
        self.config = config
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.preprocessor = preprocessor

    def _retrieve_for_tensor(self, image_tensor, top_k: int) -> str:
        embedding = self.model_manager.encode_image(image_tensor)
        query_embedding = embedding.cpu().numpy()[0].tolist()
        results = self.db_manager.query_similar([query_embedding], top_k)
        documents = results.get("documents") or [[]]
        return " ".join(documents[0])

    def retrieve(self, image: Union[str, Image.Image], top_k: int = None, use_advanced: bool = True) -> Dict:
        """Retrieve captions for an image, optionally using crop variants."""
        if top_k is None:
            top_k = self.config.default_top_k

        if self.model_manager.clip_preprocess is None:
            raise RuntimeError("CLIP model not loaded. Call model_manager.load_clip_model() first.")

        if use_advanced:
            return self._retrieve_advanced(image, top_k)
        return self._retrieve_basic(image, top_k)

    def _retrieve_basic(self, image: Union[str, Image.Image], top_k: int) -> Dict:
        tensor = self.preprocessor.preprocess_for_clip(image, self.model_manager.clip_preprocess)
        tensor = tensor.to(self.model_manager.device)
        caption = self._retrieve_for_tensor(tensor, top_k)
        return {"aggregated_caption": caption, "variants_processed": 1, "mode": "basic"}

    def _retrieve_advanced(self, image: Union[str, Image.Image], top_k: int) -> Dict:
        """Same result as querying each variant one at a time, but the CLIP
        encode and the ChromaDB query each run as a single batched call
        across all variants instead of one Python-loop round trip per
        variant -- DETR (if enabled) still dominates wall time, but this
        removes per-call dispatch/query overhead that scales with variant
        count for free."""
        variants = self.preprocessor.get_all_variants(image)

        labeled_images = [("original", variants["original"])]
        labeled_images += [(f"segment_{i + 1}", c) for i, c in enumerate(variants["segmentation_crops"])]
        labeled_images += [(f"object_{i + 1}", c) for i, c in enumerate(variants["object_detection_crops"])]

        tensors = [
            self.preprocessor.preprocess_for_clip(img, self.model_manager.clip_preprocess)
            for _, img in labeled_images
        ]
        batch = torch.cat(tensors, dim=0).to(self.model_manager.device)
        embeddings = self.model_manager.encode_image(batch)
        query_embeddings = embeddings.cpu().numpy().tolist()

        results = self.db_manager.query_similar(query_embeddings, self.config.captions_per_crop)
        documents_per_variant = results.get("documents") or [[] for _ in labeled_images]

        all_blocks = []
        variant_results = []
        for (label, _), docs in zip(labeled_images, documents_per_variant):
            caption = " ".join(docs)
            all_blocks.append(caption)
            variant_results.append({"type": label, "caption": caption})

        if self.config.aggregate_captions:
            aggregated = aggregate_captions(all_blocks)
        else:
            aggregated = " ".join(all_blocks)

        return {
            "aggregated_caption": aggregated,
            "variants_processed": len(variant_results),
            "variant_results": variant_results,
            "detected_objects": variants.get("detected_objects", []),
            "mode": "advanced",
        }
