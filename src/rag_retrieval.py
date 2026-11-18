"""
Turns an image into a retrieved caption using CLIP embeddings and ChromaDB,
optionally adding segmentation or object-detection crops as extra queries.
Config.selection_mode chooses the final caption (see caption_selection.py and
caption_aggregation.py).

RAGRetriever receives its ModelManager, DatabaseManager and ImagePreprocessor
as arguments, so tests can pass fakes.
"""

from typing import Dict, Union

import torch
from PIL import Image

from .caption_aggregation import aggregate_captions
from .caption_selection import match_distance_for, select_caption
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

    def _select(self, documents_per_variant, distances_per_variant, all_blocks):
        """Choose the final caption from the candidates per Config.selection_mode.

        Returns (caption, match_distance). match_distance is the selected
        candidate's ChromaDB distance, used as a confidence signal, or None
        in "aggregated" mode."""
        mode = self.config.selection_mode
        if mode in ("top1", "medoid"):
            caption = select_caption(documents_per_variant, distances_per_variant, mode, self.model_manager)
            distance = match_distance_for(documents_per_variant, distances_per_variant, caption)
            return caption, distance
        if self.config.aggregate_captions:
            return aggregate_captions(all_blocks), None
        return " ".join(all_blocks), None

    def retrieve(self, image: Union[str, Image.Image], top_k: int = None, use_advanced: bool = True) -> Dict:
        """Retrieve captions for an image, optionally using crop variants."""
        if top_k is None:
            top_k = self.config.default_top_k
        # top1 only needs each variant's closest candidate. Requesting one
        # result also avoids ChromaDB returning different tied candidates
        # for different n_results (see scripts/diagnose_topk_nondeterminism.py).
        if self.config.selection_mode == "top1":
            top_k = 1

        if self.model_manager.clip_preprocess is None:
            raise RuntimeError("CLIP model not loaded. Call model_manager.load_clip_model() first.")

        if use_advanced:
            return self._retrieve_advanced(image, top_k)
        return self._retrieve_basic(image, top_k)

    def _retrieve_basic(self, image: Union[str, Image.Image], top_k: int) -> Dict:
        tensor = self.preprocessor.preprocess_for_clip(image, self.model_manager.clip_preprocess)
        tensor = tensor.to(self.model_manager.device)
        embedding = self.model_manager.encode_image(tensor)
        query_embedding = embedding.cpu().numpy()[0].tolist()
        results = self.db_manager.query_similar([query_embedding], top_k)
        documents_per_variant = results.get("documents") or [[]]
        distances_per_variant = results.get("distances") or [[]]
        all_blocks = [" ".join(documents_per_variant[0])] if documents_per_variant else []
        caption, match_distance = self._select(documents_per_variant, distances_per_variant, all_blocks)
        return {
            "aggregated_caption": caption,
            "match_distance": match_distance,
            "variants_processed": 1,
            "mode": "basic",
        }

    def _retrieve_advanced(self, image: Union[str, Image.Image], top_k: int) -> Dict:
        """Retrieve candidates for the image and its crops, batching the CLIP
        encoding and the ChromaDB query across all variants."""
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

        # top_k is already 1 in top1 mode; captions_per_crop applies to the
        # medoid and aggregated modes, which need several candidates.
        query_top_k = top_k if self.config.selection_mode == "top1" else self.config.captions_per_crop
        results = self.db_manager.query_similar(query_embeddings, query_top_k)
        documents_per_variant = results.get("documents") or [[] for _ in labeled_images]
        distances_per_variant = results.get("distances") or [[] for _ in labeled_images]

        all_blocks = []
        variant_results = []
        for (label, _), docs in zip(labeled_images, documents_per_variant):
            caption = " ".join(docs)
            all_blocks.append(caption)
            variant_results.append({"type": label, "caption": caption})

        aggregated, match_distance = self._select(documents_per_variant, distances_per_variant, all_blocks)

        return {
            "aggregated_caption": aggregated,
            "match_distance": match_distance,
            "variants_processed": len(variant_results),
            "variant_results": variant_results,
            "detected_objects": variants.get("detected_objects", []),
            "mode": "advanced",
        }
