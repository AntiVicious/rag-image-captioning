"""
Retrieval-augmented generation: turns an image into a single retrieved
caption, using CLIP embeddings + ChromaDB and, optionally, segmentation/
object-detection crops for a richer per-region candidate pool. How the
final caption gets chosen out of that pool is Config.selection_mode --
see src/caption_selection.py ("medoid"/"top1") and src/caption_aggregation.py
(the legacy "aggregated" concatenation mode).

RAGRetriever takes its collaborators (ModelManager, DatabaseManager,
ImagePreprocessor) as constructor arguments rather than importing global
singletons, so it can be unit tested against fakes without touching any
real model or database.
"""

from typing import Dict, Union

import torch
from PIL import Image

from .caption_aggregation import aggregate_captions
from .caption_selection import select_caption
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

    def _select(self, documents_per_variant, distances_per_variant, all_blocks) -> str:
        """Turn a per-variant candidate pool into one final caption, per
        Config.selection_mode -- shared by both the basic and advanced
        retrieval paths."""
        mode = self.config.selection_mode
        if mode in ("top1", "medoid"):
            return select_caption(documents_per_variant, distances_per_variant, mode, self.model_manager)
        if self.config.aggregate_captions:
            return aggregate_captions(all_blocks)
        return " ".join(all_blocks)

    def retrieve(self, image: Union[str, Image.Image], top_k: int = None, use_advanced: bool = True) -> Dict:
        """Retrieve captions for an image, optionally using crop variants."""
        if top_k is None:
            top_k = self.config.default_top_k
        # top1 only ever looks at each variant's closest candidate, so query
        # n_results=1 directly -- this also sidesteps a real chromadb
        # tie-break quirk where the returned top-1 among distance-tied
        # candidates (e.g. a COCO image's ~5 co-located captions) can depend
        # on the requested n_results (see scripts/diagnose_topk_nondeterminism.py).
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
        caption = self._select(documents_per_variant, distances_per_variant, all_blocks)
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

        # top_k already reflects the top1-mode n_results=1 override applied
        # in retrieve() -- captions_per_crop is only used for medoid/aggregated,
        # which want a real candidate pool, not just each variant's closest hit.
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

        aggregated = self._select(documents_per_variant, distances_per_variant, all_blocks)

        return {
            "aggregated_caption": aggregated,
            "variants_processed": len(variant_results),
            "variant_results": variant_results,
            "detected_objects": variants.get("detected_objects", []),
            "mode": "advanced",
        }
