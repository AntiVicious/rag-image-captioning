"""
Configuration for the RAG Image Captioning system.

A single typed, instantiable Config (as opposed to bare module-level
constants) so it can be constructed with overrides in tests/CI without
monkeypatching a module, and so multiple configurations can coexist in the
same process if ever needed.
"""

import os
from dataclasses import dataclass


def _detect_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    # Dataset paths
    coco_img_dir: str = "./coco/train2017"
    coco_ann_file: str = "./coco/annotations/captions_train2017.json"

    # CLIP model
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # ChromaDB
    chroma_db_dir: str = "./chroma_db"
    collection_name: str = "coco_clip_embeddings"
    embedding_batch_size: int = 256

    # Retrieval
    default_top_k: int = 5

    # Advanced preprocessing (segmentation / object detection crops)
    enable_segmentation: bool = True
    enable_object_detection: bool = True
    crops_per_mode: int = 3
    min_crop_size: int = 48
    crop_padding: int = 10
    detection_score_threshold: float = 0.3
    segmentation_model: str = "facebook/detr-resnet-50-panoptic"
    object_detection_model: str = "facebook/detr-resnet-50"
    captions_per_crop: int = 3
    aggregate_captions: bool = True

    # Caption generation backend:
    #   "retrieval" -> aggregated retrieved captions ARE the output (no LLM, no GPU needed)
    #   "blip"      -> BLIP-2 generates the final caption from retrieved context
    caption_backend: str = "retrieval"
    blip2_model_name: str = "Salesforce/blip2-opt-2.7b"
    max_new_tokens: int = 50
    num_beams: int = 3

    def validate_paths(self) -> bool:
        """Check that the COCO dataset is present on disk."""
        return os.path.exists(self.coco_img_dir) and os.path.exists(self.coco_ann_file)

    def create_output_dirs(self) -> None:
        """Create directories this config's paths point to."""
        os.makedirs(self.chroma_db_dir, exist_ok=True)

    @property
    def device(self) -> str:
        """Resolve the compute device (imports torch lazily)."""
        return _detect_device()
