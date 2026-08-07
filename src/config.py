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
    # Default OFF: the ablation in scripts/evaluate.py (see the walkthrough
    # doc / README) shows retrieval-only-with-top1-selection beats every
    # crop-augmented top1 config on BLEU-4/METEOR/ROUGE-L/CIDEr/CLIPScore, at
    # a fraction of the latency (DETR is ~90% of per-image wall time once
    # crops are on). With selection_mode="medoid" the picture flips --
    # +segmentation beats retrieval-only there -- so crops are default-off
    # for LATENCY, not because they never help: flip on explicitly when
    # latency isn't the constraint (see results/selection_strategy_ablation.csv).
    enable_segmentation: bool = False
    enable_object_detection: bool = False
    crops_per_mode: int = 3
    min_crop_size: int = 48
    crop_padding: int = 10
    detection_score_threshold: float = 0.3
    segmentation_model: str = "facebook/detr-resnet-50-panoptic"
    object_detection_model: str = "facebook/detr-resnet-50"
    captions_per_crop: int = 3
    aggregate_captions: bool = True

    # How RAGRetriever picks a final caption out of the retrieved candidate
    # pool (src/caption_selection.py):
    #   "medoid" -> consensus pick: the candidate most similar, on average, to
    #               the rest of the pool (CLIP text-embedding similarity).
    #               Empirically the best-scoring strategy measured (see
    #               results/selection_strategy_ablation.csv) -- the default.
    #   "top1"   -> the candidate closest to the query image (lowest ChromaDB
    #               distance). Simpler, no extra model call, but scores lower.
    #   "aggregated" -> legacy behavior: concatenate/dedupe every candidate
    #               via aggregate_captions() instead of picking one. Kept for
    #               backward compatibility; not recommended (this is the
    #               concatenation that caused the original CIDEr collapse
    #               bug when output length wasn't held constant).
    selection_mode: str = "medoid"

    # Caption generation backend:
    #   "retrieval" -> aggregated retrieved captions ARE the output (no LLM, no GPU needed)
    #   "blip"      -> a BLIP captioner generates the final caption, conditioned on the
    #                  retrieved context as a text prompt
    caption_backend: str = "retrieval"
    # Salesforce/blip2-opt-2.7b (2.7B params) was the original choice here, but is
    # impractical on a CPU-only dev machine (multi-minute generations, ~15GB of
    # weights) and its empty-caption output was never root-caused before this got
    # deprioritised. Swapped for BLIP (v1) base -- ~990MB, ~247M params, genuinely
    # usable on CPU -- which supports the same text-conditioned captioning API
    # (image + text prompt in, caption out) this system needs. The attribute/method
    # names in ModelManager (blip2_model, load_blip2_model, ...) are legacy and now
    # load this model, not BLIP-2; kept as-is to avoid an unrelated rename churn.
    blip2_model_name: str = "Salesforce/blip-image-captioning-base"
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
