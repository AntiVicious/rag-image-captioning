"""
Recall@K: retriever quality, independent of caption/aggregation quality.

The ablation in scripts/evaluate.py measures END-TO-END caption quality
(BLEU/METEOR/ROUGE-L/CIDEr/CLIPScore against a reference caption), which
conflates "did CLIP+ChromaDB find the right neighbors" with "did the
aggregation/selection step turn those neighbors into good text." Recall@K
isolates the first question, using the standard image-to-text retrieval
formulation (as in CLIP-style retrieval papers): for a query IMAGE, rank
every caption in the index by embedding similarity, and check whether any
of that image's OWN ground-truth captions lands in the top K.

No held-out split here, deliberately: the "positive" being tested is
whether the query image's own captions rank highly among ALL captions in
the corpus (including its own) -- that's what "the retriever is working"
means for this metric. This is a different question from evaluate.py's
ablation, which deliberately removes an image's own captions to test
generalization to unseen images.

Usage:
    python scripts/measure_recall_at_k.py \
        --chroma-db-dir ./chroma_db --coco-img-dir ./coco/val2017 \
        --coco-ann-file ./coco/annotations/captions_val2017.json \
        --num-samples 200 --ks 1,5,10
"""

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402


def load_annotations(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann["id"])
    return by_image


def main():
    parser = argparse.ArgumentParser(description="Recall@K for the CLIP+ChromaDB retriever")
    parser.add_argument("--chroma-db-dir", default="./chroma_db")
    parser.add_argument("--coco-img-dir", default="./coco/val2017")
    parser.add_argument("--coco-ann-file", default="./coco/annotations/captions_val2017.json")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--out", default="./eval_output/recall_at_k.json")
    args = parser.parse_args()

    ks = sorted(int(k) for k in args.ks.split(","))
    max_k = max(ks)

    by_image = load_annotations(args.coco_ann_file)
    img_paths = {
        int(fn.split(".")[0]): os.path.join(args.coco_img_dir, fn)
        for fn in os.listdir(args.coco_img_dir)
        if fn.endswith(".jpg")
    }
    all_ids = sorted(set(by_image.keys()) & set(img_paths.keys()))
    sample_ids = random.Random(args.seed).sample(all_ids, min(args.num_samples, len(all_ids)))
    print(f"Recall@K over {len(sample_ids)} sampled images (seed={args.seed}), K={ks}")

    config = Config(chroma_db_dir=args.chroma_db_dir)
    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)

    print("Loading CLIP...")
    model_manager.load_clip_model()
    db_manager.initialize()
    print(f"Index: {db_manager.get_stats()['total_embeddings']} embeddings")

    hits_at_k = {k: 0 for k in ks}
    first_hit_ranks = []

    for i, img_id in enumerate(sample_ids):
        tensor = preprocessor.preprocess_for_clip(img_paths[img_id], model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor)
        query_embedding = embedding.cpu().numpy()[0].tolist()

        result = db_manager.query_similar([query_embedding], max_k)
        result_ids = (result.get("ids") or [[]])[0]
        own_ann_ids = {f"{img_id}_{a}" for a in by_image[img_id]}

        rank = next((r + 1 for r, rid in enumerate(result_ids) if rid in own_ann_ids), None)
        if rank is not None:
            first_hit_ranks.append(rank)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample_ids)}")

    n = len(sample_ids)
    recall = {f"R@{k}": hits_at_k[k] / n for k in ks}
    print(f"\n=== Recall@K (N={n}) ===")
    for k in ks:
        print(f"  R@{k}: {recall[f'R@{k}']:.3f}  ({hits_at_k[k]}/{n})")
    if first_hit_ranks:
        print(
            f"  median rank of first own-caption hit (when found): "
            f"{statistics.median(first_hit_ranks):.1f}"
        )
    misses = n - len(first_hit_ranks)
    print(f"  no own-caption in top-{max_k}: {misses}/{n} ({100 * misses / n:.1f}%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "n": n,
                "ks": ks,
                "recall": recall,
                "hits_at_k": hits_at_k,
                "median_first_hit_rank": statistics.median(first_hit_ranks) if first_hit_ranks else None,
                "misses_at_max_k": misses,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
