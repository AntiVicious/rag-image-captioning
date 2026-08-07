"""
Distribution-shift probe: how much worse is retrieval on Indian-context
images against a COCO-only index, compared to COCO images against that
same index?

For each query image, records the ChromaDB top-1 distance (lower =
closer/more confident match) and the retrieved captions. Reports
aggregate stats for two query sets against the SAME index:
  - a sample of COCO's own images (the index's native distribution)
  - the Indian-context images (food / Commons scrape)

A meaningfully higher mean/median top-1 distance for the Indian set is
direct, quantified evidence of distribution shift -- no ground-truth
captions needed for the Indian images, just their embeddings' distance
from whatever COCO has on file.

Usage:
    python scripts/measure_distribution_shift.py \
        --chroma-db-dir ./chroma_db \
        --coco-img-dir ./coco/val2017 --coco-num-samples 200 \
        --indian-img-dir ./coco_indian/food/images --indian-num-samples 200 \
        --out ./distribution_shift_results.json
"""

import argparse
import json
import os
import shutil
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402


def prepare_held_out_index(source_db_dir, work_db_dir, held_out_sources):
    """Copy chroma_db -> work_db_dir, then delete every sampled query
    image's own embeddings from the copy, so no query set can trivially
    self-match (distance 0) against the very images that built the
    index -- same leakage class scripts/evaluate.py guards against.
    Must be done for EVERY query set being probed, not just one: if the
    Indian images were themselves added to this index (the "after"
    measurement), querying with them unheld-out finds themselves at
    distance 0, which is a leakage artifact, not a real result.

    held_out_sources: list of (ann_file, filenames) pairs to exclude.
    """
    if os.path.exists(work_db_dir):
        shutil.rmtree(work_db_dir)
    shutil.copytree(source_db_dir, work_db_dir)

    ids_to_delete = []
    total_sample_images = 0
    for ann_file, filenames in held_out_sources:
        with open(ann_file) as f:
            data = json.load(f)
        sample_ids = {int(fn.split(".")[0]) for fn in filenames}
        total_sample_images += len(sample_ids)
        ids_to_delete.extend(
            f"{ann['image_id']}_{ann['id']}" for ann in data["annotations"] if ann["image_id"] in sample_ids
        )

    config = Config(chroma_db_dir=work_db_dir)
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    before = db_manager.get_stats()["total_embeddings"]
    if ids_to_delete:
        db_manager.collection.delete(ids=ids_to_delete)
    after = db_manager.get_stats()["total_embeddings"]
    print(f"Held-out index: {before} -> {after} embeddings after removing {total_sample_images} query images")
    return work_db_dir


def sample_images(img_dir, n, seed=42):
    import random

    filenames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if len(filenames) <= n:
        return filenames
    return random.Random(seed).sample(filenames, n)


def probe(img_dir, filenames, model_manager, db_manager, preprocessor, top_k=5):
    records = []
    for fn in filenames:
        img_path = os.path.join(img_dir, fn)
        tensor = preprocessor.preprocess_for_clip(img_path, model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor)
        query_embedding = embedding.cpu().numpy()[0].tolist()

        results = db_manager.query_similar([query_embedding], top_k)
        distances = (results.get("distances") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        if not distances:
            continue
        records.append(
            {
                "file": fn,
                "top1_distance": distances[0],
                "top1_caption": documents[0] if documents else None,
                "mean_topk_distance": sum(distances) / len(distances),
            }
        )
    return records


def summarize(records, label):
    top1 = [r["top1_distance"] for r in records]
    print(f"\n=== {label} (N={len(records)}) ===")
    print(f"  top-1 distance: mean={statistics.mean(top1):.4f}  median={statistics.median(top1):.4f}")
    print(f"  top-1 distance: min={min(top1):.4f}  max={max(top1):.4f}")
    for r in records[:3]:
        print(f"  example: {r['file']} -> dist={r['top1_distance']:.4f} caption={r['top1_caption']!r}")
    return {
        "n": len(records),
        "top1_distance_mean": statistics.mean(top1),
        "top1_distance_median": statistics.median(top1),
        "top1_distance_min": min(top1),
        "top1_distance_max": max(top1),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure distribution shift via ChromaDB retrieval distance")
    parser.add_argument("--chroma-db-dir", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--coco-ann-file", required=True)
    parser.add_argument("--coco-num-samples", type=int, default=200)
    parser.add_argument("--indian-img-dir", required=True)
    parser.add_argument("--indian-num-samples", type=int, default=200)
    parser.add_argument(
        "--indian-ann-file",
        default=None,
        help="Only needed if the Indian images are themselves IN --chroma-db-dir (e.g. measuring "
        "'after' augmentation) -- excludes their own embeddings so they can't self-match at "
        "distance 0. Omit when querying a COCO-only index the Indian images were never added to.",
    )
    parser.add_argument("--work-db-dir", default="/tmp/chroma_db_dist_shift")
    parser.add_argument("--out", default="./distribution_shift_results.json")
    args = parser.parse_args()

    coco_files = sample_images(args.coco_img_dir, args.coco_num_samples)
    indian_files = sample_images(args.indian_img_dir, args.indian_num_samples)
    print(f"COCO sample: {len(coco_files)} images from {args.coco_img_dir}")
    print(f"Indian sample: {len(indian_files)} images from {args.indian_img_dir}")

    held_out_sources = [(args.coco_ann_file, coco_files)]
    if args.indian_ann_file:
        held_out_sources.append((args.indian_ann_file, indian_files))
    held_out_db_dir = prepare_held_out_index(args.chroma_db_dir, args.work_db_dir, held_out_sources)

    config = Config(chroma_db_dir=held_out_db_dir)
    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)

    print("Loading CLIP...")
    model_manager.load_clip_model()
    db_manager.initialize()
    stats = db_manager.get_stats()
    print(f"Index (held-out): {stats['total_embeddings']} embeddings in '{stats['collection_name']}'")

    coco_records = probe(args.coco_img_dir, coco_files, model_manager, db_manager, preprocessor)
    indian_records = probe(args.indian_img_dir, indian_files, model_manager, db_manager, preprocessor)

    coco_summary = summarize(coco_records, "COCO images vs COCO-only index (native distribution)")
    indian_summary = summarize(indian_records, "Indian-context images vs COCO-only index")

    shift_pct = (indian_summary["top1_distance_mean"] / coco_summary["top1_distance_mean"] - 1) * 100
    print(f"\nIndian top-1 distance is {shift_pct:+.1f}% vs COCO's own top-1 distance on this index.")

    with open(args.out, "w") as f:
        json.dump(
            {
                "index_size": stats["total_embeddings"],
                "coco": coco_summary,
                "indian": indian_summary,
                "shift_pct": shift_pct,
                "coco_records": coco_records,
                "indian_records": indian_records,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
