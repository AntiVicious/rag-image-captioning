"""Recall@K diagnostic. Shows that Recall@K is not a meaningful retrieval
metric for this system; an earlier version reported 99% by mistake.

All captions of an image share one stored image embedding, so a query image
searched against an index that still contains it finds itself first. Three
variants make this explicit:

  baseline  -- real queries against the full index. Near 100%: the
               self-match.
  held-out  -- the same queries with all of their images' rows (all ~5
               captions) deleted from a copy of the index. No correct item
               remains, since COCO has no cross-image relevance labels, so
               the score is 0%.
  random    -- random unit vectors against the full index. Near the chance
               floor (~max_k / index_size), confirming the baseline's 99%
               comes only from the self-match.

The retrieval evidence for this project is the held-out ablation
(scripts/evaluate.py) and the distribution-shift measurement
(scripts/measure_distribution_shift_v2.py).

Usage:
    python scripts/measure_recall_at_k.py \\
        --chroma-db-dir ./chroma_db --coco-img-dir ./coco/val2017 \\
        --coco-ann-file ./coco/annotations/captions_val2017.json \\
        --num-samples 200 --ks 1,5,10
"""

import argparse
import json
import os
import random
import shutil
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


def prepare_held_out_index(source_db_dir, work_db_dir, by_image, sample_ids):
    """Copy chroma_db to work_db_dir and delete every caption row of each
    sampled image (they all share one embedding), so the image is absent.
    """
    if os.path.exists(work_db_dir):
        shutil.rmtree(work_db_dir)
    shutil.copytree(source_db_dir, work_db_dir)

    ids_to_delete = [f"{img_id}_{ann_id}" for img_id in sample_ids for ann_id in by_image[img_id]]
    config = Config(chroma_db_dir=work_db_dir)
    db_manager = DatabaseManager(config)
    db_manager.initialize()
    before = db_manager.get_stats()["total_embeddings"]
    db_manager.collection.delete(ids=ids_to_delete)
    after = db_manager.get_stats()["total_embeddings"]
    print(f"Held-out index: {before} -> {after} embeddings after removing {len(sample_ids)} query images")
    return work_db_dir


def run_variant(label, db_manager, queries, by_image, sample_ids, ks, max_k):
    """queries: list of (img_id, query_embedding) pairs aligned with
    sample_ids. img_id is None for random-vector queries, which always
    score 0.
    """
    hits_at_k = {k: 0 for k in ks}
    first_hit_ranks = []
    n = len(queries)

    for i, (img_id, query_embedding) in enumerate(queries):
        result = db_manager.query_similar([query_embedding], max_k)
        result_ids = (result.get("ids") or [[]])[0]
        own_ann_ids = {f"{img_id}_{a}" for a in by_image[img_id]} if img_id is not None else set()

        rank = next((r + 1 for r, rid in enumerate(result_ids) if rid in own_ann_ids), None)
        if rank is not None:
            first_hit_ranks.append(rank)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i + 1}/{n}")

    recall = {f"R@{k}": hits_at_k[k] / n for k in ks}
    print(f"\n=== {label} (N={n}) ===")
    for k in ks:
        print(f"  R@{k}: {recall[f'R@{k}']:.4f}  ({hits_at_k[k]}/{n})")
    if first_hit_ranks:
        print(
            f"  median rank of first own-caption hit (when found): {statistics.median(first_hit_ranks):.1f}"
        )
    return {
        "n": n,
        "recall": recall,
        "hits_at_k": hits_at_k,
        "median_first_hit_rank": statistics.median(first_hit_ranks) if first_hit_ranks else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Recall@K diagnostic (baseline / held-out / random control)")
    parser.add_argument("--chroma-db-dir", default="./chroma_db")
    parser.add_argument("--coco-img-dir", default="./coco/val2017")
    parser.add_argument("--coco-ann-file", default="./coco/annotations/captions_val2017.json")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ks", default="1,5,10")
    parser.add_argument("--held-out-db-dir", default="./chroma_db_recall_heldout")
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
    print(f"Sample: {len(sample_ids)} images (seed={args.seed}), K={ks}")

    config = Config(chroma_db_dir=args.chroma_db_dir)
    model_manager = ModelManager(config)
    preprocessor = ImagePreprocessor(config)
    print("Loading CLIP...")
    model_manager.load_clip_model()

    print("Encoding all query images once (reused across variants)...")
    real_queries = []
    embed_dim = None
    for i, img_id in enumerate(sample_ids):
        tensor = preprocessor.preprocess_for_clip(img_paths[img_id], model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor).cpu().numpy()[0].tolist()
        embed_dim = len(embedding)
        real_queries.append((img_id, embedding))
        if (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(sample_ids)}")

    rng = random.Random(args.seed + 1)
    random_queries = []
    for _ in sample_ids:
        vec = [rng.gauss(0, 1) for _ in range(embed_dim)]
        norm = sum(x * x for x in vec) ** 0.5
        random_queries.append((None, [x / norm for x in vec]))

    results = {}

    # --- baseline: real queries against the full index ---
    db_baseline = DatabaseManager(Config(chroma_db_dir=args.chroma_db_dir))
    db_baseline.initialize()
    print(f"\nBaseline index: {db_baseline.get_stats()['total_embeddings']} embeddings (not held out)")
    results["baseline_unheld_out"] = run_variant(
        "baseline (unheld-out, real queries -- expected: self-match, not a result)",
        db_baseline,
        real_queries,
        by_image,
        sample_ids,
        ks,
        max_k,
    )

    # --- random control: random vectors against the full index ---
    results["random_control"] = run_variant(
        "random control (unheld-out index, random-vector queries -- expected: ~chance floor)",
        db_baseline,
        random_queries,
        by_image,
        sample_ids,
        ks,
        max_k,
    )

    # --- held-out: real queries with their images removed from the index ---
    held_out_dir = prepare_held_out_index(args.chroma_db_dir, args.held_out_db_dir, by_image, sample_ids)
    db_held_out = DatabaseManager(Config(chroma_db_dir=held_out_dir))
    db_held_out.initialize()
    results["held_out"] = run_variant(
        "held-out (query images fully excluded -- expected: 0%, no ground truth exists to recall)",
        db_held_out,
        real_queries,
        by_image,
        sample_ids,
        ks,
        max_k,
    )

    chance_floor = max_k / db_baseline.get_stats()["total_embeddings"]
    print(
        f"\nChance floor for R@{max_k} on a {db_baseline.get_stats()['total_embeddings']}-embedding "
        f"index (max_k/index_size, a rough sanity bound): {chance_floor:.5f}"
    )

    print("\n=== Conclusion ===")
    print("baseline's high R@K measures embedding-pipeline self-consistency (the query image's own")
    print("embedding IS the index entry it's compared against), not retrieval quality. There is no")
    print("valid non-trivial Recall@K claim for this system without ground-truth cross-image relevance")
    print("judgments, which COCO does not provide for this task.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ks": ks, "chance_floor_at_max_k": chance_floor, **results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
