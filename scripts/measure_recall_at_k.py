"""
Recall@K diagnostic -- NOT a clean retriever-quality metric, and the
original version of this script wrongly claimed it was.

The original R@1=R@5=R@10=99.0% result was a self-match artifact, not
evidence of retrieval quality: the query image's own embedding was never
excluded from the index, and every caption for an image shares ONE stored
CLIP image embedding (captions aren't independently embedded -- this
system retrieves nearest IMAGES, not nearest TEXTS, and passes through
whichever image's captions it lands on). Re-encoding an image and
searching an index that already contains that exact image's embedding is
trivially guaranteed to find itself first; that measures pipeline
determinism, not "does the retriever find relevant neighbours."

This script now runs three variants to make that explicit instead of
asserting it, per a review of the original result:

  baseline      -- original (broken) setup: real image queries against the
                   UN-held-out index. Expected: near-100%, and that's the
                   self-match artifact, not a result.
  held-out      -- the SAME query images, but with every one of their
                   annotation rows (all ~5 captions -- they all share one
                   embedding, so removing only one caption row leaves the
                   image just as findable via a sibling row) deleted from a
                   *copy* of the index first. Once genuinely absent, there
                   is no ground-truth "correct" caption left to recall --
                   COCO has no cross-image relevance judgments -- so this
                   variant cannot score anything but 0% by the same
                   criterion, and that's the point: it demonstrates baseline's
                   99% depended entirely on the leak, not on evidence a
                   different, merely-similar image's caption was "close
                   enough."
  random        -- baseline's (un-held-out) index, but queried with random
                   unit vectors instead of real image embeddings. If this
                   ALSO comes back near 100%, the index/query wiring itself
                   is broken (matches anything, not just self). If it's
                   near the chance floor (~max_k / index_size), that
                   confirms baseline's 99% is specifically the self-match
                   leak and nothing else is wrong with the measurement path.

Conclusion this script is expected to support: there is no valid, non-
trivial Recall@K claim to make about this system as originally framed.
What retrieval-quality evidence this project actually has lives in the
held-out ablation (scripts/evaluate.py) and distribution-shift
(scripts/measure_distribution_shift_v2.py) work, which score retrieved
CAPTION TEXT against real references/distances under a proper hold-out --
not "did it find itself."

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
    """Copy chroma_db -> work_db_dir, delete EVERY annotation row (all ~5
    captions, which all share one image embedding) for each sampled image,
    so it is genuinely absent from the index, not just missing one caption
    among several identical-embedding siblings."""
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
    """queries: list of (img_id, query_embedding) pairs, aligned to sample_ids
    (img_id may be None for random-vector queries, which have no 'own
    caption' to look for and therefore always score 0 by construction --
    that's the point of the random control)."""
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
        print(f"  median rank of first own-caption hit (when found): {statistics.median(first_hit_ranks):.1f}")
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

    # --- baseline: real queries against the UN-held-out index ---
    db_baseline = DatabaseManager(Config(chroma_db_dir=args.chroma_db_dir))
    db_baseline.initialize()
    print(f"\nBaseline index: {db_baseline.get_stats()['total_embeddings']} embeddings (not held out)")
    results["baseline_unheld_out"] = run_variant(
        "baseline (unheld-out, real queries -- expected: self-match, not a result)",
        db_baseline, real_queries, by_image, sample_ids, ks, max_k,
    )

    # --- random control: same un-held-out index, random-vector queries ---
    results["random_control"] = run_variant(
        "random control (unheld-out index, random-vector queries -- expected: ~chance floor)",
        db_baseline, random_queries, by_image, sample_ids, ks, max_k,
    )

    # --- held-out: real queries against an index with the sample fully removed ---
    held_out_dir = prepare_held_out_index(args.chroma_db_dir, args.held_out_db_dir, by_image, sample_ids)
    db_held_out = DatabaseManager(Config(chroma_db_dir=held_out_dir))
    db_held_out.initialize()
    results["held_out"] = run_variant(
        "held-out (query images fully excluded -- expected: 0%, no ground truth exists to recall)",
        db_held_out, real_queries, by_image, sample_ids, ks, max_k,
    )

    chance_floor = max_k / db_baseline.get_stats()["total_embeddings"]
    print(f"\nChance floor for R@{max_k} on a {db_baseline.get_stats()['total_embeddings']}-embedding "
          f"index (max_k/index_size, a rough sanity bound): {chance_floor:.5f}")

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
