"""
Does float16 quantization's ranking churn actually cost caption quality?

scripts/quantize_index.py measures a precise but narrow thing: what fraction
of the float32 top-10 set survives in the float16 top-10 set (8.45% doesn't
survive, on the SAME 100k-embedding train2017 sample and 200 queries used
here). That number alone doesn't say whether it matters -- COCO has ~5
near-synonymous reference captions per image, so a top-1 "flip" to a
different neighbour can easily swap one valid caption for another equally
valid one. This script answers the actual downstream question: run the
SAME 200 queries' single best (top-1) caption through float32 vs float16
embeddings, score both against real COCO reference captions with the same
pycocoevalcap suite as the main ablation, and report the delta -- plus a
cleanly-defined top-1 flip rate (fraction of queries where float16's best
match is a literally different embedding row than float32's), distinct
from the top-10 set-overlap number already reported.

Reuses the exact same 100k-embedding sample, seed, and self-exclusion
methodology as quantize_index.py so the two results are directly
comparable (same queries, same index, same held-out logic).

Usage (needs the eval image for pycocoevalcap/METEOR's Java dependency):
    docker run --rm \
      -v D:\\chroma_db_scale_test:/app/chroma_db_scale_test \
      -v <clone>/coco/annotations:/app/coco_annotations \
      -v <repo>/eval_output:/app/eval_output \
      rag-image-captioning:eval python scripts/measure_quantization_downstream.py
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np


def load_annotations(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append(ann["caption"])
    return by_image


def score_with_pycocoevalcap(predictions, by_image, image_ids):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ordered = sorted(image_ids)
    gts_raw = {i: [{"caption": c} for c in by_image[i]] for i in ordered}
    res_raw = {i: [{"caption": predictions[i]}] for i in ordered}

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scores = {}
    bleu_avg, _ = Bleu(4).compute_score(gts, res)
    scores["BLEU-4"] = bleu_avg[3]
    for name, Scorer in [("METEOR", Meteor), ("ROUGE-L", Rouge), ("CIDEr", Cider)]:
        avg, _ = Scorer().compute_score(gts, res)
        scores[name] = avg
    return scores


def quantize_float16(embeddings):
    return embeddings.astype(np.float16).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Downstream caption-quality cost of float16 quantization")
    parser.add_argument("--chroma-db-dir", default="/app/chroma_db_scale_test")
    parser.add_argument("--collection-name", default="coco_clip_embeddings")
    parser.add_argument("--coco-ann-file", default="/app/coco_annotations/captions_train2017.json")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-embeddings", type=int, default=100_000)
    parser.add_argument("--out", default="/app/eval_output/quantization_downstream_results.json")
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.chroma_db_dir)
    collection = client.get_collection(args.collection_name)
    n_total = collection.count()
    n_fetch = min(args.max_embeddings, n_total)
    print(f"Fetching {n_fetch}/{n_total} embeddings from {args.chroma_db_dir}...")
    t0 = time.perf_counter()
    result = collection.get(limit=n_fetch, include=["embeddings", "documents"])
    ids = result["ids"]
    embeddings_f32 = np.array(result["embeddings"], dtype=np.float32)
    documents = np.array(result["documents"])
    del result
    print(f"Fetched in {time.perf_counter() - t0:.1f}s, shape={embeddings_f32.shape}")

    norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
    embeddings_f32 /= np.clip(norms, 1e-8, None)
    n, d = embeddings_f32.shape

    print("Loading COCO reference captions...")
    by_image = load_annotations(args.coco_ann_file)

    # ids are "{image_id}_{ann_id}" -- COCO stores ~5 caption rows per image,
    # all sharing one embedding. Without excluding every sibling row (not
    # just the sampled one), a query would almost always "retrieve" one of
    # its own real captions via a sibling row for BOTH float32 and float16,
    # scoring perfectly by construction and hiding the actual question this
    # script exists to answer (what happens when a GENUINELY different
    # neighbour gets swapped in). Same leakage class as elsewhere in this
    # project; see scripts/quantize_index.py's docstring for the twin fix.
    image_ids = np.array([gid.split("_")[0] for gid in ids])

    rng = np.random.RandomState(args.seed)
    query_rows = rng.choice(n, size=min(args.num_queries, n), replace=False)
    queries_f32 = embeddings_f32[query_rows]
    embeddings_f16 = quantize_float16(embeddings_f32)
    queries_f16 = quantize_float16(queries_f32)

    def top1(query_vector, target_matrix, exclude_mask_row):
        sims = target_matrix @ query_vector
        sims = np.where(exclude_mask_row, -np.inf, sims)
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])

    predictions_f32, predictions_f16 = {}, {}
    query_image_ids = []
    flips = 0
    flip_examples = []

    for qi, row in enumerate(query_rows):
        image_id = int(ids[row].split("_")[0])
        # skip if this image_id has no reference captions in this ann file
        # (shouldn't happen for train2017 image_ids, but guard anyway)
        if image_id not in by_image:
            continue
        query_image_ids.append(image_id)

        exclude_mask_row = image_ids == ids[row].split("_")[0]
        idx_f32, _ = top1(queries_f32[qi], embeddings_f32, exclude_mask_row)
        idx_f16, _ = top1(queries_f16[qi], embeddings_f16, exclude_mask_row)

        predictions_f32[image_id] = str(documents[idx_f32])
        predictions_f16[image_id] = str(documents[idx_f16])

        if idx_f32 != idx_f16:
            flips += 1
            if len(flip_examples) < 10:
                flip_examples.append(
                    {
                        "image_id": image_id,
                        "float32_caption": str(documents[idx_f32]),
                        "float16_caption": str(documents[idx_f16]),
                    }
                )

    n_scored = len(query_image_ids)
    flip_rate = flips / n_scored
    print(f"\nTop-1 flip rate: {flips}/{n_scored} = {flip_rate:.4f}")
    print("(this is a DIFFERENT number from the 8.45% top-10 set-overlap figure in "
          "results/quantization_summary.csv -- that one counts any of the top-10 candidates "
          "changing; this one counts only the single best pick changing)")

    print("\nSample flips (float32 vs float16 top-1 caption):")
    for ex in flip_examples[:5]:
        print(f"  image {ex['image_id']}:")
        print(f"    fp32: {ex['float32_caption']!r}")
        print(f"    fp16: {ex['float16_caption']!r}")

    print("\nScoring both prediction sets against real COCO reference captions...")
    scores_f32 = score_with_pycocoevalcap(predictions_f32, by_image, query_image_ids)
    scores_f16 = score_with_pycocoevalcap(predictions_f16, by_image, query_image_ids)

    print(f"\n=== Downstream caption quality, N={n_scored} ===")
    print(f"float32: {scores_f32}")
    print(f"float16: {scores_f16}")
    print("\n| Metric | float32 | float16 | delta |")
    print("|---|---|---|---|")
    for metric in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
        delta = scores_f16[metric] - scores_f32[metric]
        print(f"| {metric} | {scores_f32[metric]:.4f} | {scores_f16[metric]:.4f} | {delta:+.4f} |")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n_scored": n_scored,
                "top1_flip_rate": flip_rate,
                "top1_flips": flips,
                "flip_examples": flip_examples,
                "scores_float32": scores_f32,
                "scores_float16": scores_f16,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
