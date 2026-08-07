"""
Precision robustness of the retrieval ranking under float16/int8 rounding --
NOT a measured memory-savings result, despite an earlier version of this
script's output implying it was.

What this DOES measure, validly: for N query embeddings (rows of the index
itself, with the query's own row AND every sibling row sharing its image_id
excluded from its own candidate search -- COCO stores ~5 caption rows per
image, ALL sharing one identical embedding, and a first version of this
script only excluded the single sampled row, leaving the other ~4 sibling
rows in as trivially-guaranteed top-of-list matches for both float32 and
float16 alike; that inflated the measured agreement between the two by
hiding real disagreement behind guaranteed-identical sibling hits), brute-
force cosine search in float32 (ground truth) vs. each quantized-then-
dequantized representation, and report what fraction of the float32 top-k
survives in the quantized top-k.

A second measurement trap, found and fixed after the first published numbers
already had the sibling-leakage fix above: COCO contains enough near/exact-
duplicate images that exact float32 cosine-similarity ties are common (152/
200 queries in a 100k-row sample tied exactly at rank 0). The original top-1
extraction used np.argpartition + a plain np.argsort, which is NOT a stable
sort -- on a tie it can return either candidate arbitrarily, and did so
differently for float32 vs its float16-quantized self. That alone produced
an apparent 76.5% top-1 "flip rate" that was almost entirely tie-break noise,
not measured quantization error (a separate, deterministic-argmax-based
script measuring the same queries put the real number at 1.5%). Fixed by
making top_k_indices' tie-break deterministic (lowest index wins, matching
np.argmax's convention) -- see the function for detail.

What this does NOT measure: any real memory saved. Checked against current
docs/community reporting rather than assumed: ChromaDB's HNSW index (via
its hnswlib fork) stores vectors as float32 internally and has no native
scalar/product quantization option -- feeding it float16-rounded values
would not change what it stores or how much RAM/disk it uses, since it
re-expands everything to float32 regardless. The "memory saved" figures
below are the THEORETICAL bytes-per-element arithmetic on arrays pulled
OUT of ChromaDB into numpy for this experiment, not something achievable
by quantizing the live collection. The valid finding here is narrower and
still real: retrieval ranking is robust to float16 rounding (and, to a
lesser extent, int8 rounding) -- realizing that as an actual memory win
would require swapping to a vector store that supports on-disk
quantization (e.g. Qdrant's scalar quantization), which this project does
not do.

Usage:
    python scripts/quantize_index.py --chroma-db-dir /app/chroma_db_scale_test \
        --num-queries 200 --k 10
"""

import argparse
import json
import statistics
import time

import numpy as np


def recall_at_k(ground_truth_idx, candidate_idx, k):
    """Fraction of ground_truth_idx[:k] that appear anywhere in candidate_idx[:k]."""
    hits = 0
    total = 0
    for gt_row, cand_row in zip(ground_truth_idx, candidate_idx):
        gt_set = set(gt_row[:k].tolist())
        cand_set = set(cand_row[:k].tolist())
        hits += len(gt_set & cand_set)
        total += k
    return hits / total


def quantize_float16(embeddings):
    return embeddings.astype(np.float16).astype(np.float32)


def quantize_int8(embeddings):
    """Global affine scalar quantization: one scale factor for the whole
    matrix (embeddings are CLIP-normalised, so magnitudes are already in a
    narrow, comparable range across dimensions and vectors)."""
    scale = 127.0 / np.abs(embeddings).max()
    quantized = np.round(embeddings * scale).astype(np.int8)
    dequantized = quantized.astype(np.float32) / scale
    return dequantized, quantized


def main():
    parser = argparse.ArgumentParser(description="Measure quantization's memory-vs-recall tradeoff")
    parser.add_argument("--chroma-db-dir", default="/app/chroma_db_scale_test")
    parser.add_argument("--collection-name", default="coco_clip_embeddings")
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=100_000,
        help="Fetch at most this many embeddings out of the collection (via chromadb's own "
        "limit=, not a post-fetch sample) rather than all 591,753 -- chromadb's .get() "
        "materialises the full result as Python lists-of-lists before any numpy conversion, "
        "which alone exceeds this machine's WSL2 memory cap for the full index (confirmed: "
        "OOM-killed, exit 137, even before any quantization math ran). 100k is still a "
        "large, representative sample for a recall-degradation measurement.",
    )
    parser.add_argument("--out", default="/app/eval_output/quantization_results.json")
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.chroma_db_dir)
    collection = client.get_collection(args.collection_name)
    n_total = collection.count()
    n_fetch = min(args.max_embeddings, n_total)
    print(f"Fetching {n_fetch}/{n_total} embeddings from {args.chroma_db_dir}...")
    t0 = time.perf_counter()
    result = collection.get(limit=n_fetch, include=["embeddings"])
    ids = result["ids"]
    embeddings_f32 = np.array(result["embeddings"], dtype=np.float32)
    del result  # the raw Python list-of-lists chromadb returned -- release before it's needed
    print(f"Fetched in {time.perf_counter() - t0:.1f}s, shape={embeddings_f32.shape}")

    # ids are "{image_id}_{ann_id}" -- COCO stores ~5 caption rows per image,
    # all sharing one embedding, so excluding a query's own row alone still
    # leaves its siblings in as guaranteed near-identical matches.
    image_ids = np.array([gid.split("_")[0] for gid in ids])

    norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True)
    embeddings_f32 /= np.clip(norms, 1e-8, None)  # in place -- avoid doubling peak memory

    n, d = embeddings_f32.shape
    bytes_f32 = n * d * 4
    bytes_f16 = n * d * 2
    bytes_int8 = n * d * 1
    print(f"\nMemory footprint (embeddings only, {n} x {d}):")
    print(f"  float32: {bytes_f32 / 1e6:.1f} MB (baseline)")
    print(f"  float16: {bytes_f16 / 1e6:.1f} MB ({100 * (1 - bytes_f16 / bytes_f32):.0f}% smaller)")
    print(f"  int8:    {bytes_int8 / 1e6:.1f} MB ({100 * (1 - bytes_int8 / bytes_f32):.0f}% smaller)")

    rng = np.random.RandomState(args.seed)
    query_rows = rng.choice(n, size=min(args.num_queries, n), replace=False)
    queries_f32 = embeddings_f32[query_rows]

    # Exclude every row sharing a query's image_id (not just the sampled row
    # itself) from that query's candidate search -- see module docstring.
    query_image_ids = image_ids[query_rows]
    exclude_mask = image_ids[None, :] == query_image_ids[:, None]  # (num_queries, n)
    avg_excluded = exclude_mask.sum(axis=1).mean()
    print(f"\nExcluding an average of {avg_excluded:.1f} sibling rows per query (own image_id's other captions)")

    print(f"\nBrute-force top-{args.k} search, N={len(query_rows)} queries, against all {n} embeddings...")

    def top_k_indices(query_matrix, target_matrix, k, exclude_mask):
        # cosine similarity via matmul (both sides already unit-normalised).
        # COCO has enough near/exact-duplicate images that exact float32 ties
        # in cosine similarity are common, not rare (confirmed empirically:
        # 152/200 queries had an exact tie at rank 0 in a 100k-row sample).
        # np.argsort's default quicksort is NOT stable, so on a tie it picks
        # an arbitrary winner that can differ between two otherwise-identical
        # runs (e.g. float32 vs its float16-quantized self) -- inflating the
        # measured top-1 "flip rate" with tie-break noise that has nothing to
        # do with quantization. Break ties deterministically by lowest row
        # index (matching np.argmax's convention) via: pre-sort the partition
        # by index, then stable-sort by similarity.
        sims = query_matrix @ target_matrix.T
        sims = np.where(exclude_mask, -np.inf, sims)
        # argpartition for speed, then sort just the top-k
        part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        part = np.sort(part, axis=1)  # ascending index order -- ties resolve to lowest index
        row_idx = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row_idx, part], axis=1, kind="stable")
        return part[row_idx, order]

    gt_idx = top_k_indices(queries_f32, embeddings_f32, args.k, exclude_mask)

    embeddings_f16 = quantize_float16(embeddings_f32)
    queries_f16 = quantize_float16(queries_f32)
    f16_idx = top_k_indices(queries_f16, embeddings_f16, args.k, exclude_mask)
    f16_recall = recall_at_k(gt_idx, f16_idx, args.k)

    embeddings_i8, _ = quantize_int8(embeddings_f32)
    queries_i8, _ = quantize_int8(queries_f32)
    i8_idx = top_k_indices(queries_i8, embeddings_i8, args.k, exclude_mask)
    i8_recall = recall_at_k(gt_idx, i8_idx, args.k)

    # Top-1 flip rate: a coarser, more interpretable companion number to the
    # top-k SET-overlap figure above -- "does the single best pick change,"
    # not "does anything in the top-k change." Distinct metrics, reported
    # separately rather than conflated (top-10 set overlap != rank
    # correlation != top-1 agreement -- this script reports the first and
    # third; see scripts/measure_quantization_downstream.py for what a
    # top-1 flip actually costs in caption quality, not just ranking).
    top1_flips = int((gt_idx[:, 0] != f16_idx[:, 0]).sum())
    top1_flip_rate = top1_flips / len(query_rows)

    print(f"\n=== Metric 1: top-{args.k} SET OVERLAP vs. float32 (sibling rows excluded) ===")
    print("(fraction of the float32 top-k CANDIDATE SET that also appears, in any order, in the "
          "quantized top-k set -- counts any position changing, not just the best one)")
    print(f"  float16: {f16_recall:.4f}  ({100 * (1 - f16_recall):.2f}% of top-{args.k} set changed)")
    print(f"  int8:    {i8_recall:.4f}  ({100 * (1 - i8_recall):.2f}% of top-{args.k} set changed)")

    print(f"\n=== Metric 2: TOP-1 FLIP RATE vs. float32 (float16 only) ===")
    print("(fraction of queries where the single BEST pick is a literally different row under "
          "float16 -- a stricter, more interpretable number than set overlap; see "
          "scripts/measure_quantization_downstream.py for what this costs in real caption quality)")
    print(f"  float16: {top1_flip_rate:.4f}  ({top1_flips}/{len(query_rows)} queries flip their top-1 pick)")

    print(f"\nTheoretical bytes-per-element memory (NOT achievable in ChromaDB -- see module docstring):")
    print(f"  float32: {bytes_f32 / 1e6:.1f} MB   float16: {bytes_f16 / 1e6:.1f} MB   int8: {bytes_int8 / 1e6:.1f} MB")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n_embeddings": n,
                "dim": d,
                "num_queries": len(query_rows),
                "k": args.k,
                "sibling_rows_excluded": True,
                "avg_excluded_per_query": float(avg_excluded),
                "theoretical_memory_mb_not_achievable_in_chromadb": {
                    "float32": bytes_f32 / 1e6,
                    "float16": bytes_f16 / 1e6,
                    "int8": bytes_int8 / 1e6,
                },
                "top_k_set_overlap_vs_float32": {"k": args.k, "float16": f16_recall, "int8": i8_recall},
                "top1_flip_rate_vs_float32": {"float16": top1_flip_rate, "flips": top1_flips},
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
