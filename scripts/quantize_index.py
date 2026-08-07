"""
Index quantization: memory saved vs. retrieval-quality lost.

The 591,753-embedding train2017-scale index (see the scale-test section of
the walkthrough doc / README) is ~1.2GB of float32 vectors before any
quantization. This measures what float16 and int8 scalar quantization
actually cost in retrieval quality, using each quantized representation's
OWN top-k agreement with the float32 ground-truth ranking as the metric --
not a proxy, a direct measurement: for N sampled query embeddings, brute-
force cosine search the full index in float32 (ground truth) and in each
quantized representation (dequantized back to float32 for a fair, common
comparison space), then report what fraction of the float32 top-k survives
in the quantized top-k.

ChromaDB's own HNSW index doesn't expose a quantization knob directly, so
this operates on the raw embeddings fetched out of the collection -- it
measures what quantizing the VECTORS costs, independent of any particular
ANN index implementation built on top of them.

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
    embeddings_f32 = np.array(result["embeddings"], dtype=np.float32)
    del result  # the raw Python list-of-lists chromadb returned -- release before it's needed
    print(f"Fetched in {time.perf_counter() - t0:.1f}s, shape={embeddings_f32.shape}")

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

    print(f"\nBrute-force top-{args.k} search, N={len(query_rows)} queries, against all {n} embeddings...")

    def top_k_indices(query_matrix, target_matrix, k):
        # cosine similarity via matmul (both sides already unit-normalised)
        sims = query_matrix @ target_matrix.T
        # argpartition for speed, then sort just the top-k
        part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        row_idx = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[row_idx, part], axis=1)
        return part[row_idx, order]

    gt_idx = top_k_indices(queries_f32, embeddings_f32, args.k)

    embeddings_f16 = quantize_float16(embeddings_f32)
    queries_f16 = quantize_float16(queries_f32)
    f16_idx = top_k_indices(queries_f16, embeddings_f16, args.k)
    f16_recall = recall_at_k(gt_idx, f16_idx, args.k)

    embeddings_i8, _ = quantize_int8(embeddings_f32)
    queries_i8, _ = quantize_int8(queries_f32)
    i8_idx = top_k_indices(queries_i8, embeddings_i8, args.k)
    i8_recall = recall_at_k(gt_idx, i8_idx, args.k)

    print(f"\n=== Recall@{args.k} of quantized search vs. float32 ground truth ===")
    print(f"  float16: {f16_recall:.4f}  ({100 * (1 - f16_recall):.2f}% of top-{args.k} lost, "
          f"{100 * (1 - bytes_f16 / bytes_f32):.0f}% memory saved)")
    print(f"  int8:    {i8_recall:.4f}  ({100 * (1 - i8_recall):.2f}% of top-{args.k} lost, "
          f"{100 * (1 - bytes_int8 / bytes_f32):.0f}% memory saved)")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n_embeddings": n,
                "dim": d,
                "num_queries": len(query_rows),
                "k": args.k,
                "memory_mb": {"float32": bytes_f32 / 1e6, "float16": bytes_f16 / 1e6, "int8": bytes_int8 / 1e6},
                "recall_vs_float32": {"float16": f16_recall, "int8": i8_recall},
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
