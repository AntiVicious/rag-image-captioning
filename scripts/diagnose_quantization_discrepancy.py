"""
Throwaway diagnostic that found a real bug: quantize_index.py reported a
76.5% float16 top-1 flip rate; measure_quantization_downstream.py, on the
same 100k-embedding sample/seed=42/200-queries/exclusion logic, reported
1.5%. Same math on paper -- a 50x gap demanded explanation before either
number could be trusted.

Two hypotheses were tested here, in order:

1. Floating-point non-associativity between batched matmul (GEMM, used by
   quantize_index.py's vectorised top_k_indices) and looped single-query
   matmul (GEMV, used by measure_quantization_downstream.py's top1()) --
   plausible given the near-tie regime the top-10 set-overlap numbers imply.
   FALSIFIED: computing top-1 both ways on the identical float32 similarity
   data, with no quantization involved at all, gave 0/200 disagreements.

2. The one remaining difference: quantize_index.py extracts top-1 via
   np.argpartition + np.argsort (for top-k speed), while the downstream
   script uses plain np.argmax. Isolating THAT specific mechanism (this
   file's main() below) on the SAME batched similarity matrix found the
   real cause: 152/200 disagreements, every one inspected an exact float32
   tie (identical similarity to 8 significant figures). np.argsort's
   default quicksort is not stable, so on a tie it picks an arbitrary
   winner -- differently for float32 vs its float16-quantized self -- while
   np.argmax deterministically picks the lowest index. COCO has enough
   near/exact-duplicate images that these ties are common, not rare.

Fix applied to scripts/quantize_index.py's top_k_indices: sort the
partition by index before the similarity sort, and make that sort stable,
so ties resolve to the lowest index (matching np.argmax's convention).
Re-running afterward brought quantize_index.py's flip rate to 1.5%,
matching measure_quantization_downstream.py exactly.
"""

import time

import numpy as np


def main():
    import chromadb

    client = chromadb.PersistentClient(path="/app/chroma_db_scale_test")
    collection = client.get_collection("coco_clip_embeddings")
    n_fetch = 100_000
    print(f"Fetching {n_fetch} embeddings...")
    t0 = time.perf_counter()
    result = collection.get(limit=n_fetch, include=["embeddings"])
    ids = result["ids"]
    embeddings = np.array(result["embeddings"], dtype=np.float32)
    del result
    print(f"Fetched in {time.perf_counter() - t0:.1f}s")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.clip(norms, 1e-8, None)
    n, d = embeddings.shape
    image_ids = np.array([gid.split("_")[0] for gid in ids])

    rng = np.random.RandomState(42)
    query_rows = rng.choice(n, size=200, replace=False)
    queries = embeddings[query_rows]
    query_image_ids = image_ids[query_rows]

    exclude_mask = image_ids[None, :] == query_image_ids[:, None]
    sims = queries @ embeddings.T
    sims = np.where(exclude_mask, -np.inf, sims)

    # method A: plain argmax
    top1_argmax = np.argmax(sims, axis=1)

    # method B: exactly quantize_index.py's top_k_indices chain, k=10
    k = 10
    part = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    row_idx = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row_idx, part], axis=1)
    top_k_idx = part[row_idx, order]
    top1_chain = top_k_idx[:, 0]

    disagree = top1_argmax != top1_chain
    n_disagree = int(disagree.sum())
    print(f"\nplain argmax vs argpartition+argsort chain (k={k}), SAME sims matrix:")
    print(f"  disagreements: {n_disagree}/{len(query_rows)} = {n_disagree / len(query_rows):.4f}")

    if n_disagree > 0:
        idx = np.where(disagree)[0][0]
        a, c = top1_argmax[idx], top1_chain[idx]
        print(f"\n  Example (query index {idx}, row {query_rows[idx]}):")
        print(f"    argmax says: idx={a}  sim={sims[idx, a]:.8f}")
        print(f"    chain  says: idx={c}  sim={sims[idx, c]:.8f}")
        print(f"    is chain's pick even IN the top-{k} set? {c in part[idx].tolist()}")
        print(f"    is argmax's pick even IN the top-{k} set? {a in part[idx].tolist()}")
        # rank of argmax's true best within the full row
        true_rank = int((sims[idx] > sims[idx, a]).sum())
        print(f"    true rank of argmax's pick (0=best): {true_rank}")


if __name__ == "__main__":
    main()
