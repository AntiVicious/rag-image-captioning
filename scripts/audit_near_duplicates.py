"""
Near-duplicate audit on the Indian-context image set, and a corrected
distribution-shift re-measurement that excludes near-duplicate leakage.

Motivation: scripts/measure_distribution_shift.py's "after augmentation"
run reported Indian top-1 distance beating COCO's own baseline (0.162 vs
0.352 mean) -- and 54/200 (27%) of the sampled Indian query images had
top-1 distance suspiciously close to 0.0 (see dist_shift_output/
distribution_shift_after2.json). That is not evidence retrieval got better;
it's evidence the Indian set (particularly the Indian Foods dataset, which
uses one templated caption per dish class across ~300 photos/class) contains
near-duplicate images that trivially retrieve each other regardless of
templated captions, and the original held-out logic only excluded a query
image's OWN embedding, not its near-duplicate twins elsewhere in the index.

This script:
  1. Pulls every embedding out of the augmented ChromaDB collection directly
     (no CLIP/DETR inference needed -- the embeddings are already computed
     and stored).
  2. Computes full pairwise cosine similarity within the Indian subset
     (embeddings are unit-normalised, so cosine sim = dot product).
  3. Flags pairs above --threshold (default 0.95) as near-duplicates and
     unions them into clusters (union-find).
  4. Cross-references the same 200-image Indian query sample used by
     measure_distribution_shift.py's "after" run (read directly from its
     output JSON, not re-derived) against those clusters.
  5. Re-measures top-1 distance for that query set with near-dup twins
     ALSO excluded from the index (not just the query image itself) --
     entirely in numpy, since all embeddings are already in memory.

Usage (run inside the eval/app Docker image so chromadb matches the version
that built the index):
    docker run --rm -v D:\\chroma_db_augmented:/app/chroma_db_augmented \\
      -v <repo>/dist_shift_output:/app/dist_shift_output \\
      rag-image-captioning:eval python scripts/audit_near_duplicates.py
"""

import argparse
import json
import re
import statistics

import numpy as np

INDIAN_ID_RE = re.compile(r"^(\d+)_\1$")


def fetch_all(collection):
    """chromadb .get() with no ids/where returns everything; embeddings come
    back as a list of lists, fetched in one call since 37,552 rows is small."""
    result = collection.get(include=["embeddings", "documents"])
    return result["ids"], np.array(result["embeddings"], dtype=np.float32), result["documents"]


def union_find_clusters(pairs, n):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return clusters


def main():
    parser = argparse.ArgumentParser(description="Near-duplicate audit on the Indian image set")
    parser.add_argument("--chroma-db-dir", default="/app/chroma_db_augmented")
    parser.add_argument("--collection-name", default="coco_clip_embeddings")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--after-results",
        default="/app/dist_shift_output/distribution_shift_after2.json",
        help="Existing 'after augmentation' results JSON, to reuse its exact 200-image query sample.",
    )
    parser.add_argument("--out", default="/app/dist_shift_output/near_duplicate_audit.json")
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    import chromadb

    client = chromadb.PersistentClient(path=args.chroma_db_dir)
    collection = client.get_collection(args.collection_name)
    print(f"Fetching all {collection.count()} embeddings...")
    ids, embeddings, documents = fetch_all(collection)
    id_index = {id_: i for i, id_ in enumerate(ids)}

    # unit-normalise defensively (should already be normalised at write time)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)

    is_indian = np.array([bool(INDIAN_ID_RE.match(id_)) for id_ in ids])
    indian_idx = np.where(is_indian)[0]
    print(f"Total: {len(ids)}  Indian subset: {len(indian_idx)}  COCO subset: {(~is_indian).sum()}")

    indian_emb = embeddings[indian_idx]
    n = len(indian_idx)

    # --- 1. Pairwise near-duplicate detection within the Indian subset ---
    print(f"Computing pairwise cosine similarity within {n} Indian images (chunked)...")
    max_sim = np.zeros(n, dtype=np.float32)
    max_sim_partner = np.full(n, -1, dtype=np.int64)
    near_dup_pairs = []
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        chunk_sims = indian_emb[start:end] @ indian_emb.T  # (chunk, n)
        for local_i, global_i in enumerate(range(start, end)):
            row = chunk_sims[local_i].copy()
            row[global_i] = -1.0  # exclude self
            best_j = int(np.argmax(row))
            max_sim[global_i] = row[best_j]
            max_sim_partner[global_i] = best_j
            hits = np.where(row >= args.threshold)[0]
            for j in hits:
                if j > global_i:  # dedupe (i, j) vs (j, i)
                    near_dup_pairs.append((global_i, int(j)))
        print(f"  {end}/{n}")

    clusters = union_find_clusters(near_dup_pairs, n)
    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    n_flagged = sum(len(v) for v in multi_clusters.values())
    exact_dupes = int((max_sim > 0.9999).sum())

    print(f"\n=== Near-duplicate summary (threshold={args.threshold}) ===")
    print(f"Images with a >= {args.threshold} twin: {n_flagged}/{n} ({100 * n_flagged / n:.1f}%)")
    print(f"Images with a near-EXACT (>0.9999) twin: {exact_dupes}/{n} ({100 * exact_dupes / n:.1f}%)")
    print(f"Number of clusters (size > 1): {len(multi_clusters)}")
    sizes = sorted((len(v) for v in multi_clusters.values()), reverse=True)
    print(f"Largest cluster sizes: {sizes[:10]}")
    print(f"max_sim distribution: mean={max_sim.mean():.4f} median={statistics.median(max_sim.tolist()):.4f}")

    # image_id -> set of embedding-row-indices sharing a caption-record id
    # (an Indian image can appear once; ids are "{image_id}_{image_id}")
    indian_global_ids = [ids[i] for i in indian_idx]

    # --- 2. Cross-reference against the existing 200-image "after" query sample ---
    with open(args.after_results) as f:
        after = json.load(f)
    query_records = after["indian_records"]
    query_image_ids = [int(r["file"].split(".")[0]) for r in query_records]
    print(f"\nQuery sample from {args.after_results}: {len(query_image_ids)} images")

    local_pos = {int(gid.split("_")[0]): i for i, gid in enumerate(indian_global_ids)}
    flagged_in_sample = 0
    for img_id, rec in zip(query_image_ids, query_records):
        local_i = local_pos.get(img_id)
        if local_i is None:
            continue
        if max_sim[local_i] >= args.threshold:
            flagged_in_sample += 1
    print(
        f"Of the {len(query_image_ids)} sampled Indian query images, "
        f"{flagged_in_sample} ({100 * flagged_in_sample / len(query_image_ids):.1f}%) "
        f"have a >= {args.threshold} twin elsewhere in the index."
    )

    # --- 3. Corrected distance re-measurement: exclude twins, not just self ---
    # Must replicate the ORIGINAL held-out setup exactly (a single fixed index
    # with ALL 200 COCO query images' rows AND all 200 Indian query images'
    # rows removed -- prepare_held_out_index() in measure_distribution_shift.py
    # does this as one batch exclusion, not per-query) and then, on top of
    # that SAME fixed index, additionally remove near-dup twins. Excluding
    # twins per-query against the full unfiltered index (as an earlier version
    # of this script did) accidentally left the other 199 held-out Indian
    # images and 200 held-out COCO images available as candidates -- a laxer
    # index than the original measurement used, which silently produces a
    # BETTER-looking (lower) distance for reasons that have nothing to do
    # with deduplication. Caught by a sanity check below, not assumed away.
    coco_query_ids = {int(r["file"].split(".")[0]) for r in after["coco_records"]}
    indian_query_ids = set(query_image_ids)

    full_ids = list(ids)
    base_exclude = set()
    for gid in full_ids:
        img_id = int(gid.split("_")[0])
        if img_id in coco_query_ids or img_id in indian_query_ids:
            base_exclude.add(gid)

    twin_exclude = set()
    for img_id in query_image_ids:
        local_i = local_pos.get(img_id)
        if local_i is None:
            continue
        cluster_root = next((root for root, members in clusters.items() if local_i in members), None)
        if cluster_root is None:
            continue
        for j in clusters[cluster_root]:
            twin_exclude.add(indian_global_ids[j])

    full_exclude = base_exclude | twin_exclude
    fixed_mask = np.array([gid not in full_exclude for gid in full_ids])
    replication_mask = np.array([gid not in base_exclude for gid in full_ids])  # no twin exclusion, for sanity check
    print(
        f"\nFixed held-out index: {len(full_ids)} -> base held-out {len(full_ids) - replication_mask.sum()} "
        f"-> +{len(full_exclude) - len(base_exclude)} additional near-dup twins -> {fixed_mask.sum()} candidates"
    )

    documents_arr = np.array(documents)

    def query_against(mask, local_i):
        query_emb = indian_emb[local_i]
        sims = embeddings[mask] @ query_emb
        best_sim = float(sims.max())
        best_doc = documents_arr[mask][int(np.argmax(sims))]
        return best_sim, best_doc

    replication_records = []
    corrected_records = []
    for img_id, rec in zip(query_image_ids, query_records):
        local_i = local_pos.get(img_id)
        if local_i is None:
            replication_records.append(rec)
            corrected_records.append(rec)
            continue
        # chromadb's default distance space here is squared L2, not (1 - cosine).
        # For unit-normalised embeddings ||a-b||^2 == 2 - 2*cos(a,b), which is
        # exactly double (1 - cos) -- confirmed empirically: an earlier version
        # of this script reported "1 - cos" and its replication-check mean came
        # out at precisely half the original stored value (0.0812 vs 0.1624).
        # argmax(cos) still correctly identifies the nearest neighbour either
        # way (it's a monotonic rescaling); only the reported magnitude needed
        # the factor of 2.
        rep_sim, rep_doc = query_against(replication_mask, local_i)
        replication_records.append(
            {"file": rec["file"], "top1_distance": 2.0 * (1.0 - rep_sim), "top1_caption": rep_doc}
        )
        corr_sim, corr_doc = query_against(fixed_mask, local_i)
        corrected_records.append(
            {"file": rec["file"], "top1_distance": 2.0 * (1.0 - corr_sim), "top1_caption": corr_doc}
        )

    rep_dists = [r["top1_distance"] for r in replication_records]
    print(f"\nReplication check (same held-out setup as the original run, twins NOT yet excluded): "
          f"mean={statistics.mean(rep_dists):.4f}  median={statistics.median(rep_dists):.4f}")
    print(f"  (original stored result was mean={after['indian']['top1_distance_mean']:.4f}  "
          f"median={after['indian']['top1_distance_median']:.4f} -- should match closely)")

    orig_dists = [r["top1_distance"] for r in query_records]
    corr_dists = [r["top1_distance"] for r in corrected_records]
    print(f"\nOriginal stored (twins NOT excluded):  mean={statistics.mean(orig_dists):.4f}  "
          f"median={statistics.median(orig_dists):.4f}")
    print(f"Corrected (same held-out set + twins excluded):     mean={statistics.mean(corr_dists):.4f}  "
          f"median={statistics.median(corr_dists):.4f}")

    coco_baseline_mean = after["coco"]["top1_distance_mean"]
    corrected_shift_pct = (statistics.mean(corr_dists) / coco_baseline_mean - 1) * 100
    print(f"\nCorrected Indian top-1 distance is {corrected_shift_pct:+.1f}% vs COCO's own "
          f"baseline ({coco_baseline_mean:.4f}) -- was {after['shift_pct']:+.1f}% uncorrected.")

    with open(args.out, "w") as f:
        json.dump(
            {
                "threshold": args.threshold,
                "n_indian": n,
                "n_flagged": n_flagged,
                "n_exact_dupes": exact_dupes,
                "n_clusters": len(multi_clusters),
                "cluster_sizes": sizes,
                "flagged_in_query_sample": flagged_in_sample,
                "query_sample_size": len(query_image_ids),
                "replication_mean": statistics.mean(rep_dists),
                "replication_median": statistics.median(rep_dists),
                "original_mean": statistics.mean(orig_dists),
                "original_median": statistics.median(orig_dists),
                "corrected_mean": statistics.mean(corr_dists),
                "corrected_median": statistics.median(corr_dists),
                "coco_baseline_mean": coco_baseline_mean,
                "corrected_shift_pct": corrected_shift_pct,
                "original_shift_pct": after["shift_pct"],
                "corrected_records": corrected_records,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
