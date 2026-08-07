"""
Corrected, consolidated distribution-shift measurement.

Supersedes scripts/measure_distribution_shift.py + scripts/audit_near_duplicates.py
for this specific before/after/dedup-corrected number, fixing a real scope bug
found while preparing hand-labeled references: the original "after" query
sample (dist_shift_output/distribution_shift_after2.json) was drawn ONLY from
the --indian-img-dir the run happened to point at, which was the Indian FOOD
subset -- so the entire "-53.8% / -36.1%" distribution-shift result to date
represents food images only, not the full 12,538-image Indian-context set
(7,768 of which are Wikimedia Commons photos: festivals, weddings, monuments,
etc., NOT food). This script draws a single query sample uniformly from the
WHOLE merged annotation set, so both subsets are represented proportionally.

Operates entirely on embeddings already stored in ChromaDB (chroma_db_augmented
has both COCO and Indian rows; the pre-augmentation COCO-only index is used
for the "before" measurement) -- no image files, no CLIP model needed, so this
runs in seconds, not minutes.

Usage (inside the eval/app Docker image, for a matching chromadb version):
    docker run --rm \\
      -v D:\\chroma_db_augmented:/app/chroma_db_augmented \\
      -v <clone>/chroma_db:/app/chroma_db_before \\
      -v <repo>/dist_shift_output:/app/dist_shift_output \\
      rag-image-captioning:eval python scripts/measure_distribution_shift_v2.py
"""

import argparse
import json
import re
import statistics

import numpy as np

INDIAN_ID_RE = re.compile(r"^(\d+)_\1$")


def fetch_all(collection):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented-db", default="/app/chroma_db_augmented")
    parser.add_argument("--before-db", default="/app/chroma_db_before")
    parser.add_argument("--collection-name", default="coco_clip_embeddings")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=float, default=0.95)
    parser.add_argument("--out", default="/app/dist_shift_output/distribution_shift_v2.json")
    args = parser.parse_args()

    import chromadb

    aug_client = chromadb.PersistentClient(path=args.augmented_db)
    aug_coll = aug_client.get_collection(args.collection_name)
    print(f"Fetching all {aug_coll.count()} embeddings from augmented index...")
    ids, embeddings, documents = fetch_all(aug_coll)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-8, None)
    documents = np.array(documents)

    is_indian = np.array([bool(INDIAN_ID_RE.match(id_)) for id_ in ids])
    indian_idx = np.where(is_indian)[0]
    coco_idx = np.where(~is_indian)[0]
    print(f"Augmented index: {len(ids)} total  ({len(indian_idx)} Indian, {len(coco_idx)} COCO)")

    before_client = chromadb.PersistentClient(path=args.before_db)
    before_coll = before_client.get_collection(args.collection_name)
    before_ids, before_embeddings, _ = fetch_all(before_coll)
    before_norms = np.linalg.norm(before_embeddings, axis=1, keepdims=True)
    before_embeddings = before_embeddings / np.clip(before_norms, 1e-8, None)
    print(f"Pre-augmentation (COCO-only) index: {len(before_ids)} embeddings")

    # --- sample 200 Indian images uniformly across the FULL merged set ---
    indian_image_ids = sorted({int(ids[i].split("_")[0]) for i in indian_idx})
    rng = np.random.RandomState(args.seed)
    sample_indian_ids = rng.choice(indian_image_ids, size=min(args.num_samples, len(indian_image_ids)), replace=False)
    n_food = int((sample_indian_ids < 91000000).sum())
    n_commons = int((sample_indian_ids >= 91000000).sum())
    print(f"Indian query sample: {len(sample_indian_ids)} images ({n_food} food, {n_commons} commons)")

    # --- sample 200 COCO images (by unique image id, for the baseline) ---
    coco_image_ids = sorted({int(ids[i].split("_")[0]) for i in coco_idx})
    sample_coco_ids = rng.choice(coco_image_ids, size=min(args.num_samples, len(coco_image_ids)), replace=False)

    indian_pos = {int(ids[i].split("_")[0]): i for i in indian_idx}
    coco_pos_before = {int(before_ids[i].split("_")[0]): i for i in range(len(before_ids))}

    # --- BEFORE: Indian sample vs the pre-augmentation COCO-only index ---
    before_records = []
    for img_id in sample_indian_ids:
        emb = embeddings[indian_pos[int(img_id)]]
        sims = before_embeddings @ emb
        best_sim = float(sims.max())
        before_records.append({"image_id": int(img_id), "top1_distance": 2.0 * (1.0 - best_sim)})
    before_dists = [r["top1_distance"] for r in before_records]

    # --- COCO baseline: held-out (exclude the 200 sampled COCO images' own rows) ---
    coco_query_gids = {gid for gid in before_ids if int(gid.split("_")[0]) in set(sample_coco_ids.tolist())}
    coco_mask = np.array([gid not in coco_query_gids for gid in before_ids])
    coco_baseline_records = []
    for img_id in sample_coco_ids:
        # a query image's embedding is identical across all its caption rows;
        # use the first row found for this image_id
        row = next(i for i, gid in enumerate(before_ids) if int(gid.split("_")[0]) == img_id)
        emb = before_embeddings[row]
        sims = before_embeddings[coco_mask] @ emb
        best_sim = float(sims.max())
        coco_baseline_records.append({"image_id": int(img_id), "top1_distance": 2.0 * (1.0 - best_sim)})
    coco_baseline_dists = [r["top1_distance"] for r in coco_baseline_records]

    # --- near-duplicate detection within the full Indian subset ---
    print("Computing pairwise similarity within the Indian subset for near-dup detection...")
    indian_emb_all = embeddings[indian_idx]
    n = len(indian_idx)
    max_sim = np.zeros(n, dtype=np.float32)
    near_dup_pairs = []
    chunk = 1000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims_chunk = indian_emb_all[start:end] @ indian_emb_all.T
        for local_i, gi in enumerate(range(start, end)):
            row = sims_chunk[local_i].copy()
            row[gi] = -1.0
            max_sim[gi] = row.max()
            hits = np.where(row >= args.near_dup_threshold)[0]
            for j in hits:
                if j > gi:
                    near_dup_pairs.append((gi, int(j)))
    clusters = union_find_clusters(near_dup_pairs, n)
    indian_global_ids = [ids[i] for i in indian_idx]

    # --- AFTER (uncorrected): held-out index, twins NOT excluded ---
    base_exclude = set()
    sample_indian_id_set = set(int(x) for x in sample_indian_ids)
    sample_coco_id_set = set(int(x) for x in sample_coco_ids)
    for gid in ids:
        img_id = int(gid.split("_")[0])
        if img_id in sample_indian_id_set or img_id in sample_coco_id_set:
            base_exclude.add(gid)

    # --- AFTER (corrected): also exclude near-dup twins of each query image ---
    twin_exclude = set()
    # map global embeddings-row -> position within indian_idx array
    global_to_local = {gi: li for li, gi in enumerate(indian_idx)}
    for img_id in sample_indian_ids:
        global_i = indian_pos[int(img_id)]
        local_i = global_to_local[global_i]
        cluster_root = next((root for root, members in clusters.items() if local_i in members), None)
        if cluster_root is None:
            continue
        for j in clusters[cluster_root]:
            twin_exclude.add(indian_global_ids[j])

    after_mask = np.array([gid not in base_exclude for gid in ids])
    after_corrected_mask = np.array([gid not in (base_exclude | twin_exclude) for gid in ids])
    print(
        f"Held-out index: {len(ids)} -> {after_mask.sum()} (base) -> {after_corrected_mask.sum()} "
        f"(+{len(twin_exclude - base_exclude)} near-dup twins excluded)"
    )

    after_records = []
    after_corrected_records = []
    for img_id in sample_indian_ids:
        emb = embeddings[indian_pos[int(img_id)]]
        sims_after = embeddings[after_mask] @ emb
        best_after = float(sims_after.max())
        after_records.append({"image_id": int(img_id), "top1_distance": 2.0 * (1.0 - best_after)})

        sims_corr = embeddings[after_corrected_mask] @ emb
        best_corr = float(sims_corr.max())
        best_doc = documents[after_corrected_mask][int(np.argmax(sims_corr))]
        after_corrected_records.append(
            {"image_id": int(img_id), "top1_distance": 2.0 * (1.0 - best_corr), "top1_caption": str(best_doc)}
        )

    after_dists = [r["top1_distance"] for r in after_records]
    after_corr_dists = [r["top1_distance"] for r in after_corrected_records]

    coco_mean = statistics.mean(coco_baseline_dists)
    print(f"\n=== Results (N={len(sample_indian_ids)}, {n_food} food + {n_commons} commons) ===")
    print(f"COCO baseline:            mean={coco_mean:.4f}  median={statistics.median(coco_baseline_dists):.4f}")
    print(f"Indian BEFORE:            mean={statistics.mean(before_dists):.4f}  "
          f"median={statistics.median(before_dists):.4f}  "
          f"({100 * (statistics.mean(before_dists) / coco_mean - 1):+.1f}% vs COCO)")
    print(f"Indian AFTER (uncorrected): mean={statistics.mean(after_dists):.4f}  "
          f"median={statistics.median(after_dists):.4f}  "
          f"({100 * (statistics.mean(after_dists) / coco_mean - 1):+.1f}% vs COCO)")
    print(f"Indian AFTER (dedup-corrected): mean={statistics.mean(after_corr_dists):.4f}  "
          f"median={statistics.median(after_corr_dists):.4f}  "
          f"({100 * (statistics.mean(after_corr_dists) / coco_mean - 1):+.1f}% vs COCO)")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n_samples": len(sample_indian_ids),
                "n_food": n_food,
                "n_commons": n_commons,
                "coco_baseline_mean": coco_mean,
                "coco_baseline_median": statistics.median(coco_baseline_dists),
                "before_mean": statistics.mean(before_dists),
                "before_median": statistics.median(before_dists),
                "after_uncorrected_mean": statistics.mean(after_dists),
                "after_uncorrected_median": statistics.median(after_dists),
                "after_corrected_mean": statistics.mean(after_corr_dists),
                "after_corrected_median": statistics.median(after_corr_dists),
                "sample_image_ids": [int(x) for x in sample_indian_ids],
                "after_corrected_records": after_corrected_records,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
