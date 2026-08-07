"""
Throwaway diagnostic, round 3: is chromadb .get()'s row order stable across
separate calls / separate include lists? quantize_index.py fetches
include=["embeddings"]; measure_quantization_downstream.py fetches
include=["embeddings", "documents"]. Both then do
rng.choice(n, size=200, replace=False) with the SAME seed to pick query
rows -- but that samples POSITIONAL indices, not ids. If .get()'s row order
differs between the two calls, the same seed would silently select
different actual images in each script, which would fully explain a large
divergence in results despite seemingly-identical logic.
"""

import chromadb

client = chromadb.PersistentClient(path="/app/chroma_db_scale_test")
collection = client.get_collection("coco_clip_embeddings")

n_fetch = 100_000

result_a = collection.get(limit=n_fetch, include=["embeddings"])
ids_a = result_a["ids"]

result_b = collection.get(limit=n_fetch, include=["embeddings", "documents"])
ids_b = result_b["ids"]

result_c = collection.get(limit=n_fetch, include=["embeddings"])
ids_c = result_c["ids"]

print(f"len(ids_a)={len(ids_a)}  len(ids_b)={len(ids_b)}  len(ids_c)={len(ids_c)}")
print(f"first 5 ids_a: {ids_a[:5]}")
print(f"first 5 ids_b: {ids_b[:5]}")
print(f"first 5 ids_c: {ids_c[:5]}")

print(f"\nids_a == ids_b (same order, different include list)? {ids_a == ids_b}")
print(f"ids_a == ids_c (same order, same include list, separate call)? {ids_a == ids_c}")

if ids_a != ids_b:
    n_diff = sum(1 for x, y in zip(ids_a, ids_b) if x != y)
    print(f"  positions differing between a/b: {n_diff}/{len(ids_a)}")
    for i in range(len(ids_a)):
        if ids_a[i] != ids_b[i]:
            print(f"  first differing position: {i}: a={ids_a[i]!r} b={ids_b[i]!r}")
            break

if ids_a != ids_c:
    n_diff = sum(1 for x, y in zip(ids_a, ids_c) if x != y)
    print(f"  positions differing between a/c: {n_diff}/{len(ids_a)}")
    for i in range(len(ids_a)):
        if ids_a[i] != ids_c[i]:
            print(f"  first differing position: {i}: a={ids_a[i]!r} c={ids_c[i]!r}")
            break
