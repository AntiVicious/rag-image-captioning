"""
Strategies for choosing one caption from the retrieved candidates.
(caption_aggregation.py instead combines several candidates, which scores
lower; see results/selection_strategy_ablation.csv.)

    top1   -- the candidate closest to the query image (lowest ChromaDB
              distance). No extra model calls.
    medoid -- the candidate most similar, on average, to the other
              candidates, using CLIP text embeddings. Needs one extra
              encode_text() call and scores higher than top1 when crops
              add more candidates.
"""

from typing import List, Sequence


def select_top1(
    documents_per_variant: Sequence[List[str]], distances_per_variant: Sequence[List[float]]
) -> str:
    """Candidate with the lowest ChromaDB distance across all variants.

    Each variant's results are sorted by distance, so only each variant's
    first candidate is compared."""
    best_doc, best_dist = None, None
    for docs, dists in zip(documents_per_variant, distances_per_variant):
        if not docs:
            continue
        if best_dist is None or dists[0] < best_dist:
            best_dist, best_doc = dists[0], docs[0]
    return best_doc or ""


def select_medoid(documents_per_variant: Sequence[List[str]], model_manager) -> str:
    candidates = [d for docs in documents_per_variant for d in docs]
    if not candidates:
        return ""
    seen = set()
    unique_candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    embeddings = model_manager.encode_text(unique_candidates)  # (n, d), already normalised
    sim_matrix = embeddings @ embeddings.T  # cosine, since normalised
    n = len(unique_candidates)
    sim_matrix.fill_diagonal_(0.0)
    mean_sim = sim_matrix.sum(dim=1) / (n - 1)
    best_idx = int(mean_sim.argmax().item())
    return unique_candidates[best_idx]


def select_caption(
    documents_per_variant: Sequence[List[str]],
    distances_per_variant: Sequence[List[float]],
    mode: str,
    model_manager,
) -> str:
    if mode == "top1":
        return select_top1(documents_per_variant, distances_per_variant)
    if mode == "medoid":
        return select_medoid(documents_per_variant, model_manager)
    raise ValueError(f"Unknown selection mode: {mode!r} (expected 'top1' or 'medoid')")


def match_distance_for(
    documents_per_variant: Sequence[List[str]], distances_per_variant: Sequence[List[float]], document: str
):
    """ChromaDB distance of a selected document, found in the candidate lists.

    Used as the confidence signal: a large distance means the index probably
    has nothing similar to the query, so the caption is likely wrong. The
    medoid choice isn't necessarily the closest candidate, so its distance
    has to be looked up."""
    for docs, dists in zip(documents_per_variant, distances_per_variant):
        for doc, dist in zip(docs, dists):
            if doc == document:
                return dist
    return None
