"""
Caption SELECTION strategies: choosing ONE caption out of a pool of
retrieved candidates. Contrast with caption_aggregation.py, which
concatenates/dedupes MULTIPLE candidates into one block -- that's a
different (and, per the ablation in scripts/evaluate.py /
results/selection_strategy_ablation.csv, worse-scoring) strategy, kept
available as Config.selection_mode == "aggregated" for backward
compatibility.

medoid needs a live CLIP model (text-embedding similarity), so -- unlike
caption_aggregation.py -- this module is not import-safe without ML deps;
kept separate for that reason.

    top1   -- the candidate closest to the QUERY IMAGE (lowest ChromaDB
              distance). No extra model calls.
    medoid -- the candidate closest, on average, to every OTHER candidate
              in the retrieved pool -- a consensus pick, via CLIP text-
              embedding similarity, instead of "closest to the query
              image." Costs one extra encode_text() call over the pool,
              and empirically outperforms top1 once there's more than one
              variant's worth of candidates to pick from.
"""

from typing import List, Sequence


def select_top1(documents_per_variant: Sequence[List[str]], distances_per_variant: Sequence[List[float]]) -> str:
    """Global best candidate across every variant queried, by ChromaDB
    distance. Each variant's own results are already distance-sorted
    ascending, so only each variant's first candidate can possibly win."""
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
