"""
Caption aggregation: merges retrieved caption blocks into one context string.
"""
from typing import List


def aggregate_captions(caption_blocks: List[str], max_length: int = 500) -> str:
    """Dedupe retrieved caption blocks at the sentence level, order-preserving.

    Each entry in ``caption_blocks`` may itself be several captions joined
    by ``' '`` (as produced by a top-k retrieval query). Deduping at the
    *word* level would turn the result into an ungrammatical word-bag
    (e.g. "dog runs the grass cat sits") and wreck downstream text-quality
    metrics; this dedupes whole sentences instead, using a
    normalised-lowercase key, and truncates at a sentence boundary if the
    joined result exceeds ``max_length``.
    """
    seen_norm = set()
    unique_sentences: List[str] = []

    for block in caption_blocks:
        if not block:
            continue
        pieces = [p.strip() for p in block.replace('. ', '.\n').split('\n')]
        for piece in pieces:
            if not piece:
                continue
            norm = ' '.join(piece.lower().split()).rstrip('.')
            if norm in seen_norm or len(norm) < 4:
                continue
            seen_norm.add(norm)
            unique_sentences.append(piece if piece.endswith('.') else piece + '.')

    aggregated = ' '.join(unique_sentences)

    if len(aggregated) > max_length:
        aggregated = aggregated[:max_length].rsplit('.', 1)[0] + '.'

    return aggregated
