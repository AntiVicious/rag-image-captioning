"""
Fast, dependency-free tests for src/caption_aggregation.py.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.caption_aggregation import aggregate_captions  # noqa: E402


def test_no_word_level_dedup():
    # Sentences sharing words must stay intact, not collapse into a word-bag.
    blocks = ["a dog runs in the grass.", "a cat sits in the grass."]
    result = aggregate_captions(blocks)
    assert "a dog runs in the grass." in result
    assert "a cat sits in the grass." in result


def test_sentence_level_dedup():
    blocks = [
        "A dog runs in the grass. A cat sits in the grass.",
        "a dog runs in the grass.",
    ]
    result = aggregate_captions(blocks)
    assert result.lower().count("a dog runs in the grass.") == 1
    assert "a cat sits in the grass." in result.lower()


def test_order_preserved():
    blocks = ["first sentence here.", "second sentence here. third sentence here."]
    result = aggregate_captions(blocks)
    idx_first = result.find("first sentence here.")
    idx_second = result.find("second sentence here.")
    idx_third = result.find("third sentence here.")
    assert -1 not in (idx_first, idx_second, idx_third)
    assert idx_first < idx_second < idx_third


def test_truncation_at_sentence_boundary():
    long_sentences = [f"this is sentence number {i} in a long caption block." for i in range(50)]
    result = aggregate_captions(long_sentences, max_length=500)
    assert len(result) <= 500
    assert result.endswith(".")


def test_empty_and_short_fragments_dropped():
    blocks = ["", None, "hi.", "a valid sentence here."]
    result = aggregate_captions(blocks)
    assert "hi." not in result
    assert "a valid sentence here." in result


CASES = [
    test_no_word_level_dedup,
    test_sentence_level_dedup,
    test_order_preserved,
    test_truncation_at_sentence_boundary,
    test_empty_and_short_fragments_dropped,
]


def main() -> int:
    failures = 0
    for case in CASES:
        try:
            case()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {case.__name__}: {e}")
        else:
            print(f"PASS {case.__name__}")
    if failures:
        print(f"\n{failures}/{len(CASES)} tests failed")
        return 1
    print(f"\nAll {len(CASES)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
