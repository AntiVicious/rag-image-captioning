"""
Tests for src/rag_retrieval.py using fake collaborators (no real CLIP,
ChromaDB, or DETR involved) — RAGRetriever takes its dependencies as
constructor args specifically so this is possible.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import Config  # noqa: E402
from src.rag_retrieval import RAGRetriever  # noqa: E402


class FakeModelManager:
    def __init__(self, embedding):
        self.device = "cpu"
        self.clip_preprocess = lambda img: torch.zeros(3, 2, 2)
        self._embedding = embedding

    def encode_image(self, tensor):
        # real encode_image accepts a batch of N variant tensors in one
        # call; the fake mirrors that by returning N copies of the fixed
        # embedding, one per row in the input batch.
        return self._embedding.repeat(tensor.shape[0], 1)


class FakeDatabaseManager:
    """Returns one caption list per query embedding, all from a single
    batched query_similar call (matching the real batched API)."""

    def __init__(self, documents_by_query):
        self._documents_by_query = documents_by_query

    def query_similar(self, query_embeddings, top_k):
        return {"documents": self._documents_by_query[: len(query_embeddings)]}


class FakePreprocessor:
    def __init__(self, variants=None):
        self._variants = variants

    def preprocess_for_clip(self, image, clip_preprocess):
        return torch.zeros(1, 3, 2, 2)

    def get_all_variants(self, image):
        return self._variants


def test_basic_retrieval_returns_joined_documents():
    model_manager = FakeModelManager(torch.tensor([[1.0, 0.0]]))
    db_manager = FakeDatabaseManager([["a cat sits.", "a cat lies."]])
    retriever = RAGRetriever(Config(), model_manager, db_manager, FakePreprocessor())

    result = retriever.retrieve("fake.jpg", use_advanced=False)

    assert result["mode"] == "basic"
    assert "a cat sits." in result["aggregated_caption"]
    assert "a cat lies." in result["aggregated_caption"]


def test_advanced_retrieval_aggregates_and_dedupes_across_variants():
    model_manager = FakeModelManager(torch.tensor([[1.0, 0.0]]))
    db_manager = FakeDatabaseManager(
        [
            ["a dog runs in the grass."],  # original
            ["a dog runs in the grass."],  # segment 1 (duplicate of original)
            ["a close-up of a dog's face."],  # object 1
        ]
    )
    variants = {
        "original": Image.new("RGB", (10, 10)),
        "segmentation_crops": [Image.new("RGB", (10, 10))],
        "object_detection_crops": [Image.new("RGB", (10, 10))],
        "detected_objects": [{"label_id": 1, "score": 0.9, "box": [0, 0, 10, 10]}],
    }
    retriever = RAGRetriever(Config(), model_manager, db_manager, FakePreprocessor(variants))

    result = retriever.retrieve(variants["original"], use_advanced=True)

    assert result["mode"] == "advanced"
    assert result["variants_processed"] == 3
    assert result["aggregated_caption"].lower().count("a dog runs in the grass.") == 1
    assert "a close-up of a dog's face." in result["aggregated_caption"]
    assert result["detected_objects"] == variants["detected_objects"]


def test_raises_when_clip_not_loaded():
    model_manager = FakeModelManager(torch.tensor([[1.0]]))
    model_manager.clip_preprocess = None
    retriever = RAGRetriever(Config(), model_manager, FakeDatabaseManager([["x"]]), FakePreprocessor())

    try:
        retriever.retrieve("fake.jpg")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when clip_preprocess is None")


CASES = [
    test_basic_retrieval_returns_joined_documents,
    test_advanced_retrieval_aggregates_and_dedupes_across_variants,
    test_raises_when_clip_not_loaded,
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
