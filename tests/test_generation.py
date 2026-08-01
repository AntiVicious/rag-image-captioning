"""
Tests for src/generation.py using fake collaborators — no real CLIP,
ChromaDB, or BLIP-2 weights involved.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import Config  # noqa: E402
from src.generation import CaptionGenerator  # noqa: E402
from src.rag_retrieval import RAGRetriever  # noqa: E402


class FakeDatabaseManager:
    def __init__(self, documents_by_call):
        self._documents_by_call = documents_by_call
        self._calls = 0

    def query_similar(self, query_embedding, top_k):
        docs = self._documents_by_call[self._calls]
        self._calls += 1
        return {"documents": [docs]}


class FakePreprocessor:
    def preprocess_for_clip(self, image, clip_preprocess):
        return torch.zeros(1, 3, 2, 2)

    def get_all_variants(self, image):
        raise AssertionError("advanced preprocessing should not be used in these tests")


class FakeBlip2Inputs(dict):
    def to(self, device):
        return self


class FakeBlip2Processor:
    def __call__(self, images, text, return_tensors):
        return FakeBlip2Inputs({"input_ids": torch.zeros(1, 3, dtype=torch.long)})

    def decode(self, ids, skip_special_tokens=True):
        return "Caption: a fake generated caption."


class FakeBlip2Model:
    def generate(self, **kwargs):
        return torch.zeros(1, 3, dtype=torch.long)


class FakeModelManager:
    def __init__(self, with_blip2: bool = False):
        self.device = "cpu"
        self.clip_preprocess = lambda img: torch.zeros(3, 2, 2)
        self._embedding = torch.tensor([[1.0, 0.0]])
        self.blip2_model = FakeBlip2Model() if with_blip2 else None
        self.blip2_processor = FakeBlip2Processor() if with_blip2 else None
        self.load_blip2_calls = 0

    def encode_image(self, tensor):
        return self._embedding

    def load_blip2_model(self):
        self.load_blip2_calls += 1
        self.blip2_model = FakeBlip2Model()
        self.blip2_processor = FakeBlip2Processor()


def _make_generator(caption_backend: str, model_manager=None):
    model_manager = model_manager or FakeModelManager()
    db_manager = FakeDatabaseManager([["a cat sits on a mat."]])
    config = Config(caption_backend=caption_backend)
    retriever = RAGRetriever(config, model_manager, db_manager, FakePreprocessor())
    return CaptionGenerator(config, model_manager, retriever), model_manager


def test_retrieval_backend_returns_aggregated_caption_as_is():
    generator, _ = _make_generator("retrieval")
    result = generator.generate(Image.new("RGB", (10, 10)), use_advanced=False)

    assert result["backend"] == "retrieval"
    assert result["generated_caption"] == "a cat sits on a mat."
    assert result["retrieved_context"] == "a cat sits on a mat."


def test_blip_backend_lazy_loads_and_strips_prefix():
    model_manager = FakeModelManager(with_blip2=False)
    generator, mm = _make_generator("blip", model_manager)

    result = generator.generate(Image.new("RGB", (10, 10)), use_advanced=False)

    assert mm.load_blip2_calls == 1  # was loaded on demand
    assert result["backend"] == "blip"
    assert result["generated_caption"] == "a fake generated caption."
    assert result["retrieved_context"] == "a cat sits on a mat."


def test_blip_backend_skips_reload_if_already_loaded():
    model_manager = FakeModelManager(with_blip2=True)
    generator, mm = _make_generator("blip", model_manager)

    generator.generate(Image.new("RGB", (10, 10)), use_advanced=False)

    assert mm.load_blip2_calls == 0


def test_unknown_backend_raises():
    generator, _ = _make_generator("not-a-real-backend")
    try:
        generator.generate(Image.new("RGB", (10, 10)), use_advanced=False)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown caption_backend")


CASES = [
    test_retrieval_backend_returns_aggregated_caption_as_is,
    test_blip_backend_lazy_loads_and_strips_prefix,
    test_blip_backend_skips_reload_if_already_loaded,
    test_unknown_backend_raises,
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
