"""
Tests for src/models.py that verify the lazy-loading contract WITHOUT
triggering an actual CLIP weights download (~350MB) — that would make CI
slow and flaky on network hiccups. A real download is exercised manually /
in the smoke test instead, once one exists.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import Config
from src.models import ModelManager


def test_construction_does_not_load_clip():
    manager = ModelManager(Config())
    assert manager.clip_model is None
    assert manager.clip_preprocess is None


def test_encode_image_before_load_raises():
    import torch

    manager = ModelManager(Config())
    try:
        manager.encode_image(torch.zeros(1, 3, 224, 224))
    except RuntimeError:
        pass
    else:
        raise AssertionError("encode_image should raise before load_clip_model() is called")


def test_device_resolves_without_error():
    manager = ModelManager(Config())
    assert manager.device in ("cpu", "cuda")


CASES = [
    test_construction_does_not_load_clip,
    test_encode_image_before_load_raises,
    test_device_resolves_without_error,
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
