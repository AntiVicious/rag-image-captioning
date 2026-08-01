"""
Tests for src/image_preprocessing.py that verify the lazy-loading contract
and the crop math directly, WITHOUT downloading the DETR checkpoints
(~170MB+ each). A real segmentation/detection run belongs in the smoke test.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from src.config import Config
from src.image_preprocessing import ImagePreprocessor


def test_construction_does_not_load_detection_models():
    preprocessor = ImagePreprocessor(Config())
    assert preprocessor.segmentation_model is None
    assert preprocessor.object_detection_model is None


def test_segmentation_disabled_returns_empty_without_loading_model():
    preprocessor = ImagePreprocessor(Config(enable_segmentation=False))
    crops = preprocessor.get_segmentation_crops(Image.new("RGB", (100, 100)))
    assert crops == []
    assert preprocessor.segmentation_model is None


def test_object_detection_disabled_returns_empty_without_loading_model():
    preprocessor = ImagePreprocessor(Config(enable_object_detection=False))
    crops, detected = preprocessor.get_object_detection_crops(Image.new("RGB", (100, 100)))
    assert crops == []
    assert detected == []
    assert preprocessor.object_detection_model is None


def test_pad_and_crop_respects_min_crop_size():
    preprocessor = ImagePreprocessor(Config(min_crop_size=64, crop_padding=0))
    image = Image.new("RGB", (200, 200))
    assert preprocessor._pad_and_crop(image, 0, 0, 10, 10) is None  # too small
    crop = preprocessor._pad_and_crop(image, 0, 0, 100, 100)
    assert crop is not None
    assert crop.size == (100, 100)


def test_pad_and_crop_clips_to_image_bounds():
    preprocessor = ImagePreprocessor(Config(min_crop_size=10, crop_padding=50))
    image = Image.new("RGB", (100, 100))
    crop = preprocessor._pad_and_crop(image, 40, 40, 60, 60)
    assert crop is not None
    assert crop.size == (100, 100)  # padding pushed past bounds, clipped


def test_preprocess_for_clip_converts_and_unsqueezes():
    import torch

    preprocessor = ImagePreprocessor(Config())
    image = Image.new("RGB", (32, 32))
    fake_clip_preprocess = lambda img: torch.zeros(3, 32, 32)
    tensor = preprocessor.preprocess_for_clip(image, fake_clip_preprocess)
    assert tuple(tensor.shape) == (1, 3, 32, 32)


CASES = [
    test_construction_does_not_load_detection_models,
    test_segmentation_disabled_returns_empty_without_loading_model,
    test_object_detection_disabled_returns_empty_without_loading_model,
    test_pad_and_crop_respects_min_crop_size,
    test_pad_and_crop_clips_to_image_bounds,
    test_preprocess_for_clip_converts_and_unsqueezes,
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
