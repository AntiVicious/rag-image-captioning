"""
Fast tests for src/config.py. No ML deps: Config only imports torch lazily
inside the .device property, so constructing/inspecting it is cheap.
"""

import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import Config  # noqa: E402


def test_defaults_favor_no_gpu_by_default():
    # This project is developed/tested on machines without a GPU; retrieval
    # mode must be the default so tests/CI never need BLIP-2 weights.
    config = Config()
    assert config.caption_backend == "retrieval"


def test_validate_paths_false_when_coco_absent():
    config = Config(coco_img_dir="./does-not-exist", coco_ann_file="./also-does-not-exist.json")
    assert config.validate_paths() is False


def test_create_output_dirs_creates_chroma_dir():
    tmp_dir = tempfile.mkdtemp()
    try:
        chroma_dir = os.path.join(tmp_dir, "chroma_db")
        config = Config(chroma_db_dir=chroma_dir)
        assert not os.path.exists(chroma_dir)
        config.create_output_dirs()
        assert os.path.isdir(chroma_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_configs_are_independent_instances():
    a = Config(caption_backend="blip")
    b = Config()
    assert a.caption_backend == "blip"
    assert b.caption_backend == "retrieval"


CASES = [
    test_defaults_favor_no_gpu_by_default,
    test_validate_paths_false_when_coco_absent,
    test_create_output_dirs_creates_chroma_dir,
    test_configs_are_independent_instances,
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
