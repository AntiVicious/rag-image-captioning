"""
Tests for src/cli.py argument parsing and config building — never invokes
Pipeline.setup()/caption_image(), so no real model/database work happens.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.cli import _build_config, build_parser  # noqa: E402


def test_no_subcommand_exits():
    parser = build_parser()
    try:
        parser.parse_args([])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit when no subcommand is given")


def test_caption_requires_image():
    parser = build_parser()
    try:
        parser.parse_args(["caption"])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit when --image is missing")


def test_caption_parses_image_and_skip_detr():
    parser = build_parser()
    args = parser.parse_args(["caption", "--image", "foo.jpg", "--skip-detr"])
    assert args.command == "caption"
    assert args.image == "foo.jpg"
    assert args.skip_detr is True


def test_build_config_overrides_backend():
    parser = build_parser()
    args = parser.parse_args(["--backend", "blip", "setup"])
    config = _build_config(args)
    assert config.caption_backend == "blip"


def test_build_config_defaults_to_configs_own_default():
    parser = build_parser()
    args = parser.parse_args(["setup"])
    config = _build_config(args)
    assert config.caption_backend == "retrieval"


def test_build_config_disables_detr_on_skip_flag():
    parser = build_parser()
    args = parser.parse_args(["caption", "--image", "foo.jpg", "--skip-detr"])
    config = _build_config(args)
    assert config.enable_segmentation is False
    assert config.enable_object_detection is False


def test_build_config_leaves_detr_disabled_by_default():
    """Config.enable_segmentation/enable_object_detection default to False
    (see src/config.py) since the ablation in scripts/evaluate.py shows
    crop-augmented retrieval scores worse than retrieval-only at far higher
    latency. --skip-detr is therefore a no-op today, kept for explicitness
    and forward-compat if an opt-in crops flag is added later."""
    parser = build_parser()
    args = parser.parse_args(["caption", "--image", "foo.jpg"])
    config = _build_config(args)
    assert config.enable_segmentation is False
    assert config.enable_object_detection is False


def test_build_config_overrides_coco_paths():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--coco-img-dir",
            "./coco/val2017",
            "--coco-ann-file",
            "./coco/annotations/captions_val2017.json",
            "build-db",
        ]
    )
    config = _build_config(args)
    assert config.coco_img_dir == "./coco/val2017"
    assert config.coco_ann_file == "./coco/annotations/captions_val2017.json"


def test_build_config_defaults_to_configs_own_coco_paths():
    parser = build_parser()
    args = parser.parse_args(["build-db"])
    config = _build_config(args)
    assert config.coco_img_dir == "./coco/train2017"
    assert config.coco_ann_file == "./coco/annotations/captions_train2017.json"


def test_build_config_overrides_chroma_db_dir():
    parser = build_parser()
    args = parser.parse_args(["--chroma-db-dir", "./chroma_db_scale_test", "build-db"])
    config = _build_config(args)
    assert config.chroma_db_dir == "./chroma_db_scale_test"


def test_build_config_defaults_to_configs_own_chroma_db_dir():
    parser = build_parser()
    args = parser.parse_args(["build-db"])
    config = _build_config(args)
    assert config.chroma_db_dir == "./chroma_db"


CASES = [
    test_no_subcommand_exits,
    test_caption_requires_image,
    test_caption_parses_image_and_skip_detr,
    test_build_config_overrides_backend,
    test_build_config_defaults_to_configs_own_default,
    test_build_config_disables_detr_on_skip_flag,
    test_build_config_leaves_detr_disabled_by_default,
    test_build_config_overrides_coco_paths,
    test_build_config_defaults_to_configs_own_coco_paths,
    test_build_config_overrides_chroma_db_dir,
    test_build_config_defaults_to_configs_own_chroma_db_dir,
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
