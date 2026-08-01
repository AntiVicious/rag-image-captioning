"""
Tests for src/pipeline.py that verify Pipeline construction wires all
managers together WITHOUT loading any model or opening the database — every
collaborator it builds is itself lazy, so this just checks that stays true.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import Config
from src.pipeline import Pipeline


def test_construction_does_not_load_anything():
    pipeline = Pipeline(Config())
    assert pipeline.model_manager.clip_model is None
    assert pipeline.model_manager.blip2_model is None
    assert pipeline.db_manager.collection is None
    assert pipeline.preprocessor.segmentation_model is None
    assert pipeline.preprocessor.object_detection_model is None


def test_default_config_is_created_when_none_given():
    pipeline = Pipeline()
    assert isinstance(pipeline.config, Config)
    assert pipeline.config.caption_backend == "retrieval"


def test_collaborators_share_the_same_config_instance():
    config = Config(caption_backend="blip")
    pipeline = Pipeline(config)
    assert pipeline.model_manager.config is config
    assert pipeline.db_manager.config is config
    assert pipeline.preprocessor.config is config
    assert pipeline.retriever.config is config
    assert pipeline.generator.config is config


CASES = [
    test_construction_does_not_load_anything,
    test_default_config_is_created_when_none_given,
    test_collaborators_share_the_same_config_instance,
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
