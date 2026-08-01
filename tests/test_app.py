"""
Tests app.py's initial render using Streamlit's AppTest harness (headless —
no browser, no server). No image is uploaded and the pipeline is never
initialized, so this never touches CLIP/ChromaDB/BLIP-2.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_app_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(_REPO_ROOT, "app.py"))
    at.run(timeout=30)
    assert not at.exception


def test_app_shows_upload_prompt_before_any_file_is_uploaded():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(_REPO_ROOT, "app.py"))
    at.run(timeout=30)
    info_texts = [info.value for info in at.info]
    assert any("Upload an image" in text for text in info_texts)


def test_app_has_backend_selector_defaulting_to_retrieval():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(_REPO_ROOT, "app.py"))
    at.run(timeout=30)
    assert len(at.selectbox) == 1
    assert at.selectbox[0].value == "retrieval"


CASES = [
    test_app_renders_without_exception,
    test_app_shows_upload_prompt_before_any_file_is_uploaded,
    test_app_has_backend_selector_defaulting_to_retrieval,
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
