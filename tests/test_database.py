"""
Tests for src/database.py against a real, throwaway ChromaDB instance in a
temp directory — no network access or real embeddings needed, just small
fake vectors, so this stays fast.
"""
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import Config
from src.database import DatabaseManager


def _make_manager(tmp_dir: str) -> DatabaseManager:
    config = Config(chroma_db_dir=os.path.join(tmp_dir, "chroma_db"))
    return DatabaseManager(config)


def test_initialize_creates_collection():
    tmp_dir = tempfile.mkdtemp()
    try:
        manager = _make_manager(tmp_dir)
        collection = manager.initialize()
        assert collection is not None
        assert manager.get_stats()["total_embeddings"] == 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_add_and_query_round_trip():
    tmp_dir = tempfile.mkdtemp()
    try:
        manager = _make_manager(tmp_dir)
        manager.add_embeddings(
            embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            documents=["a cat sits on a mat.", "a dog runs in a park."],
            ids=["img1_cap1", "img2_cap1"],
        )
        assert manager.get_stats()["total_embeddings"] == 2

        results = manager.query_similar([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results["documents"][0][0] == "a cat sits on a mat."
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_reopening_existing_collection_preserves_data():
    tmp_dir = tempfile.mkdtemp()
    try:
        config = Config(chroma_db_dir=os.path.join(tmp_dir, "chroma_db"))
        DatabaseManager(config).add_embeddings(
            embeddings=[[0.5, 0.5, 0.0, 0.0]], documents=["a bird flies."], ids=["img3_cap1"]
        )
        reopened = DatabaseManager(config)
        assert reopened.get_stats()["total_embeddings"] == 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


CASES = [
    test_initialize_creates_collection,
    test_add_and_query_round_trip,
    test_reopening_existing_collection_preserves_data,
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
