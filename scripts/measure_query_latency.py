"""
Measure ChromaDB query latency (p50/p95) at a given index scale, using
real CLIP embeddings from a sample of images.

Usage:
    python scripts/measure_query_latency.py --chroma-db-dir ./chroma_db_scale_test \
        --coco-img-dir ./coco/train2017 --num-queries 200
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Measure ChromaDB query latency at scale")
    parser.add_argument("--chroma-db-dir", required=True)
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = Config(chroma_db_dir=args.chroma_db_dir)
    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)

    print("Loading CLIP...")
    model_manager.load_clip_model()
    db_manager.initialize()
    stats = db_manager.get_stats()
    print(f"Index: {stats['total_embeddings']} embeddings in '{stats['collection_name']}'")

    filenames = sorted(f for f in os.listdir(args.coco_img_dir) if f.endswith(".jpg"))
    sample = filenames[:: max(1, len(filenames) // args.num_queries)][: args.num_queries]
    print(f"Sampling {len(sample)} images from {len(filenames)} available")

    query_latencies = []
    encode_latencies = []
    for fn in sample:
        img_path = os.path.join(args.coco_img_dir, fn)
        tensor = preprocessor.preprocess_for_clip(img_path, model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)

        t0 = time.perf_counter()
        embedding = model_manager.encode_image(tensor)
        t1 = time.perf_counter()
        query_embedding = embedding.cpu().numpy()[0].tolist()
        db_manager.query_similar([query_embedding], args.top_k)
        t2 = time.perf_counter()

        encode_latencies.append(t1 - t0)
        query_latencies.append(t2 - t1)

    def pct(vals, p):
        vals_sorted = sorted(vals)
        idx = min(int(len(vals_sorted) * p), len(vals_sorted) - 1)
        return vals_sorted[idx]

    print(f"\n=== Chroma query latency ({stats['total_embeddings']} embeddings, N={len(sample)}) ===")
    print(f"  p50: {pct(query_latencies, 0.50) * 1000:.2f} ms")
    print(f"  p95: {pct(query_latencies, 0.95) * 1000:.2f} ms")
    print(f"  p99: {pct(query_latencies, 0.99) * 1000:.2f} ms")
    print(f"  mean: {statistics.mean(query_latencies) * 1000:.2f} ms")
    print("\n=== CLIP encode latency (for reference, same run) ===")
    print(f"  p50: {pct(encode_latencies, 0.50) * 1000:.2f} ms")
    print(f"  mean: {statistics.mean(encode_latencies) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
