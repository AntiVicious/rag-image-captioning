"""
Throwaway diagnostic for a warning evaluate.py raised on its own: the
retrieval-only ablation row (output-mode top1, n_results=captions_per_crop=3,
takes the first/closest of the 3) and the top1-verbatim baseline (n_results=1)
query the SAME held-out index with the SAME single-image embedding, and
should therefore always return the same top-1 document -- but their CIDEr
scores disagreed (0.5043 vs 0.4704) on the N=200 corrected ablation run.

Hypothesis: chromadb's HNSW approximate-nearest-neighbour search does not
guarantee the same top-1 result for different requested n_results (ef-search
scales with k), so asking for top-3 vs top-1 against the same index can
legitimately return a different closest match on ties/near-ties -- not a bug
in the top1-selection logic itself.

Usage: same mounts as scripts/evaluate.py.
"""

import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    build_image_path_index,
    load_annotations,
    pick_eval_split,
    prepare_eval_index,
)


def main():
    coco_img_dir = "/app/coco/val2017"
    coco_ann_file = "/app/coco/annotations/captions_val2017.json"
    source_db = "/app/chroma_db"
    eval_db = "/app/chroma_db_eval_diag"

    by_image = load_annotations(coco_ann_file)
    img_paths = build_image_path_index(coco_img_dir)
    eval_ids = pick_eval_split(by_image, 200, 42)

    config = prepare_eval_index(source_db, eval_db, by_image, eval_ids)
    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)
    model_manager.load_clip_model()
    db_manager.initialize()

    mismatches = []
    for img_id in sorted(eval_ids):
        img_path = img_paths.get(img_id)
        if not img_path:
            continue
        tensor = preprocessor.preprocess_for_clip(img_path, model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor)
        query_embedding = embedding.cpu().numpy().tolist()

        res_k1 = db_manager.query_similar(query_embedding, 1)
        res_k3 = db_manager.query_similar(query_embedding, 3)

        doc_k1 = (res_k1.get("documents") or [[]])[0]
        dist_k1 = (res_k1.get("distances") or [[]])[0]
        doc_k3 = (res_k3.get("documents") or [[]])[0]
        dist_k3 = (res_k3.get("distances") or [[]])[0]

        top1_from_k1 = doc_k1[0] if doc_k1 else None
        top1_from_k3 = doc_k3[0] if doc_k3 else None

        if top1_from_k1 != top1_from_k3:
            mismatches.append(
                {
                    "image_id": img_id,
                    "k1_doc": top1_from_k1,
                    "k1_dist": dist_k1[0] if dist_k1 else None,
                    "k3_doc": top1_from_k3,
                    "k3_dist": dist_k3[0] if dist_k3 else None,
                }
            )

    print(f"\n{len(mismatches)}/{len(eval_ids)} images: top-1 differs between n_results=1 and n_results=3 queries")
    for m in mismatches[:15]:
        print(f"  image_id={m['image_id']}")
        print(f"    k=1 top-1: dist={m['k1_dist']:.6f}  {m['k1_doc']!r}")
        print(f"    k=3 top-1: dist={m['k3_dist']:.6f}  {m['k3_doc']!r}")
        if m["k1_dist"] is not None and m["k3_dist"] is not None:
            print(f"    distance delta: {abs(m['k1_dist'] - m['k3_dist']):.6f}")


if __name__ == "__main__":
    main()
