"""
Ablation evaluation: BLEU-4, METEOR, ROUGE-L, CIDEr on a held-out split of
COCO val2017, across four retrieval configurations:

    retrieval-only      -- CLIP query on the original image only
    +segmentation       -- + top segmentation crops (DETR panoptic)
    +object-detection   -- + top object-detection crops (DETR)
    all-seven           -- both crop types together

Held-out split (avoids retrieval self-match leakage): --num-eval-images
image_ids are sampled deterministically (--seed) from val2017 and their
embeddings are DELETED from a *copy* of chroma_db before scoring, so
retrieval can never return the query image's own captions. The remaining
images serve as the retrieval corpus. The live chroma_db used by the app
is never touched -- this script operates on ./chroma_db_eval.

Also records per-stage latency (DETR segmentation / DETR object detection /
CLIP encode / Chroma query / aggregate) as a side effect of the same run.

Usage:
    python scripts/evaluate.py --num-eval-images 200 --seed 42
"""

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.caption_aggregation import aggregate_captions  # noqa: E402
from src.config import Config  # noqa: E402
from src.database import DatabaseManager  # noqa: E402
from src.image_preprocessing import ImagePreprocessor  # noqa: E402
from src.models import ModelManager  # noqa: E402

ABLATIONS = [
    ("retrieval-only", False, False),
    ("+segmentation", True, False),
    ("+object-detection", False, True),
    ("all-seven", True, True),
]


def load_annotations(ann_file):
    """image_id -> [(ann_id, caption), ...], covering every caption COCO has for that image."""
    with open(ann_file) as f:
        data = json.load(f)
    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[ann["image_id"]].append((ann["id"], ann["caption"]))
    return by_image


def build_image_path_index(img_dir):
    return {
        int(fn.split(".")[0]): os.path.join(img_dir, fn) for fn in os.listdir(img_dir) if fn.endswith(".jpg")
    }


def pick_eval_split(by_image, num_eval, seed):
    all_ids = sorted(by_image.keys())
    rng = random.Random(seed)
    return set(rng.sample(all_ids, num_eval))


def prepare_eval_index(source_db_dir, eval_db_dir, by_image, eval_ids):
    """Copy chroma_db -> chroma_db_eval, then delete the eval images' own
    embeddings from the copy so retrieval can't self-match."""
    if os.path.exists(eval_db_dir):
        shutil.rmtree(eval_db_dir)
    shutil.copytree(source_db_dir, eval_db_dir)

    config = Config(chroma_db_dir=eval_db_dir)
    db_manager = DatabaseManager(config)
    db_manager.initialize()

    ids_to_delete = [f"{img_id}_{ann_id}" for img_id in eval_ids for ann_id, _ in by_image[img_id]]
    before = db_manager.get_stats()["total_embeddings"]
    db_manager.collection.delete(ids=ids_to_delete)
    after = db_manager.get_stats()["total_embeddings"]
    print(f"Index: {before} -> {after} embeddings after removing {len(eval_ids)} held-out images")
    return config


def instrumented_retrieve(image_path, config, model_manager, db_manager, preprocessor, timings):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    t0 = time.perf_counter()
    seg_crops = preprocessor.get_segmentation_crops(image) if config.enable_segmentation else []
    t1 = time.perf_counter()
    if config.enable_object_detection:
        obj_crops, _detected = preprocessor.get_object_detection_crops(image)
    else:
        obj_crops = []
    t2 = time.perf_counter()
    timings["detr_segmentation"].append(t1 - t0)
    timings["detr_object_detection"].append(t2 - t1)

    variants = [image] + list(seg_crops) + list(obj_crops)

    import torch

    t_a = time.perf_counter()
    tensors = [preprocessor.preprocess_for_clip(v, model_manager.clip_preprocess) for v in variants]
    batch = torch.cat(tensors, dim=0).to(model_manager.device)
    embeddings = model_manager.encode_image(batch)
    t_b = time.perf_counter()
    query_embeddings = embeddings.cpu().numpy().tolist()
    results = db_manager.query_similar(query_embeddings, config.captions_per_crop)
    t_c = time.perf_counter()

    documents_per_variant = results.get("documents") or [[] for _ in variants]
    blocks = [" ".join(docs) for docs in documents_per_variant]

    n = len(variants)
    timings["clip_encode_batch_total"].append(t_b - t_a)
    timings["clip_encode_per_variant"].append((t_b - t_a) / n)
    timings["chroma_query_batch_total"].append(t_c - t_b)
    timings["chroma_query_per_variant"].append((t_c - t_b) / n)
    timings["variants_per_image"].append(n)

    t_agg0 = time.perf_counter()
    aggregated = aggregate_captions(blocks)
    t_agg1 = time.perf_counter()
    timings["aggregate"].append(t_agg1 - t_agg0)

    return aggregated


def score_with_pycocoevalcap(predictions, by_image, eval_ids):
    """Returns (corpus-level scores, per-image scores keyed by image_id).

    Per-image alignment: gts/res are built by iterating sorted(eval_ids), and
    pycocoevalcap's scorers iterate gts.keys() (dict insertion order,
    preserved since Python 3.7) internally, so the per-image score lists
    they return line up with sorted(eval_ids) -- NOT the original (unordered)
    eval_ids set.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ordered_ids = sorted(eval_ids)
    gts_raw = {img_id: [{"caption": c} for _, c in by_image[img_id]] for img_id in ordered_ids}
    res_raw = {img_id: [{"caption": predictions[img_id]}] for img_id in ordered_ids}

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scores = {}
    per_image = {img_id: {} for img_id in ordered_ids}

    bleu_avg, bleu_per_image = Bleu(4).compute_score(gts, res)
    scores["BLEU-4"] = bleu_avg[3]
    for img_id, s in zip(ordered_ids, bleu_per_image[3]):
        per_image[img_id]["BLEU-4"] = s

    for metric_name, Scorer in [("METEOR", Meteor), ("ROUGE-L", Rouge), ("CIDEr", Cider)]:
        avg, per_img_scores = Scorer().compute_score(gts, res)
        scores[metric_name] = avg
        for img_id, s in zip(ordered_ids, per_img_scores):
            per_image[img_id][metric_name] = s

    return scores, per_image


def main():
    parser = argparse.ArgumentParser(description="Ablation evaluation on held-out COCO val2017")
    parser.add_argument("--num-eval-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coco-img-dir", default="./coco/val2017")
    parser.add_argument("--coco-ann-file", default="./coco/annotations/captions_val2017.json")
    parser.add_argument("--source-chroma-db", default="./chroma_db")
    parser.add_argument("--eval-chroma-db", default="./chroma_db_eval")
    parser.add_argument("--out", default="./eval_results.json")
    parser.add_argument(
        "--ablations",
        default=None,
        help="Comma-separated subset of ablation names to run (default: all four). "
        "E.g. --ablations retrieval-only to run just the baseline at a larger N.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Include per-image predictions + per-image scores in --out.",
    )
    args = parser.parse_args()

    ablations = ABLATIONS
    if args.ablations:
        wanted = set(args.ablations.split(","))
        ablations = [a for a in ABLATIONS if a[0] in wanted]

    by_image = load_annotations(args.coco_ann_file)
    img_paths = build_image_path_index(args.coco_img_dir)
    eval_ids = pick_eval_split(by_image, args.num_eval_images, args.seed)
    print(f"Eval split: {len(eval_ids)} held-out images (seed={args.seed})")

    config = prepare_eval_index(args.source_chroma_db, args.eval_chroma_db, by_image, eval_ids)
    config.caption_backend = "retrieval"

    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)

    print("Loading CLIP...")
    model_manager.load_clip_model()
    db_manager.initialize()

    all_results = {}
    for name, seg, obj in ablations:
        config.enable_segmentation = seg
        config.enable_object_detection = obj
        print(f"\n=== {name} (segmentation={seg}, object_detection={obj}) ===")

        predictions = {}
        timings = defaultdict(list)
        t_start = time.perf_counter()
        for i, img_id in enumerate(sorted(eval_ids)):
            img_path = img_paths.get(img_id)
            if not img_path:
                continue
            caption = instrumented_retrieve(
                img_path, config, model_manager, db_manager, preprocessor, timings
            )
            predictions[img_id] = caption
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(eval_ids)}")
        t_end = time.perf_counter()

        scores, per_image_scores = score_with_pycocoevalcap(predictions, by_image, eval_ids)
        latency = {
            stage: {
                "mean_ms": round(statistics.mean(vals) * 1000, 2),
                "p50_ms": round(statistics.median(vals) * 1000, 2),
                "calls": len(vals),
            }
            for stage, vals in timings.items()
            if vals
        }
        all_results[name] = {
            "scores": scores,
            "num_images": len(eval_ids),
            "latency_per_image_s": round((t_end - t_start) / len(eval_ids), 3),
            "stage_latency": latency,
            "total_wall_s": round(t_end - t_start, 1),
        }
        if args.save_predictions:
            all_results[name]["per_image"] = {
                str(img_id): {"prediction": predictions[img_id], "scores": per_image_scores[img_id]}
                for img_id in predictions
            }
        print(f"  scores: {scores}")
        print(f"  avg latency/image: {all_results[name]['latency_per_image_s']}s")

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")

    print("\n| Config | N | BLEU-4 | METEOR | ROUGE-L | CIDEr | s/image |")
    print("|---|---|---|---|---|---|---|")
    for name, _, _ in ablations:
        r = all_results[name]
        s = r["scores"]
        print(
            f"| {name} | {r['num_images']} | {s['BLEU-4']:.3f} | {s['METEOR']:.3f} | "
            f"{s['ROUGE-L']:.3f} | {s['CIDEr']:.3f} | {r['latency_per_image_s']:.2f} |"
        )


if __name__ == "__main__":
    main()
