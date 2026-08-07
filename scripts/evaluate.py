"""
Ablation evaluation: BLEU-4, METEOR, ROUGE-L, CIDEr, CLIPScore on a held-out
split of COCO val2017, across four retrieval configurations:

    retrieval-only      -- CLIP query on the original image only
    +segmentation       -- + top segmentation crops (DETR panoptic)
    +object-detection   -- + top object-detection crops (DETR)
    all-seven           -- both crop types together

plus two baseline rows that are NOT part of the crop ablation:

    generic-floor       -- a single fixed caption for every image, no
                            retrieval at all -- the score floor: what "zero
                            information" looks like on these metrics.
    top1-verbatim       -- the single nearest-neighbour caption for the
                            original (uncropped) image, returned as-is, no
                            aggregation. The obvious baseline the crop
                            ablations should be beating.

Output length is held constant across every row: --output-mode top1
(the default) makes every config emit exactly one caption -- the single
retrieved candidate with the lowest ChromaDB distance across all variants
queried -- rather than concatenating every variant's top-k candidates via
aggregate_captions(). The original --output-mode aggregated is kept for
reference/reproducibility of the earlier (length-confounded) numbers, but
is no longer the default: n-gram metrics like CIDEr are extremely sensitive
to candidate length, so comparing configs that emit different amounts of
text was comparing length, not retrieval quality. See CLAUDE.md / the
walkthrough doc for the corrected numbers and why the old CIDEr=~0 result
for all-seven was a length artifact, not (only) evidence crops hurt.

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
    python scripts/evaluate.py --output-mode aggregated  # old, length-confounded behavior
    python scripts/evaluate.py --debug-print-config all-seven --debug-print-n 10
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
from src.caption_selection import select_medoid, select_top1  # noqa: E402
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

# The score floor: a single fixed caption emitted for every image, regardless
# of content. Any config that doesn't clearly beat this isn't adding value.
GENERIC_FLOOR_CAPTION = "a photo of a scene"

# CLIPScore (Hessel et al. 2021, https://arxiv.org/abs/2104.08718): reference-
# free cosine similarity between a caption's CLIP text embedding and the
# query image's CLIP embedding, rescaled. Computed here with the same
# open_clip ViT-B-32/openai backbone already loaded for retrieval (the
# paper's own reference implementation also defaults to ViT-B/32), so it's
# measuring similarity in the exact embedding space this system retrieves
# in. Unlike BLEU/METEOR/ROUGE-L/CIDEr it does not collapse as candidate
# length changes, which is what makes it the right metric for comparing
# configs whose outputs differ in length.
CLIPSCORE_WEIGHT = 2.5


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


def clip_score(image_embedding, caption, model_manager):
    """CLIPScore for one (image embedding, caption) pair: 2.5 * max(cos, 0).
    image_embedding must already be a normalised CLIP embedding (1, D)."""
    import torch.nn.functional as F

    text_embedding = model_manager.encode_text([caption])
    cos = F.cosine_similarity(image_embedding, text_embedding).item()
    return CLIPSCORE_WEIGHT * max(cos, 0.0)


def get_random_crops(image, n, seed_key):
    """Control for the DETR-crop ablation: n "dumb" crops at random scale/
    position, no object/segment awareness at all. Deterministic per image
    (seeded on seed_key, e.g. the image_id) so repeat runs are reproducible.
    If DETR-aware crops and blind random crops score the same, the finding
    upgrades from "DETR crops specifically don't help" to "crop-based
    grounding doesn't help regardless of how the crops are chosen" -- a
    stronger, cheaper-to-support claim."""
    rng = random.Random(seed_key)
    w, h = image.size
    crops = []
    for _ in range(n):
        # each crop covers 40-80% of each dimension, placed at a random
        # valid offset -- "dumb" on purpose: no notion of what's IN the crop
        crop_w = max(1, int(w * rng.uniform(0.4, 0.8)))
        crop_h = max(1, int(h * rng.uniform(0.4, 0.8)))
        x0 = rng.randint(0, max(0, w - crop_w))
        y0 = rng.randint(0, max(0, h - crop_h))
        crops.append(image.crop((x0, y0, x0 + crop_w, y0 + crop_h)))
    return crops


def instrumented_retrieve(
    image_path, config, model_manager, db_manager, preprocessor, timings, output_mode="top1", random_crop_n=0
):
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

    rand_crops = get_random_crops(image, random_crop_n, seed_key=image_path) if random_crop_n else []
    variants = [image] + list(seg_crops) + list(obj_crops) + rand_crops

    import torch

    t_a = time.perf_counter()
    tensors = [preprocessor.preprocess_for_clip(v, model_manager.clip_preprocess) for v in variants]
    batch = torch.cat(tensors, dim=0).to(model_manager.device)
    embeddings = model_manager.encode_image(batch)
    t_b = time.perf_counter()
    query_embeddings = embeddings.cpu().numpy().tolist()
    # top1 mode only ever looks at each variant's closest candidate (index 0),
    # so request n_results=1 directly rather than captions_per_crop and
    # discarding the rest. This also sidesteps a real tie-breaking quirk: a
    # COCO image's ~5 captions all share one embedding, so when the nearest
    # neighbour is a COCO image there are multiple candidates tied at the
    # EXACT same distance, and which one chromadb/hnswlib returns at index 0
    # was observed to depend on the requested n_results (see
    # scripts/diagnose_topk_nondeterminism.py -- 145/200 images differed
    # between n_results=1 and n_results=3 queries, always at distance delta
    # 0.000000, i.e. a tie-break artifact, not a different nearest image).
    top_k = 1 if output_mode == "top1" else config.captions_per_crop
    # medoid needs a real candidate pool to choose a consensus from, same as
    # aggregated mode's pool size -- only top1 can shrink the query to k=1.
    results = db_manager.query_similar(query_embeddings, top_k)
    t_c = time.perf_counter()

    documents_per_variant = results.get("documents") or [[] for _ in variants]
    distances_per_variant = results.get("distances") or [[] for _ in variants]

    n = len(variants)
    timings["clip_encode_batch_total"].append(t_b - t_a)
    timings["clip_encode_per_variant"].append((t_b - t_a) / n)
    timings["chroma_query_batch_total"].append(t_c - t_b)
    timings["chroma_query_per_variant"].append((t_c - t_b) / n)
    timings["variants_per_image"].append(n)

    t_agg0 = time.perf_counter()
    if output_mode == "top1":
        # Global best candidate across every variant queried, by distance.
        caption = select_top1(documents_per_variant, distances_per_variant)
    elif output_mode == "medoid":
        caption = select_medoid(documents_per_variant, model_manager)
    else:
        blocks = [" ".join(docs) for docs in documents_per_variant]
        caption = aggregate_captions(blocks)
    t_agg1 = time.perf_counter()
    timings["aggregate"].append(t_agg1 - t_agg0)

    # CLIPScore against the ORIGINAL (uncropped) image embedding -- variants[0]
    # -- regardless of how many crop variants contributed to retrieval, so the
    # metric always answers the same question: "does the produced caption
    # describe THIS image."
    score = clip_score(embeddings[0:1], caption, model_manager) if caption else 0.0
    timings["clipscore"].append(score)

    return caption


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


def run_generic_floor(eval_ids, img_paths, model_manager, preprocessor):
    """Score floor baseline: the SAME fixed caption for every image, no
    retrieval at all. Tells us what "zero information" scores as -- any
    ablation config that doesn't clearly clear this isn't adding value."""
    predictions = {}
    clip_scores = []
    for img_id in sorted(eval_ids):
        img_path = img_paths.get(img_id)
        if not img_path:
            continue
        predictions[img_id] = GENERIC_FLOOR_CAPTION
        tensor = preprocessor.preprocess_for_clip(img_path, model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor)
        clip_scores.append(clip_score(embedding, GENERIC_FLOOR_CAPTION, model_manager))
    return predictions, clip_scores


def run_top1_verbatim(eval_ids, img_paths, model_manager, db_manager, preprocessor):
    """The obvious baseline the crop ablations need to beat: single nearest-
    neighbour caption for the ORIGINAL uncropped image, returned verbatim --
    no crops, no aggregation, n_results=1. (Should numerically match the
    retrieval-only ablation row run with --output-mode top1: that row already
    reduces to exactly this once output length is controlled. Computed here
    independently as a sanity check on that -- if the two diverge, something
    in the ablation loop's top1 selection is broken.)"""
    predictions = {}
    clip_scores = []
    t_start = time.perf_counter()
    for img_id in sorted(eval_ids):
        img_path = img_paths.get(img_id)
        if not img_path:
            continue
        tensor = preprocessor.preprocess_for_clip(img_path, model_manager.clip_preprocess)
        tensor = tensor.to(model_manager.device)
        embedding = model_manager.encode_image(tensor)
        results = db_manager.query_similar(embedding.cpu().numpy().tolist(), 1)
        docs = (results.get("documents") or [[]])[0]
        caption = docs[0] if docs else ""
        predictions[img_id] = caption
        clip_scores.append(clip_score(embedding, caption, model_manager) if caption else 0.0)
    t_end = time.perf_counter()
    return predictions, clip_scores, (t_end - t_start) / max(len(predictions), 1)


def debug_print_pairs(name, predictions, by_image, eval_ids, n):
    """Print raw candidate/reference pairs BEFORE pycocoevalcap tokenizes
    them, so a suspiciously extreme score (e.g. CIDEr ~1e-9) can be eyeballed
    against the actual text instead of trusted at face value."""
    print(f"\n--- raw candidate/reference pairs: {name} (first {n}) ---")
    for img_id in sorted(eval_ids)[:n]:
        if img_id not in predictions:
            continue
        refs = [c for _, c in by_image[img_id]]
        print(f"  image_id={img_id}")
        print(f"    candidate: {predictions[img_id]!r}")
        print(f"    references: {refs!r}")


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
        "--output-mode",
        choices=["top1", "medoid", "aggregated"],
        default="top1",
        help="top1 (default): every config emits exactly one caption, the globally best "
        "candidate by distance -- output length held constant across configs, so n-gram "
        "metrics measure retrieval quality, not verbosity. medoid: also emits exactly one "
        "caption, but picks the candidate most similar (by CLIP text embedding) to the rest "
        "of the retrieved pool, rather than closest to the query image -- a consensus pick "
        "instead of a nearest-neighbour pick. aggregated: the old behavior, concatenating "
        "every variant's top-k candidates via aggregate_captions() -- kept for reproducing "
        "the original (length-confounded) numbers.",
    )
    parser.add_argument(
        "--ablations",
        default=None,
        help="Comma-separated subset of ablation names to run (default: all four). "
        "E.g. --ablations retrieval-only to run just the baseline at a larger N.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the generic-floor and top1-verbatim baseline rows.",
    )
    parser.add_argument(
        "--random-crop-control",
        action="store_true",
        help="Add a '+random-crops' row: 6 blind random crops (no DETR) per image, same "
        "variant count as all-seven. Tests whether DETR's crop SELECTION matters, or "
        "whether crop-based grounding hurts regardless of how crops are chosen.",
    )
    parser.add_argument(
        "--debug-print-config",
        default="all-seven",
        help="Print raw candidate/reference pairs for this config before scoring (set to "
        "'' to disable).",
    )
    parser.add_argument("--debug-print-n", type=int, default=10)
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
    print(f"Output mode: {args.output_mode}")

    config = prepare_eval_index(args.source_chroma_db, args.eval_chroma_db, by_image, eval_ids)
    config.caption_backend = "retrieval"

    model_manager = ModelManager(config)
    db_manager = DatabaseManager(config)
    preprocessor = ImagePreprocessor(config)

    print("Loading CLIP...")
    model_manager.load_clip_model()
    db_manager.initialize()

    all_results = {}
    row_order = []

    if not args.skip_baselines:
        print("\n=== generic-floor (no retrieval, fixed caption) ===")
        gf_predictions, gf_clip_scores = run_generic_floor(eval_ids, img_paths, model_manager, preprocessor)
        gf_scores, gf_per_image = score_with_pycocoevalcap(gf_predictions, by_image, eval_ids)
        gf_scores["CLIPScore"] = statistics.mean(gf_clip_scores)
        all_results["generic-floor"] = {
            "scores": gf_scores,
            "num_images": len(gf_predictions),
            "latency_per_image_s": 0.0,
        }
        row_order.append("generic-floor")
        print(f"  scores: {gf_scores}")

        print("\n=== top1-verbatim (uncropped image, k=1, no aggregation) ===")
        t1_predictions, t1_clip_scores, t1_latency = run_top1_verbatim(
            eval_ids, img_paths, model_manager, db_manager, preprocessor
        )
        t1_scores, t1_per_image = score_with_pycocoevalcap(t1_predictions, by_image, eval_ids)
        t1_scores["CLIPScore"] = statistics.mean(t1_clip_scores)
        all_results["top1-verbatim"] = {
            "scores": t1_scores,
            "num_images": len(t1_predictions),
            "latency_per_image_s": round(t1_latency, 3),
        }
        row_order.append("top1-verbatim")
        print(f"  scores: {t1_scores}")

    rows = [(name, seg, obj, 0) for name, seg, obj in ablations]
    if args.random_crop_control:
        # Same variant COUNT as all-seven (up to 6 non-original crops, i.e.
        # 7 total) but chosen blind -- no DETR, no notion of what's in the
        # crop. Tests whether it's specifically DETR's crop *selection* that
        # hurts, or crop-based grounding hurting regardless of how crops are
        # chosen (a stronger, cheaper-to-support claim if this also loses).
        rows.append(("+random-crops", False, False, 6))

    for name, seg, obj, random_crop_n in rows:
        config.enable_segmentation = seg
        config.enable_object_detection = obj
        print(f"\n=== {name} (segmentation={seg}, object_detection={obj}, random_crops={random_crop_n}) ===")

        predictions = {}
        timings = defaultdict(list)
        t_start = time.perf_counter()
        for i, img_id in enumerate(sorted(eval_ids)):
            img_path = img_paths.get(img_id)
            if not img_path:
                continue
            caption = instrumented_retrieve(
                img_path,
                config,
                model_manager,
                db_manager,
                preprocessor,
                timings,
                args.output_mode,
                random_crop_n,
            )
            predictions[img_id] = caption
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(eval_ids)}")
        t_end = time.perf_counter()

        if name == args.debug_print_config and args.debug_print_config:
            debug_print_pairs(name, predictions, by_image, eval_ids, args.debug_print_n)

        scores, per_image_scores = score_with_pycocoevalcap(predictions, by_image, eval_ids)
        clip_scores = timings.pop("clipscore", [])
        scores["CLIPScore"] = statistics.mean(clip_scores) if clip_scores else 0.0
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
        row_order.append(name)
        if args.save_predictions:
            all_results[name]["per_image"] = {
                str(img_id): {"prediction": predictions[img_id], "scores": per_image_scores[img_id]}
                for img_id in predictions
            }
        print(f"  scores: {scores}")
        print(f"  avg latency/image: {all_results[name]['latency_per_image_s']}s")

    if not args.skip_baselines and "retrieval-only" in row_order:
        ro = all_results["retrieval-only"]["scores"]
        t1 = all_results["top1-verbatim"]["scores"]
        if abs(ro["CIDEr"] - t1["CIDEr"]) > 1e-6 and args.output_mode == "top1":
            print(
                "\nWARNING: retrieval-only (top1 mode) and top1-verbatim CIDEr disagree "
                f"({ro['CIDEr']:.6f} vs {t1['CIDEr']:.6f}) -- these should be identical. "
                "Check the top1 selection logic in instrumented_retrieve()."
            )

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")

    print("\n| Config | N | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |")
    print("|---|---|---|---|---|---|---|---|")
    for name in row_order:
        r = all_results[name]
        s = r["scores"]
        print(
            f"| {name} | {r['num_images']} | {s['BLEU-4']:.3f} | {s['METEOR']:.3f} | "
            f"{s['ROUGE-L']:.3f} | {s['CIDEr']:.5f} | {s['CLIPScore']:.3f} | {r['latency_per_image_s']:.2f} |"
        )


if __name__ == "__main__":
    main()
