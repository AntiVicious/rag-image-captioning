"""
Assembles the demo's precomputed gallery: merges the two evaluate.py runs
(--output-mode top1 and --output-mode medoid, same --num-eval-images/--seed
so they cover the identical image set) into one demo/gallery_data.json, and
copies+downsizes the source images into demo/gallery_images/ so the demo
never needs the full COCO dataset on disk.

Why this exists: the live-upload demo path can only run the retrieval-only
backend on Streamlit Community Cloud's free tier (no transformers/DETR, no
RAM for it -- see demo/demo_app.py's docstring), so a visitor uploading a
photo never sees the project's actual best-scoring config (medoid +
segmentation crops) or the crop ablation this project's whole headline
finding is about. Precomputing all 8 configs (2 selection strategies x 4
crop configs) offline for a fixed image set and shipping the JSON lets the
demo render the real, best-scoring pipeline's output with zero runtime
model-loading cost.

Usage:
    python scripts/build_gallery_data.py \
        --coco-img-dir /path/to/coco/val2017 \
        --coco-ann-file /path/to/coco/annotations/captions_val2017.json \
        --top1-json eval_output/gallery_top1.json \
        --medoid-json eval_output/gallery_medoid.json \
        --out-dir demo
"""

import argparse
import json
import os

from PIL import Image

# display name -> key evaluate.py's ABLATIONS actually uses in its output JSON
CROP_CONFIG_JSON_KEYS = {
    "retrieval-only": "retrieval-only",
    "segmentation": "+segmentation",
    "object-detection": "+object-detection",
    "all-seven": "all-seven",
}
CROP_CONFIGS = list(CROP_CONFIG_JSON_KEYS.keys())
MAX_DIM = 640


def load_annotations(ann_file):
    with open(ann_file) as f:
        data = json.load(f)
    by_image = {}
    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann["caption"])
    return by_image


def main():
    parser = argparse.ArgumentParser(description="Build the demo's precomputed gallery data + images")
    parser.add_argument("--coco-img-dir", required=True)
    parser.add_argument("--coco-ann-file", required=True)
    parser.add_argument("--top1-json", default="eval_output/gallery_top1.json")
    parser.add_argument("--medoid-json", default="eval_output/gallery_medoid.json")
    parser.add_argument("--out-dir", default="demo")
    args = parser.parse_args()

    with open(args.top1_json) as f:
        top1_data = json.load(f)
    with open(args.medoid_json) as f:
        medoid_data = json.load(f)

    by_image = load_annotations(args.coco_ann_file)

    image_ids = sorted(int(k) for k in top1_data[CROP_CONFIG_JSON_KEYS["retrieval-only"]]["per_image"].keys())
    medoid_ids = sorted(int(k) for k in medoid_data[CROP_CONFIG_JSON_KEYS["retrieval-only"]]["per_image"].keys())
    if image_ids != medoid_ids:
        raise SystemExit(
            f"top1 and medoid runs cover different image sets ({len(image_ids)} vs {len(medoid_ids)} images) "
            "-- re-run both with the same --num-eval-images/--seed."
        )
    print(f"{len(image_ids)} images, {len(CROP_CONFIGS)} crop configs x 2 selection strategies = "
          f"{len(CROP_CONFIGS) * 2} precomputed captions per image")

    images_out_dir = os.path.join(args.out_dir, "gallery_images")
    os.makedirs(images_out_dir, exist_ok=True)

    entries = []
    for image_id in image_ids:
        src_path = os.path.join(args.coco_img_dir, f"{image_id:012d}.jpg")
        if not os.path.exists(src_path):
            raise SystemExit(f"missing source image for {image_id}: {src_path}")

        img = Image.open(src_path).convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
        out_filename = f"{image_id}.jpg"
        img.save(os.path.join(images_out_dir, out_filename), "JPEG", quality=85)

        configs = {}
        for selection_label, data in [("top1", top1_data), ("medoid", medoid_data)]:
            for crop_key in CROP_CONFIGS:
                per_image = data[CROP_CONFIG_JSON_KEYS[crop_key]]["per_image"][str(image_id)]
                configs[f"{selection_label}__{crop_key}"] = {
                    "caption": per_image["prediction"],
                    "cider": per_image["scores"].get("CIDEr"),
                }

        entries.append(
            {
                "image_id": image_id,
                "file": out_filename,
                "reference_captions": by_image.get(image_id, [])[:3],
                "configs": configs,
            }
        )

    out_json = os.path.join(args.out_dir, "gallery_data.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "crop_configs": CROP_CONFIGS,
                "selection_strategies": ["top1", "medoid"],
                "images": entries,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_json} and {len(entries)} images to {images_out_dir}")


if __name__ == "__main__":
    main()
