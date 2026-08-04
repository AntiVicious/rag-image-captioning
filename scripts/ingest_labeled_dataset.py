"""
Ingest a HuggingFace image-classification dataset (parquet shards, an
`image` column + a `label`/`target` column) into a COCO-shaped image
directory + captions JSON, so it can be indexed through the existing
`python -m src.cli build-db` path unchanged.

Why template captions: no ready-made, naturally-captioned, COCO-scale
Indian-context dataset currently exists (checked) -- what's available is
either (a) Indian-LANGUAGE translations of COCO/Visual Genome's own
Western-context images (solves language, not visual distribution shift),
or (b) genuinely Indian-context images labeled only for classification.
This script takes the (b) path and generates one template caption per
class label -- a real limitation (no per-image variation), disclosed
plainly rather than dressed up as natural captions.

Usage:
    python scripts/ingest_labeled_dataset.py \
        --repo-id bharat-raghunathan/indian-foods-dataset \
        --files <comma-separated parquet paths, see the repo's data/ listing> \
        --label-names <comma-separated class names, index-ordered> \
        --caption-template captions_food.json \
        --out-img-dir ./coco_indian/food \
        --out-ann-file ./coco_indian/captions_food.json \
        --id-offset 90000000
"""

import argparse
import io
import json
import os

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Ingest a labeled HF image dataset into COCO shape")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--files", required=True, help="Comma-separated parquet paths within the repo")
    parser.add_argument("--label-names", required=True, help="Comma-separated class names, index-ordered")
    parser.add_argument(
        "--caption-template",
        required=True,
        help="Path to a JSON file mapping label name -> caption sentence",
    )
    parser.add_argument("--out-img-dir", required=True)
    parser.add_argument("--out-ann-file", required=True)
    parser.add_argument("--id-offset", type=int, default=90000000, help="Avoid colliding with COCO image_ids")
    parser.add_argument("--label-column", default=None, help="Override: 'label' or 'target' (auto-detected)")
    args = parser.parse_args()

    label_names = args.label_names.split(",")
    with open(args.caption_template) as f:
        caption_template = json.load(f)
    missing = [name for name in label_names if name not in caption_template]
    if missing:
        raise SystemExit(f"caption template missing entries for: {missing}")

    os.makedirs(args.out_img_dir, exist_ok=True)

    annotations = []
    next_id = args.id_offset
    for rel_path in args.files.split(","):
        rel_path = rel_path.strip()
        print(f"Downloading {rel_path} ...")
        local_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=rel_path)
        df = pd.read_parquet(local_path)

        label_col = args.label_column or ("label" if "label" in df.columns else "target")
        image_col = "image"

        print(f"  {len(df)} rows, label column '{label_col}'")
        for _, row in df.iterrows():
            image_field = row[image_col]
            image_bytes = image_field["bytes"] if isinstance(image_field, dict) else image_field.get("bytes")
            label_idx = int(row[label_col])
            label_name = label_names[label_idx]

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_id = next_id
            next_id += 1
            out_path = os.path.join(args.out_img_dir, f"{img_id}.jpg")
            image.save(out_path, "JPEG", quality=90)

            annotations.append({"image_id": img_id, "id": img_id, "caption": caption_template[label_name]})

        if (next_id - args.id_offset) % 500 == 0 or rel_path == args.files.split(",")[-1].strip():
            print(f"  ingested so far: {next_id - args.id_offset}")

    with open(args.out_ann_file, "w") as f:
        json.dump({"annotations": annotations}, f)

    print(f"\nWrote {len(annotations)} images to {args.out_img_dir}")
    print(f"Wrote {args.out_ann_file}")
    print(f"image_id range: {args.id_offset} - {next_id - 1}")


if __name__ == "__main__":
    main()
