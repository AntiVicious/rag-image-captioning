"""
Merge multiple COCO-shaped captions JSON files (from ingest_labeled_dataset.py
and scrape_commons_india.py runs) into one, for a single build-db pass.

Only merges files that actually exist -- a crashed scrape run that never
reached its own json.dump() simply has nothing to contribute here, and
is skipped with a warning rather than erroring out.

Usage:
    python scripts/merge_indian_captions.py \
        --in captions_food.json captions_commons.json captions_commons_pass3.json \
        --out captions_indian_merged.json
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge COCO-shaped captions JSON files")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    all_annotations = []
    seen_image_ids = set()
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"skip (missing, likely a crashed run): {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        anns = data["annotations"]
        collisions = [a for a in anns if a["image_id"] in seen_image_ids]
        if collisions:
            raise SystemExit(f"{path}: {len(collisions)} image_id collisions with an earlier input file")
        seen_image_ids.update(a["image_id"] for a in anns)
        all_annotations.extend(anns)
        print(f"{path}: +{len(anns)} annotations")

    with open(args.out, "w") as f:
        json.dump({"annotations": all_annotations}, f)
    print(f"\nWrote {len(all_annotations)} total annotations to {args.out}")


if __name__ == "__main__":
    main()
