"""
Before/after pycocoevalcap scoring against REAL hand-written reference
captions for a sample of Indian-context images -- not the retrieved-caption-
as-ground-truth proxy the rest of the distribution-shift analysis uses.

"Before": each image is CLIP-encoded fresh and queried (top1) against the
pre-augmentation COCO-only index.
"After": the SAME image's embedding is pulled directly out of the augmented
index (it's already indexed there) and queried (top1) against that index
with its own embedding masked out, so it can't self-match.

Usage (inside the eval image, for pycocoevalcap):
    docker run --rm \\
      -v <indian_datasets>/merged_images:/app/merged_images \\
      -v <clone>/chroma_db:/app/chroma_db_before \\
      -v D:\\chroma_db_augmented:/app/chroma_db_augmented \\
      -v <repo>/dist_shift_output:/app/dist_shift_output \\
      rag-image-captioning:eval python scripts/score_hand_labeled.py
"""

import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDIAN_ID_RE = re.compile(r"^(\d+)_\1$")


def score_with_pycocoevalcap(predictions, references):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    ids = sorted(predictions.keys())
    gts_raw = {i: [{"caption": references[i]}] for i in ids}
    res_raw = {i: [{"caption": predictions[i]}] for i in ids}
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    scores = {}
    bleu_avg, _ = Bleu(4).compute_score(gts, res)
    scores["BLEU-4"] = bleu_avg[3]
    for name, Scorer in [("METEOR", Meteor), ("ROUGE-L", Rouge), ("CIDEr", Cider)]:
        avg, _ = Scorer().compute_score(gts, res)
        scores[name] = avg
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-labels", default="/app/dist_shift_output/hand_labels.json")
    parser.add_argument("--images-dir", default="/app/merged_images")
    parser.add_argument("--before-db", default="/app/chroma_db_before")
    parser.add_argument("--after-db", default="/app/chroma_db_augmented")
    parser.add_argument("--out", default="/app/dist_shift_output/hand_labeled_scores.json")
    args = parser.parse_args()

    with open(args.hand_labels) as f:
        hand_labels = {int(k): v for k, v in json.load(f).items()}

    import chromadb
    import numpy as np
    import torch

    from src.config import Config
    from src.models import ModelManager

    config = Config()
    model_manager = ModelManager(config)
    model_manager.load_clip_model()

    # --- BEFORE: fresh CLIP encode, query the pre-augmentation COCO-only index ---
    before_client = chromadb.PersistentClient(path=args.before_db)
    before_coll = before_client.get_collection(config.collection_name)

    before_predictions = {}
    for img_id in hand_labels:
        img_path = os.path.join(args.images_dir, f"{img_id}.jpg")
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        tensor = model_manager.clip_preprocess(image).unsqueeze(0).to(model_manager.device)
        with torch.no_grad():
            embedding = model_manager.encode_image(tensor)
        result = before_coll.query(query_embeddings=[embedding.cpu().numpy()[0].tolist()], n_results=1)
        docs = (result.get("documents") or [[]])[0]
        before_predictions[img_id] = docs[0] if docs else ""

    # --- AFTER: pull the already-indexed embedding, query the augmented index
    #     (numpy, self-masked) ---
    after_client = chromadb.PersistentClient(path=args.after_db)
    after_coll = after_client.get_collection(config.collection_name)
    print(f"Fetching all {after_coll.count()} embeddings from augmented index...")
    all_result = after_coll.get(include=["embeddings", "documents"])
    all_ids = all_result["ids"]
    all_embeddings = np.array(all_result["embeddings"], dtype=np.float32)
    all_documents = np.array(all_result["documents"])
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / np.clip(norms, 1e-8, None)

    id_to_row = {}
    for i, gid in enumerate(all_ids):
        img_id = int(gid.split("_")[0])
        id_to_row.setdefault(img_id, i)

    after_predictions = {}
    for img_id in hand_labels:
        row = id_to_row.get(img_id)
        if row is None:
            print(f"WARNING: {img_id} not found in augmented index, skipping")
            continue
        query_gid = all_ids[row]
        mask = np.array([gid != query_gid for gid in all_ids])
        sims = all_embeddings[mask] @ all_embeddings[row]
        best_idx = int(np.argmax(sims))
        after_predictions[img_id] = str(all_documents[mask][best_idx])

    common_ids = sorted(set(before_predictions) & set(after_predictions) & set(hand_labels))
    references = {i: hand_labels[i] for i in common_ids}
    before_scores = score_with_pycocoevalcap({i: before_predictions[i] for i in common_ids}, references)
    after_scores = score_with_pycocoevalcap({i: after_predictions[i] for i in common_ids}, references)

    print(f"\n=== Real hand-labeled references, N={len(common_ids)} ===")
    print(f"BEFORE augmentation: {before_scores}")
    print(f"AFTER augmentation:  {after_scores}")

    print("\n| Metric | Before | After |")
    print("|---|---|---|")
    for m in ["BLEU-4", "METEOR", "ROUGE-L", "CIDEr"]:
        print(f"| {m} | {before_scores[m]:.4f} | {after_scores[m]:.4f} |")

    with open(args.out, "w") as f:
        json.dump(
            {
                "n": len(common_ids),
                "before_scores": before_scores,
                "after_scores": after_scores,
                "before_predictions": before_predictions,
                "after_predictions": after_predictions,
                "references": references,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
