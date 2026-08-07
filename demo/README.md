# demo/

Streamlit Community Cloud entry point for the [live demo](https://rag-image-captioning-fxxzqkhwjuqyx5s6rthwxj.streamlit.app/) — deliberately separate from the root `app.py`/`requirements.txt`, in its own directory with its own `demo/requirements.txt`. Main file: `demo_app.py`.

## Two tabs, two different honesty tradeoffs

- **📸 Precomputed gallery (opens by default).** 40 held-out COCO val2017 images, run offline through all 8 configs this project's ablation covers (2 selection strategies x 4 crop configs) via `scripts/evaluate.py --save-predictions`, merged into `demo/gallery_data.json` and downsized into `demo/gallery_images/` by `scripts/build_gallery_data.py`. Flipping the selection/crop toggles here shows the real output of the real best-scoring config (medoid + segmentation crops) — no model loaded, no upload needed, zero runtime cost.
- **📤 Upload your own.** Runs live retrieval against a real index, but honestly scoped to what Community Cloud's free tier (~1GB RAM) can actually run: retrieval-only, no DETR crops, no BLIP. It cannot run the project's actual shipped default — that needs `transformers`/DETR and more RAM than this host offers. The gallery tab is what a visitor should look at to see that config's real output; this tab is for trying an arbitrary photo against the lighter-weight path.

## Why this is a separate directory, not just a separate file

Streamlit Community Cloud looks for a `requirements.txt` next to the app's main file first. The root `requirements.txt` pins `transformers==4.46.3` for a real, Linux-reproduced DETR-loading bug (see the main README's "Known issues") — but that pin transitively ceilings `tokenizers<0.21`, and building an old `tokenizers` from source needs a Rust toolchain new enough for whatever Python Community Cloud currently defaults to, which isn't guaranteed. Three separate packages (`torch`/`torchvision`, then `sentencepiece`, then `tokenizers`-via-`transformers`) failed to build against Community Cloud's Python this way before the fix became "don't install `transformers` in the demo at all," not "unpin the next thing that breaks."

This is safe, not just convenient: every `transformers` import in `src/` (DETR loaders in `image_preprocessing.py`, the BLIP loader in `models.py`) is inside a lazy-loading function body, never at module level — so `demo_app.py`, which never calls those paths, genuinely doesn't need `transformers` installed at all. `demo/requirements.txt` has no exact pins, on purpose: this demo runs on whatever Python Community Cloud happens to default to on a given day, which this project doesn't control.

## What's different from the "real" shipped app

Driven by Community Cloud's free-tier constraints (~1GB RAM; no `transformers` installed here at all):

1. The ChromaDB index for the upload tab (COCO val2017 + ~12,500 Indian-context images) isn't in the repo (`chroma_db/` is gitignored) — downloaded from a public HF Dataset ([`AntiVicious/rag-image-captioning144-index`](https://huggingface.co/datasets/AntiVicious/rag-image-captioning144-index)) on first boot, if not already present locally.
2. The upload tab is retrieval-only, no crops, no BLIP toggle. The actual shipped default (medoid selection + DETR segmentation crops, see `src/config.py` and the main README) needs `transformers`/DETR and well over 1GB RAM in practice — neither fits a ~1GB, `transformers`-free free host. Run this project locally via Docker (see the main README) for the real default with live upload, or use the gallery tab to see its real, precomputed output.

## Running this locally without Docker

GitHub Codespaces (badge in the main README) uses `.devcontainer/devcontainer.json` to install `demo/requirements.txt` and launch `streamlit run demo/demo_app.py` automatically — same lightweight path as the deployed demo, no local setup. Locally: `pip install -r demo/requirements.txt && streamlit run demo/demo_app.py` from the repo root.

## Regenerating the gallery

```bash
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db:ro" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 40 --seed 42 --coco-img-dir /app/coco/val2017 \
    --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval_gallery_top1 \
    --out /app/eval_output/gallery_top1.json --output-mode top1 --skip-baselines --save-predictions

# repeat with --output-mode medoid --eval-chroma-db /app/chroma_db_eval_gallery_medoid --out /app/eval_output/gallery_medoid.json

python scripts/build_gallery_data.py \
  --coco-img-dir /path/to/coco/val2017 --coco-ann-file /path/to/coco/annotations/captions_val2017.json \
  --top1-json eval_output/gallery_top1.json --medoid-json eval_output/gallery_medoid.json --out-dir demo
```
