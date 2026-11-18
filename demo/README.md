# demo/

Streamlit Community Cloud app for the
[live demo](https://rag-image-captioning-fxxzqkhwjuqyx5s6rthwxj.streamlit.app/).
Main file: `demo_app.py`, with its own `requirements.txt`, separate from the
root app.

## Tabs

- **Precomputed gallery (default).** 40 held-out COCO val2017 images, run
  offline through all 8 ablation configurations (2 selection strategies × 4
  crop configurations) with `scripts/evaluate.py --save-predictions`, then
  merged into `gallery_data.json` and resized into `gallery_images/` by
  `scripts/build_gallery_data.py`. The toggles show the real output of each
  configuration, including the default (medoid + segmentation crops), without
  loading any model.
- **Upload your own.** Live retrieval against the full index, limited to
  what the free tier (about 1 GB RAM) can run: retrieval only, without DETR
  crops or BLIP. For the default configuration on your own images, run the
  project locally with Docker.

## Why a Separate Directory

Streamlit Community Cloud installs the `requirements.txt` next to the main
file. The root `requirements.txt` pins `transformers==4.46.3`, which
requires an old `tokenizers` release with no prebuilt wheel for the Python
version Community Cloud uses; building it from source failed. The demo
therefore doesn't install `transformers` at all.

This works because every `transformers` import in `src/`
(`image_preprocessing.py` for DETR, `models.py` for BLIP) is inside a
function, and the demo never calls those functions. `demo/requirements.txt`
is unpinned, since the host's Python version can change.

## Data

The ChromaDB index (COCO val2017 plus about 12,500 Indian-context images) is
not in the repository. The app downloads it on first start from the
Hugging Face dataset
[`AntiVicious/rag-image-captioning144-index`](https://huggingface.co/datasets/AntiVicious/rag-image-captioning144-index).

## Running Locally

The Codespaces badge in the main README opens a container that installs
`demo/requirements.txt` and starts the demo (see
`.devcontainer/devcontainer.json`). Locally, from the repository root:

```bash
pip install -r demo/requirements.txt && streamlit run demo/demo_app.py
```

## Regenerating the Gallery

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
