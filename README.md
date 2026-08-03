# RAG Image Captioning

Retrieval-Augmented Generation image captioning: CLIP embeddings + ChromaDB retrieval over COCO captions, with BLIP-2 generating the final caption from retrieved context.

Built incrementally, one module at a time, each step verified by CI. See commit history for the build-up.

## Status

Core retrieval pipeline works end-to-end (verified against a real, populated database). Advanced (segmentation/object-detection crop) retrieval is a known limitation on this stack right now — see Known issues.

## Running locally (Windows)

ChromaDB's native Rust embedding-insert path (`collection.add()`/`query()`) crashes with an access violation on at least one Windows dev machine, reproducing even against a bare in-memory `EphemeralClient` — this is a known, unresolved upstream issue ([chroma-core/chroma#3058](https://github.com/chroma-core/chroma/issues/3058)). **Use Docker** (below) to run `build-db` and the app; don't expect `python -m src.cli build-db` to work directly on Windows.

## Running with Docker

```bash
docker build -t rag-image-captioning:local .

# Build the database against COCO val2017 (~1GB, 5K images -- much cheaper
# than the ~18GB train2017 default). Mount your coco/ and chroma_db/ dirs so
# the built database persists on the host.
docker run --rm \
  -v "/path/to/coco:/app/coco" \
  -v "/path/to/chroma_db:/app/chroma_db" \
  rag-image-captioning:local \
  python -m src.cli --coco-img-dir ./coco/val2017 \
    --coco-ann-file ./coco/annotations/captions_val2017.json build-db

# Run the Streamlit app
docker run --rm -p 8501:8501 \
  -v "/path/to/chroma_db:/app/chroma_db" \
  rag-image-captioning:local
```

### WSL2 memory cap (Windows + Docker Desktop)

WSL2 defaults to a memory limit of 50% of host RAM if no `.wslconfig` exists, which can be too small for loading multiple CV models (CLIP + DETR) at once — this showed up as the container being silently OOM-killed (`exit 137`) partway through segmentation model loading, confirmed via the WSL2 kernel's own `dmesg` (`Out of memory: Killed process ... (python)`), not a chromadb or code issue. If you hit this, raise the limit in `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=11GB
```

then `wsl --shutdown` followed by restarting Docker Desktop (an app-only restart isn't enough — WSL2 needs a full shutdown to reallocate).

## Known issues

- **ChromaDB native crash on Windows** (see above) — always run `build-db` and the app via Docker on Windows.
- **Advanced retrieval (segmentation/object-detection crops) currently broken.** `DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")` fails in two different ways depending on the `transformers` version:
  - `transformers` 5.x (auto-resolved from `>=4.35.0`): loads successfully but then blows past 10GB+ RSS and gets OOM-killed — looks like a real memory regression for this checkpoint, not just an under-provisioned VM (raising the WSL2 memory cap didn't help; peak usage scaled to match whatever ceiling was available).
  - `transformers==4.44.0` (pinned to work around the above): no longer OOMs, but its safetensors-detection logic doesn't find this checkpoint's safetensors weights and falls back to an auto-conversion code path that crashes on a `huggingface_hub` API incompatibility (`ValueError: dictionary update sequence element #0 has length 1; 2 is required`) — reproduces identically even when `huggingface_hub` is pinned to a version contemporaneous with 4.44.0 (`0.24.6`).
  - Net effect: the "Use segmentation/object-detection crops" checkbox in the Streamlit app defaults to **off**. Basic retrieval (CLIP + ChromaDB, no crops) is fully working and is the recommended path until this is root-caused further.
