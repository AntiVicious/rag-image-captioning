# RAG Image Captioning

Retrieval-Augmented Generation image captioning: CLIP embeddings + ChromaDB retrieval over COCO captions, with BLIP-2 generating the final caption from retrieved context.

Built incrementally, one module at a time, each step verified by CI. See commit history for the build-up.

## Status

Core retrieval pipeline works end-to-end, including advanced (segmentation/object-detection crop) retrieval — verified against a real, populated database.

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
- **`transformers` is pinned to `4.46.3`.** Two real, independent upstream bugs affect `DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")` outside this range:
  - `transformers` 5.x (what `>=4.35.0` auto-resolves to): loads successfully but then blows past 10GB+ RSS and gets OOM-killed — a real memory regression for this checkpoint, not just an under-provisioned VM (raising the WSL2 memory cap didn't help; peak usage scaled to match whatever ceiling was available).
  - `transformers` 4.44.0 and earlier: `safetensors_conversion.auto_conversion()` constructs `HfApi(headers=http_user_agent())`, passing a user-agent *string* where a headers *dict* is required, so `build_hf_headers()`'s `dict.update()` blows up (`ValueError: dictionary update sequence element #0 has length 1; 2 is required`). Fixed upstream in [huggingface/transformers#34010](https://github.com/huggingface/transformers/pull/34010), first shipped in `4.46.0`. `4.46.3` avoids both bugs.
