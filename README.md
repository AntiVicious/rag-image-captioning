# RAG Image Captioning

Retrieval-Augmented Generation image captioning: CLIP embeddings + ChromaDB retrieval over COCO captions. Optional DETR segmentation/object-detection crops decompose an image into sub-regions before retrieval; optional BLIP conditions a generated caption on the retrieved context. Built incrementally, one module at a time, each step verified by CI.

**Live demo:** deployed via Streamlit Community Cloud from this repo, main file `demo/demo_app.py` — link goes here once the Streamlit Cloud deploy finishes (github.com sign-in required to connect the repo, can't be automated). The demo index (COCO val2017 + Indian-context images) is hosted separately as a public HF Dataset ([AntiVicious/rag-image-captioning144-index](https://huggingface.co/datasets/AntiVicious/rag-image-captioning144-index)) and downloaded on first boot, since Community Cloud's free tier can't run the Docker-based full app. `demo/` has its own, deliberately unpinned `requirements.txt` and deliberately excludes `transformers` entirely (not just DETR/BLIP) — the root `requirements.txt`'s exact pins (chosen for real, documented bugs on the local dev machine — see below) turned out to transitively drag in build failures against whatever Python Community Cloud's free tier happens to default to, on three different packages before this was isolated. The demo is retrieval-only, no crops, no BLIP toggle; the real shipped default (medoid + segmentation crops, below) needs `transformers`/DETR and more RAM than the free host offers — run this repo locally with Docker for that.

## The finding

**How you consume a retrieved candidate pool matters as much as how you retrieve it — a claim that generalizes past this one system to RAG architectures broadly.** N=200, held-out split of COCO val2017, output length held constant across every cell (full table and reproduction commands in `results/README.md`):

| Selection → | top1 (nearest-neighbor) | medoid (consensus) |
|---|---|---|
| no crops | 0.469 CIDEr, 0.28s/image | 0.486 CIDEr, 0.47s/image |
| **+ segmentation crops** | 0.422 CIDEr, 10.43s/image | **0.537 CIDEr, 10.46s/image** |
| + object-detection crops | 0.406 CIDEr, 2.75s/image | 0.506 CIDEr, 3.46s/image |
| + both crop types | 0.395 CIDEr, 11.87s/image | 0.512 CIDEr, 12.84s/image |

**Shipped default: medoid selection + segmentation crops** — best score of every config measured. `Config.selection_mode="medoid"` (pick whichever retrieved candidate the *rest* of the pool agrees with, via CLIP text-embedding similarity, instead of whichever is closest to the query image — about 20 lines) and `Config.enable_segmentation=True`. `enable_object_detection` stays off — under medoid it's still worse than segmentation alone. Turn crops off (`--skip-detr` / `enable_segmentation=False`) if the ~10s/image DETR cost isn't worth an ~11% CIDEr gain for your use case; the faster path scores 0.486 CIDEr at 0.47s/image, still better than the original top1 baseline.

Read across either row and the picture flips: under top1, crops make things *worse* — more candidates competing for "closest to the query image" dilutes the signal, and it's specifically DETR's small, single-object-zoomed crops doing the damage (a control row using six large, blind, non-DETR crops at the same variant count scores 0.479, matching/beating plain retrieval-only — see `results/README.md`). Under medoid, the same crops are additive. Neither row alone is the finding; the interaction is.

### How this was found

This came from two separate measurements, run weeks apart, that each only tested one row of the table above and each concluded something different. First pass (top1 selection only) said "crops hurt, ship retrieval-only" — and that table also briefly reported CIDEr ≈ 7×10⁻¹⁰ for the crops-on config before a real bug (concatenating every crop's retrieved captions into one long block, confounding length-sensitive metrics with verbosity) got caught and fixed. A later, second pass testing medoid selection reversed the crop conclusion without changing the retrieval or the crops themselves — which is what prompted going back and building the full 2×4 table instead of trusting either measurement in isolation. Two smaller, cheaper checks were also run and are documented honestly in `results/README.md` including where they *didn't* hold up: a Recall@K measurement that turned out to be a self-match artifact and was retracted, and an index-quantization experiment whose "memory saved" framing turned out to be theoretical rather than achievable in ChromaDB's actual storage layer.

### Does retrieval hold up on out-of-distribution images?

COCO skews Western in objects, scenes, and phrasing. I sourced ~12,500 Indian-context images (a CC0 food dataset + a Wikimedia Commons scrape across 18 categories — festivals, temples, weddings, street life) and measured retrieval distance before and after adding them to the index, using a properly held-out, near-duplicate-corrected, food+monuments-mixed sample (N=200; ~46% of the Indian set turned out to have a near-duplicate elsewhere in it, mostly from the templated-caption food dataset, which was corrected for):

- **Before augmentation:** Indian-context images retrieve **56% worse** than COCO images retrieve against their own distribution (real distribution shift, as expected).
- **After augmentation:** that gap closes to **19.5% better** than COCO's own baseline.
- **Validated against real hand-written reference captions** (not just retrieval distance) on a 53-image sample: METEOR 0.047→0.081, ROUGE-L 0.154→0.182, CIDEr 0.113→0.288, all improving after augmentation. Grown from an initial N=20 spot-check; still short of a fully stable corpus-level sample size (see `results/README.md`).

Full methodology, the two self-inflicted leakage bugs caught and fixed while measuring this, and reproduction commands are in `scripts/measure_distribution_shift_v2.py` and `scripts/audit_near_duplicates.py`.

## Architecture

One image in. CLIP encodes it (and, if enabled, DETR-derived crops). ChromaDB returns the nearest captions on file per variant. `Config.selection_mode` picks ONE final caption out of that pool — `"medoid"` (the default: the candidate most similar, on average, to the rest of the pool, i.e. the consensus pick — the best-scoring strategy measured, see above) or `"top1"` (closest to the query image, simpler, scores a bit lower). That caption is either returned as-is (`caption_backend="retrieval"`, the default — no LLM, no GPU) or handed to a small BLIP captioner as conditioning context (`caption_backend="blip"` — BLIP base, ~247M params, CPU-feasible; not BLIP-2, which needs a GPU this project doesn't assume).

## Status

Core retrieval pipeline works end-to-end, ablation-evaluated, and the crop/BLIP-2 assumptions in the original design have both been tested and revised based on measurement rather than intuition.

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

## Reproducing the numbers

```bash
# Ablation table (needs the eval image -- see Dockerfile.eval; bundles a JRE
# for pycocoevalcap's METEOR/PTBTokenizer, kept out of the production image)
docker build -f Dockerfile.eval -t rag-image-captioning:eval .
docker run --rm \
  -v "/path/to/coco:/app/coco" \
  -v "/path/to/chroma_db:/app/chroma_db:ro" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 \
    --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --out /app/eval_output/eval_results.json

# Distribution-shift measurement (Indian-context images, before/after augmentation)
docker run --rm \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/chroma_db_before:/app/chroma_db_before" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/measure_distribution_shift_v2.py
```

Raw results backing the table above: `eval_output/eval_results_corrected_top1_v2.json`, `dist_shift_output/distribution_shift_v2.json`, `dist_shift_output/near_duplicate_audit.json`, `dist_shift_output/hand_labeled_scores.json`.

## Known issues

- **ChromaDB native crash on Windows** (see above) — always run `build-db` and the app via Docker on Windows.
- **`transformers` is pinned to `4.46.3`.** Two real, independent upstream bugs affect `DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")` outside this range:
  - `transformers` 5.x (what `>=4.35.0` auto-resolves to): loads successfully but then blows past 10GB+ RSS and gets OOM-killed — a real memory regression for this checkpoint, not just an under-provisioned VM (raising the WSL2 memory cap didn't help; peak usage scaled to match whatever ceiling was available).
  - `transformers` 4.44.0 and earlier: `safetensors_conversion.auto_conversion()` constructs `HfApi(headers=http_user_agent())`, passing a user-agent *string* where a headers *dict* is required, so `build_hf_headers()`'s `dict.update()` blows up (`ValueError: dictionary update sequence element #0 has length 1; 2 is required`). Fixed upstream in [huggingface/transformers#34010](https://github.com/huggingface/transformers/pull/34010), first shipped in `4.46.0`. `4.46.3` avoids both bugs.
- **The Indian-context distribution-shift and hand-labeled validation numbers are N=200 / N=53 respectively.** Real, held-out, and (for the distance metric) near-duplicate-corrected, but N=53 real references is still a modest sample for a corpus-level metric like CIDEr. Scaling further toward N=150 is the clearest remaining gap in this project's evidence.
- **Recall@K and index quantization were both re-examined and partially retracted after review** (see `results/README.md`) — an original "99% Recall@K" claim was a self-match artifact with no valid non-trivial replacement, and an original "50% memory saved, <1% recall lost" quantization claim conflated theoretical bytes-per-element arithmetic with an achievable result (ChromaDB has no native quantization) and understated the real ranking change by excluding a guaranteed self-match slot from the metric. Both sections were rewritten to report what was actually measured.
