# RAG Image Captioning

Retrieval-Augmented Generation image captioning: CLIP embeddings + ChromaDB retrieval over COCO captions. Optional DETR segmentation/object-detection crops decompose an image into sub-regions before retrieval; optional BLIP conditions a generated caption on the retrieved context. Built incrementally, one module at a time, each step verified by CI.

## The finding

The obvious design for this kind of system is "decompose the image with DETR, retrieve context per crop, aggregate everything, maybe generate a final caption with an LLM on top." I built that, then measured it against the simple version — CLIP on the whole image, no crops, no LLM — on a held-out split of COCO val2017 (N=200, output length held constant across configs so BLEU/METEOR/ROUGE-L/CIDEr aren't just measuring verbosity).

| Config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|
| generic-floor (no retrieval, fixed caption) | 0.021 | 0.067 | 0.322 | 0.009 | 0.543 | 0.00 |
| **retrieval-only** (CLIP on the whole image, no crops) | **0.098** | **0.160** | **0.352** | **0.469** | **0.672** | 0.28 |
| + segmentation crops | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| + object-detection crops | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| + both ("all-seven" variants/image) | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |

**Under this selection strategy, retrieval-only wins on every metric, at ~42x less latency than adding both crop types.** `enable_segmentation`/`enable_object_detection` default to `False` in `src/config.py` because of this; the crop pipeline is still there for anyone who wants to reproduce or challenge the result, but it isn't what the app runs by default.

The `generic-floor` row (a single fixed caption for every image, zero retrieval) is the actual score floor — it's what "no information" looks like on these metrics, and it's there so the other rows have something honest to be compared against, not just each other.

*(An earlier version of this table reported CIDEr ≈ 7×10⁻¹⁰ for the crops-on config — that was a real bug, not a real result: aggregating every crop's top-k retrieved captions into one long concatenated block confounded the length-sensitive metrics with verbosity rather than quality. Fixed by holding output length constant — one best candidate per config, selected by retrieval distance — before comparing.)*

**But "crops hurt" turned out to be incomplete, not wrong — it was specific to how the candidates get consumed.** Two follow-up experiments (`results/selection_strategy_ablation.csv`):

- **Selection strategy matters as much as the crops themselves.** The table above always picks the candidate *closest to the query image* (`--output-mode top1`). Swapping to **medoid selection** — pick the candidate most similar, on average, to the *rest* of the retrieved pool, a ~20-line consensus pick via CLIP text embeddings (`--output-mode medoid`) — flips the result: `retrieval-only` improves for free (0.469 → 0.486 CIDEr), and **`+segmentation` becomes the best config measured anywhere in this project (0.537 CIDEr)**, beating every top1 row including plain retrieval-only. Crops *do* help — top1 was just the wrong way to consume a larger candidate pool.
- **DETR's crop *characteristics*, not "having extra crops," are what hurt under top1.** A control row with 6 blind, large (40–80%-of-image) random crops — no DETR, same variant count as all-seven — scores 0.479 CIDEr under top1, matching/beating plain retrieval-only and clearly beating DETR's crops (0.395), at 1/14th the latency. DETR's crops are typically small and zoomed to a single object, pulling captions that drift from the whole-scene gestalt a reference caption actually describes; large blind crops don't have that problem.

Also measured, independently: **Recall@K on the retriever itself is 99% at K=1** (`results/recall_at_k.csv`) — CLIP+ChromaDB is reliably finding the right neighborhood; end-to-end caption quality is bounded by aggregation/selection choices and by paraphrase variance between human annotators, not by retrieval failing to find relevant images. And **quantizing the 591K-embedding scale-test index to float16 costs under 1% Recall@10 for 50% less memory** — a near-free win if this index needs to fit in a smaller footprint (`results/quantization_summary.csv`).

### Does retrieval hold up on out-of-distribution images?

COCO skews Western in objects, scenes, and phrasing. I sourced ~12,500 Indian-context images (a CC0 food dataset + a Wikimedia Commons scrape across 18 categories — festivals, temples, weddings, street life) and measured retrieval distance before and after adding them to the index, using a properly held-out, near-duplicate-corrected, food+monuments-mixed sample (N=200; ~46% of the Indian set turned out to have a near-duplicate elsewhere in it, mostly from the templated-caption food dataset, which was corrected for):

- **Before augmentation:** Indian-context images retrieve **56% worse** than COCO images retrieve against their own distribution (real distribution shift, as expected).
- **After augmentation:** that gap closes to **19.5% better** than COCO's own baseline.
- **Validated against real hand-written reference captions** (not just retrieval distance) on a 20-image sample: METEOR 0.044→0.086, ROUGE-L 0.158→0.191, CIDEr 0.129→0.310, all improving after augmentation.

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
- **The Indian-context distribution-shift and hand-labeled validation numbers are N=200 / N=20 respectively.** Real, held-out, and (for the distance metric) near-duplicate-corrected, but N=20 real references is a small sample — BLEU-4 is ~0 and uninformative at that size. Scaling the hand-labeled set is open follow-up work.
