# RAG Image Captioning

[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://rag-image-captioning-fxxzqkhwjuqyx5s6rthwxj.streamlit.app/) [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/AntiVicious/rag-image-captioning)

Retrieval-Augmented Generation image captioning: CLIP embeddings + ChromaDB retrieval over COCO captions. Optional DETR segmentation/object-detection crops decompose an image into sub-regions before retrieval; optional BLIP conditions a generated caption on the retrieved context. Built incrementally, one module at a time, each step verified by CI.

**[Live demo →](https://rag-image-captioning-fxxzqkhwjuqyx5s6rthwxj.streamlit.app/)** Opens on a precomputed gallery (40 held-out images x all 8 ablation configs, zero runtime cost, including the real best-scoring config), with a live "Upload your own" tab alongside it. Deployment details, hosting constraints, and why the two tabs differ are in [`demo/README.md`](demo/README.md).

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

This came from two separate measurements, run weeks apart, that each only tested one row of the table above and each concluded something different. First pass (top1 selection only) said "crops hurt, ship retrieval-only" — and that table also briefly reported CIDEr ≈ 7×10⁻¹⁰ for the crops-on config before a real bug (concatenating every crop's retrieved captions into one long block, confounding length-sensitive metrics with verbosity) got caught and fixed. A later, second pass testing medoid selection reversed the crop conclusion without changing the retrieval or the crops themselves — which is what prompted going back and building the full 2×4 table instead of trusting either measurement in isolation. Two smaller, cheaper checks were also run and had their own corrections — a Recall@K measurement that turned out to be a self-match artifact and was retracted, and an index-quantization experiment that took three rounds to get right — both broken down as transferable lessons in "Measurement traps in retrieval evaluation" below.

### Does retrieval hold up on out-of-distribution images?

COCO skews Western in objects, scenes, and phrasing. I sourced ~12,500 Indian-context images (a CC0 food dataset + a Wikimedia Commons scrape across 18 categories — festivals, temples, weddings, street life) and measured retrieval distance before and after adding them to the index, using a properly held-out, near-duplicate-corrected, food+monuments-mixed sample (N=200; ~46% of the Indian set turned out to have a near-duplicate elsewhere in it, mostly from the templated-caption food dataset, which was corrected for):

- **Before augmentation:** Indian-context images retrieve **56% worse** than COCO images retrieve against their own distribution (real distribution shift, as expected).
- **After augmentation:** that gap closes to **19.5% better** than COCO's own baseline. Read this as evidence augmentation worked, not as a claim Indian-context retrieval is now inherently better than COCO's: even dedup-corrected, the Indian set spans a narrower target (18 Commons categories + 15 food dishes vs. COCO's 80 object categories), so higher within-category density — and a shorter average nearest-neighbor distance — is partly expected from scope alone.
- **Validated against real hand-written reference captions** (not just retrieval distance) on a 53-image sample: BLEU-4 0.012→0.018, METEOR 0.047→0.081, ROUGE-L 0.154→0.182, CIDEr 0.113→0.288, all improving after augmentation, with BLEU-4 showing real (if small) signal for the first time at this sample size. Grown from an initial N=20 spot-check; still short of a fully stable corpus-level sample size (see `results/README.md`). This CIDEr is against a different, Indian-context-only reference set — not comparable to the 0.395–0.537 CIDEr range in the ablation table above, which scores COCO val2017 predictions against COCO's own references.

Full methodology, the two self-inflicted leakage bugs caught and fixed while measuring this, and reproduction commands are in `scripts/measure_distribution_shift_v2.py` and `scripts/audit_near_duplicates.py`.

## Measurement traps in retrieval evaluation

Six numbers in this project were wrong at some point and got caught before shipping — CIDEr scoring, two separate instances of self-match leakage, a sampling-scope bug, a Recall@K claim with no valid null case, and a quantization result wrong on two independent counts. Listed chronologically that reads as a mistake log; listed by what each one generalizes to, it's a checklist for evaluating any retrieval system, not just this one. Each entry: the trap, how to catch it before it ships, and where it showed up here.

1. **A metric can be confounded by an unmeasured variable instead of the one you're testing.** Two configs score differently — it's tempting to credit the variable you changed, but an incidental side effect of *how the comparison was implemented* can dominate instead. Catch it by holding every variable but one constant, and treating an implausibly large score swing as a bug signal, not a strong effect (CIDEr ≈ 7×10⁻¹⁰ is not "crops are terrible," it's a scorer choking on something). Here: concatenating every crop's retrieved captions into one long block confounded CIDEr with output length/verbosity, not caption quality — fixed by holding output length constant (one best candidate per config) before comparing selection strategies.

2. **Self-match leakage: excluding a query from itself isn't the same as excluding everything that behaves identically to it.** Recurred three separate times here. Catch it by asking, before trusting any retrieval-quality number: could the ground-truth item and a candidate share an identity, a stored embedding, or a trivially-derived relationship that isn't the thing being measured? Here: (a) an original Recall@K of 99% was the query image finding its own embedding, still present, in the index; (b) a fp16/int8 quantization "ranking stability" number was inflated because only the sampled row — not its ~4 sibling caption rows, all sharing one CLIP image embedding — was excluded from its own candidate search; (c) an Indian-context image with a near-duplicate twin elsewhere in the index retrieved its twin, inflating the "after augmentation, vs. COCO baseline" distance comparison to an apparent −62% before excluding near-duplicate twins (not just exact self-matches) corrected it to the real −19.5%.

3. **A sampling scope can be narrower than the claim built on top of it.** A script argument silently points at a subset, and every downstream number inherits that scope without saying so anywhere in the output. Catch it by auditing what population a sample was *actually* drawn from — the exact directory/query used — not what the report's title implies. Here: a full distribution-shift measurement pass sampled only from the food subset of the merged Indian-context set, leaving 7,768 non-food Wikimedia Commons images (festivals, temples, weddings) completely unrepresented in a result reported as covering "Indian-context images" generally — fixed by sampling from the merged annotation set directly.

4. **A ranking metric can have no valid null case for the system as designed.** Recall@K assumes a ground-truth "correct" item genuinely absent from the query; if the system's own stored embeddings *are* its ground truth, there may be no cross-image relevance judgment left to recover once a query is properly held out — the metric isn't measuring badly, it's measuring something undefined for this task. Catch it by constructing the exact null case (query fully removed) before publishing a rank-based metric, and checking whether the metric's true value is even defined — not just whether the number looks good. Here: a three-variant breakdown (self-match baseline, random-vector control, genuine held-out) showed zero valid signal in the held-out case, because COCO provides no cross-image relevance judgments for this task; the claim was retracted rather than patched.

5. **A theoretical calculation can pass as a measured, achievable result.** "N bytes saved" from casting an array's dtype is true of the array in isolation and says nothing about whether the system that actually stores it benefits. Catch it by tracing the number through to the specific storage/serving layer that would need to realize it, not stopping at the array math. Here: fp16/int8 "memory saved" was computed on numpy arrays pulled out of ChromaDB, whose HNSW index stores everything as float32 internally regardless of what was fed in — not an achievable saving without a different vector store.

6. **A summary percentage can silently average over two different questions.** "X% changed" doesn't say whether X% of *any* position in a result set changed, or whether the single *best* pick changed — these can differ by 50x on identical data. Catch it by naming precisely what a percentage counts before reporting it, and treating two related numbers that are mathematically inconsistent with each other as a bug report, not noise. Here: an original, undifferentiated "8.45% changed" figure was split into top-10 set overlap (99.65% stable, i.e. 0.35% churn) and top-1 flip rate — which itself first came back at an impossible 76.5% next to that 99.65%-stable top-10 set, traced to an unstable sort's arbitrary tie-breaking on the many exact-similarity ties COCO's near-duplicate images produce. Fixed, the real number is 1.5%, and running the actual downstream task (real captions scored against real references) shows even that residual flipping doesn't cost caption quality. Full detail: `results/README.md`.

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
- **Retracted claims and their corrections are documented, not hidden** — Recall@K had no valid non-trivial replacement once genuinely held out and was withdrawn, and the index-quantization result went through three rounds of correction before it was trustworthy. See "Measurement traps in retrieval evaluation" above and `results/README.md` for the full detail on both.
