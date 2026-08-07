# Results

Tracked, reproducible summaries of the experiments referenced in the top-level README. The raw per-image prediction dumps these are computed from (`eval_output/`, `dist_shift_output/`) are gitignored — they're multi-hundred-KB regeneratable artifacts, not source. What's committed here is the small aggregate CSV each raw run produces, plus the exact command to regenerate it.

## Ablation table (`ablation_table.csv`)

N=200, held-out split of COCO val2017 (seed=42), output length held constant across configs (`--output-mode top1`, the default).

| Config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|
| generic-floor | 0.021 | 0.067 | 0.322 | 0.009 | 0.543 | 0.00 |
| top1-verbatim | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.24 |
| **retrieval-only** | **0.098** | **0.160** | **0.352** | **0.469** | **0.672** | 0.28 |
| +segmentation | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| +object-detection | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| all-seven | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |

`retrieval-only` and `top1-verbatim` are two independent code paths that should always agree (see `scripts/evaluate.py`'s built-in consistency check) — they do here, bit-for-bit, which is itself evidence the top1 selection logic is correct.

Reproduce:
```bash
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
```

## Distribution shift, Indian-context images (`distribution_shift_summary.csv`)

N=200 per row, sampled uniformly across the full 12,538-image merged Indian-context set (73 food + 127 Wikimedia Commons in this run — proportional to the set's actual ~38%/62% composition). ChromaDB top-1 distance, lower = closer/more confident retrieval match. "corrected" excludes near-duplicate twins (≥0.95 CLIP similarity) from the candidate index, not just the query image's own embedding — ~46% of the Indian set has such a twin, mostly from the templated-caption food dataset.

| | mean distance | median distance | vs. COCO baseline |
|---|---|---|---|
| COCO baseline | 0.335 | 0.337 | — |
| Indian, before augmentation | 0.522 | 0.509 | +55.9% (worse) |
| Indian, after augmentation (uncorrected) | 0.127 | 0.099 | −62.1% (looks better, but leaky) |
| **Indian, after augmentation (dedup-corrected)** | **0.270** | **0.246** | **−19.5% (real)** |

Reproduce:
```bash
docker run --rm \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/chroma_db_before_augmentation:/app/chroma_db_before" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/measure_distribution_shift_v2.py
```

Near-duplicate audit that motivated the correction:
```bash
docker run --rm \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/audit_near_duplicates.py
```

## Hand-labeled validation (`hand_labeled_summary.csv`)

N=20 Indian-context images (8 food + 12 Commons), each hand-viewed and given a real, from-scratch reference caption — not derived from the dataset's own template/scrape captions. Scored before vs. after augmentation with the same pycocoevalcap suite as the main ablation.

| Metric | Before | After |
|---|---|---|
| BLEU-4 | ~0.000003 | ~0.000005 |
| METEOR | 0.044 | 0.086 |
| ROUGE-L | 0.158 | 0.191 |
| CIDEr | 0.129 | 0.310 |

BLEU-4 is uninformative at N=20 with one reference per image (needs multiple references and/or a much larger N before it means anything) — included for completeness, not as evidence either way. METEOR/ROUGE-L/CIDEr all move the same direction as the distance-based measurement above, independently, which is the point of having built this: distance-based and reference-based methodology agreeing is stronger evidence than either alone. Scaling this hand-labeled set past N=20 is open follow-up work; references live in `dist_shift_output/hand_labels.json`.

Reproduce:
```bash
docker run --rm \
  -v "/path/to/merged_images:/app/merged_images" \
  -v "/path/to/chroma_db_before_augmentation:/app/chroma_db_before" \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/score_hand_labeled.py
```

## Recall@K -- retriever quality, independent of caption text (`recall_at_k.csv`)

N=200. The ablation table above measures END-TO-END caption quality (does the retrieved text match a reference caption), which conflates retriever quality with selection/aggregation quality. Recall@K isolates the first question: for a query image, rank every caption in the corpus by CLIP similarity and check whether the image's OWN ground-truth caption lands in the top K (standard image-to-text retrieval formulation, no held-out split -- the "positive" being tested is whether an image's own captions rank highly among everything, including itself).

| K | Recall@K |
|---|---|
| 1 | 0.990 |
| 5 | 0.990 |
| 10 | 0.990 |

Median rank of the first own-caption hit, when found: **1** (i.e. usually the literal #1 result). Only 2/200 images had no own-caption in the top 10. **The retriever itself is not the bottleneck** — when end-to-end caption quality falls short of a reference, it's because a correctly-similar image's caption still isn't the same sentence a human would write fresh for this specific photo, not because CLIP+ChromaDB failed to find it.

Reproduce:
```bash
docker run --rm \
  -v "/path/to/coco:/app/coco" \
  -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:local python scripts/measure_recall_at_k.py \
    --chroma-db-dir /app/chroma_db \
    --coco-img-dir /app/coco/val2017 \
    --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --num-samples 200 --ks 1,5,10
```

## Selection strategy x crop source (`selection_strategy_ablation.csv`)

The main ablation table picks the single retrieved candidate CLOSEST TO THE QUERY IMAGE (`--output-mode top1`). **Medoid selection** — pick the candidate most similar, on average, to the *rest* of the retrieved pool (a consensus pick, via CLIP text-embedding similarity) — is a different, ~20-line strategy for consuming the same retrieved pool, tested as `--output-mode medoid`.

| Selection | Config | CIDEr | s/image |
|---|---|---|---|
| top1 | retrieval-only | 0.469 | 0.28 |
| top1 | +segmentation | 0.422 | 10.43 |
| top1 | all-seven (DETR) | 0.395 | 11.87 |
| top1 | **+random-crops** (blind, no DETR) | **0.479** | 0.99 |
| medoid | retrieval-only | 0.486 | 0.47 |
| **medoid** | **+segmentation** | **0.537** | 10.46 |
| medoid | all-seven (DETR) | 0.512 | 12.84 |

Two things happen at once here, and they're independent variables, not one:

1. **Medoid selection is a strictly better, near-free upgrade over top1.** `retrieval-only` alone goes from 0.469 (top1) to 0.486 (medoid) at +0.19s/image (CLIP text-encoding the small candidate pool). Under medoid, **`+segmentation` becomes the single best config measured anywhere in this project** (CIDEr 0.537) — crops DO help, just not when consumed by nearest-neighbour selection.
2. **DETR's specific crop *characteristics* — not "having extra crops" — are what hurt under top1.** `+random-crops` (6 blind, large [40-80% of image] crops, no DETR, same variant count as all-seven) scores 0.479 under top1 — matching/slightly beating `retrieval-only`, and clearly beating DETR's `all-seven` (0.395), at 1/14th the latency. DETR's crops are typically small and zoomed to a single object; large blind crops stay closer to the whole-image gestalt the reference captions actually describe.

Net effect: the honest headline is not "crops hurt" or "crops help" — it's "top1 selection was the wrong way to consume a large or DETR-zoomed candidate pool; medoid consumes it correctly, and at that point DETR's crops earn their latency cost."

Reproduce:
```bash
# medoid selection, all four crop configs
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval_medoid \
    --out /app/eval_output/eval_results_medoid.json --output-mode medoid --skip-baselines

# random-crop control (blind crops vs DETR crops vs no crops), top1 selection
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval_randcrop \
    --out /app/eval_output/eval_results_random_crop_control.json \
    --output-mode top1 --ablations retrieval-only,all-seven --random-crop-control --skip-baselines
```

## Index quantization (`quantization_summary.csv`)

100,000-embedding sample of the 591,753-embedding train2017-scale index (see the walkthrough doc's scale-test section) — full-index fetch OOM-killed on this machine's WSL2 memory cap (chromadb's `.get()` materialises the whole result as Python lists-of-lists before any numpy conversion, which alone exceeds it; 100k is a large, still-representative sample). Brute-force cosine top-10 search, N=200 queries: recall of each quantized representation's top-10 against the float32 ground truth.

| Representation | Memory | Recall@10 vs. float32 |
|---|---|---|
| float32 (baseline) | 204.8 MB | 100% |
| **float16** | **102.4 MB (−50%)** | **99.15% (−0.85%)** |
| int8 | 51.2 MB (−75%) | 93.90% (−6.10%) |

**float16 is close to free** — half the memory for under 1% recall loss, no accuracy-sensitive use case would notice. **int8 is a real tradeoff** — 75% memory saved, but a genuine 6% of top-10 results change, worth it only under real memory pressure (e.g. the full 591k-scale index at int8 would be ~300MB instead of ~1.2GB).

Reproduce:
```bash
docker run --rm \
  -v "/path/to/chroma_db_scale_test:/app/chroma_db_scale_test" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:local python scripts/quantize_index.py \
    --chroma-db-dir /app/chroma_db_scale_test --max-embeddings 100000 --num-queries 200 --k 10
```
