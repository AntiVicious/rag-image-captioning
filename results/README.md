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
