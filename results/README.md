# Results

Summary tables for the experiments in the main README, with the commands to
reproduce them. The raw per-image outputs (`eval_output/`,
`dist_shift_output/`) are not committed; each CSV here is the aggregate of
one run.

## Selection Strategy × Crop Configuration (`selection_strategy_ablation.csv`)

N = 200 held-out COCO val2017 images, seed 42:

| Selection | Crop config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|---|
| top1 (nearest neighbour) | none (retrieval-only) | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.28 |
| top1 | segmentation | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| top1 | object-detection | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| top1 | both (all-seven) | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |
| medoid (consensus) | none (retrieval-only) | 0.106 | 0.170 | 0.380 | 0.486 | 0.677 | 0.47 |
| **medoid** | **segmentation** | **0.128** | **0.178** | **0.386** | **0.537** | 0.670 | 10.46 |
| medoid | object-detection | 0.116 | 0.169 | 0.371 | 0.506 | 0.658 | 3.46 |
| medoid | both (all-seven) | 0.126 | 0.171 | 0.377 | 0.512 | 0.663 | 12.84 |
| top1 | random crops (control, no DETR) | 0.095 | 0.160 | 0.359 | 0.479 | 0.653 | 0.99 |

**Default:** medoid selection with segmentation crops
(`Config.selection_mode="medoid"`, `Config.enable_segmentation=True`,
`Config.enable_object_detection=False`), the best CIDEr measured, at about
10 s/image on CPU. `--skip-detr` gives the faster 0.47 s/image path.

**Interpretation:**

- Under **top1**, DETR crops score worse than no crops and worse than six
  large random crops at the same number of variants. Small, single-object
  crops retrieve captions that don't describe the whole scene, and with top1
  one of them often wins.
- Under **medoid**, which picks the candidate the rest of the pool agrees
  with (CLIP text-embedding similarity, one extra `encode_text()` call), a
  larger and more varied pool helps, and segmentation crops become the best
  configuration.

Testing only one selection strategy would have given the wrong conclusion
about crops in either direction.

Reproduce:

```bash
docker build -f Dockerfile.eval -t rag-image-captioning:eval .

# top1 selection, all four crop configurations
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval \
    --out /app/eval_output/eval_results.json --output-mode top1

# medoid selection, all four crop configurations
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval_medoid \
    --out /app/eval_output/eval_results_medoid.json --output-mode medoid --skip-baselines

# random-crop control, top1 selection
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

## Baselines, top1 Selection (`ablation_table.csv`)

| Config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|
| generic-floor | 0.021 | 0.067 | 0.322 | 0.009 | 0.543 | 0.00 |
| top1-verbatim | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.24 |
| retrieval-only | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.28 |
| +segmentation | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| +object-detection | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| all-seven | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |

- **generic-floor** gives every image the same fixed caption: the score with
  no information.
- **top1-verbatim** and **retrieval-only** are independent code paths that
  must agree; `scripts/evaluate.py` checks this, and they match exactly.

An earlier version reported CIDEr of about 7×10⁻¹⁰ for crop configurations.
That was a bug: the captions retrieved for every crop were concatenated into
one long block, so the metric measured length rather than quality. The
fixed evaluation returns one caption per configuration.

## Distribution Shift, Indian-Context Images (`distribution_shift_summary.csv`)

N = 200 per row, sampled across the 12,538-image Indian-context set (73 food
and 127 Wikimedia Commons images, matching the set's composition). Values
are ChromaDB top-1 distances; lower means a closer match. The corrected row
excludes near-duplicate twins (CLIP similarity ≥ 0.95) from the candidates;
about 46% of the set has one, mostly in the food dataset.

| | Mean distance | Median distance | vs. COCO baseline |
|---|---|---|---|
| COCO baseline | 0.335 | 0.337 | — |
| Indian, before augmentation | 0.522 | 0.509 | +55.9% |
| Indian, after augmentation (uncorrected) | 0.127 | 0.099 | −62.1% (inflated by near-duplicates) |
| **Indian, after augmentation (corrected)** | **0.270** | **0.246** | **−19.5%** |

The Indian set covers a narrower range (18 Commons categories and 15 food
dishes, against COCO's 80 object categories), so part of the lower distance
comes from that density rather than from augmentation alone.

Reproduce:

```bash
docker run --rm \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/chroma_db_before_augmentation:/app/chroma_db_before" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/measure_distribution_shift_v2.py

# near-duplicate audit
docker run --rm \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/audit_near_duplicates.py
```

## Hand-Labelled Validation (`hand_labeled_summary.csv`)

N = 53 Indian-context images (20 food, 33 Commons), each given a reference
caption written by hand after viewing the image. Scored before and after
augmentation with pycocoevalcap. These references differ from COCO's, so the
scores aren't comparable with the tables above.

| Metric | Before | After |
|---|---|---|
| BLEU-4 | 0.0119 | 0.0183 |
| METEOR | 0.0469 | 0.0815 |
| ROUGE-L | 0.1545 | 0.1815 |
| CIDEr | 0.1128 | 0.2879 |

All four metrics improve, consistent with the distance results. The sample
grew from an initial 20 images; around 150 would give a stable corpus-level
CIDEr. References are in `dist_shift_output/hand_labels.json`.

Reproduce:

```bash
docker run --rm \
  -v "/path/to/merged_images:/app/merged_images" \
  -v "/path/to/chroma_db_before_augmentation:/app/chroma_db_before" \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/score_hand_labeled.py
```

## Recall@K: Withdrawn (`recall_at_k.csv`)

An earlier version reported R@1 = R@5 = R@10 = 99%. That came from each query
image finding its own embedding in the index, since all of an image's
captions share one stored image embedding. Three variants on the same 200
images:

| Variant | R@1 | R@5 | R@10 | Meaning |
|---|---|---|---|---|
| Not held out, real queries | 0.99 | 0.99 | 0.99 | Self-match |
| Not held out, random-vector queries | 0.00 | 0.00 | 0.00 | Confirms the 99% is the self-match |
| Query images removed from the index | 0.00 | 0.00 | 0.00 | No correct item remains; COCO has no cross-image relevance labels |

Recall@K has no meaningful value for this system, so no Recall@K claim is
made. The ablation and distribution-shift results above are the retrieval
evidence.

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

## Index Quantization (`quantization_summary.csv`, `quantization_downstream_summary.csv`)

Does storing embeddings at lower precision change retrieval? Three
corrections were needed before the numbers were reliable:

1. **Memory savings are theoretical.** ChromaDB's HNSW index stores float32
   and has no quantization option
   ([collection configuration](https://docs.trychroma.com/docs/collections/configure)),
   so the sizes below are array arithmetic only. A store with scalar
   quantization, such as Qdrant, would be needed to realise them.
2. **Sibling rows leaked.** Each COCO image has about 5 caption rows sharing
   one embedding. Excluding only the sampled row left its siblings as
   guaranteed matches; all sibling rows are now excluded.
3. **Two questions were mixed.** "8.45% changed" combined top-10 set changes
   with top-1 changes. The top-1 rate first measured 76.5%, which is
   impossible next to a 99.65%-stable top-10 set. The cause was an unstable
   sort: 152 of 200 queries had exact similarity ties at rank 0, which were
   broken differently for float32 and float16. With deterministic
   tie-breaking (lowest index, as `np.argmax` does) the rate is 1.5%,
   matching a separate script (`scripts/measure_quantization_downstream.py`).

| Representation | Memory (theoretical) | Top-10 overlap vs. float32 | Top-1 flip rate |
|---|---|---|---|
| float32 | 204.8 MB | 100% | — |
| float16 | 102.4 MB (−50%) | 99.65% | 1.5% (3/200) |
| int8 | 51.2 MB (−75%) | 91.15% | not measured |

Scoring the float32-selected and float16-selected top-1 captions against
COCO references (same 200 queries):

| Metric | float32 | float16 | Difference |
|---|---|---|---|
| BLEU-4 | 0.1254 | 0.1235 | −0.0019 |
| METEOR | 0.1819 | 0.1816 | −0.0003 |
| ROUGE-L | 0.3776 | 0.3767 | −0.0009 |
| CIDEr | 0.5073 | 0.5076 | +0.0003 |

All differences are within noise: the few float16 flips land on captions of
similar quality. int8 was not evaluated at the top-1 or caption level.

Reproduce:

```bash
docker run --rm \
  -v "/path/to/chroma_db_scale_test:/app/chroma_db_scale_test" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:local python scripts/quantize_index.py \
    --chroma-db-dir /app/chroma_db_scale_test --max-embeddings 100000 --num-queries 200 --k 10

# caption-quality effect of float16 (needs the eval image for METEOR)
docker run --rm \
  -v "/path/to/chroma_db_scale_test:/app/chroma_db_scale_test" \
  -v "/path/to/coco/annotations:/app/coco_annotations" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/measure_quantization_downstream.py
```
