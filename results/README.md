# Results

Tracked, reproducible summaries of the experiments referenced in the top-level README. The raw per-image prediction dumps these are computed from (`eval_output/`, `dist_shift_output/`) are gitignored — they're multi-hundred-KB regeneratable artifacts, not source. What's committed here is the small aggregate CSV each raw run produces, plus the exact command to regenerate it.

## The headline finding: selection strategy × crop config (`selection_strategy_ablation.csv`)

**Read this table first — it's the actual conclusion.** Two earlier, separately-published tables (an ablation that said "crops hurt" and a follow-up that said "crops help") each told only half the story because each held one variable fixed. This is both variables at once, N=200, held-out COCO val2017 split, seed=42:

| Selection | Crop config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|---|
| top1 (nearest-neighbor) | none (retrieval-only) | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.28 |
| top1 | segmentation | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| top1 | object-detection | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| top1 | both (all-seven) | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |
| medoid (consensus) | none (retrieval-only) | 0.106 | 0.170 | 0.380 | 0.486 | 0.677 | 0.47 |
| **medoid** | **segmentation** | **0.128** | **0.178** | **0.386** | **0.537** | 0.670 | 10.46 |
| medoid | object-detection | 0.116 | 0.169 | 0.371 | 0.506 | 0.658 | 3.46 |
| medoid | both (all-seven) | 0.126 | 0.171 | 0.377 | 0.512 | 0.663 | 12.84 |
| top1 | random-crops (blind control, no DETR) | 0.095 | 0.160 | 0.359 | 0.479 | 0.653 | 0.99 |

**Shipped default: medoid selection + segmentation crops** (`Config.selection_mode="medoid"`, `Config.enable_segmentation=True`, `Config.enable_object_detection=False`) — the best CIDEr of any config measured, at a real but bounded latency cost (~10s/image on this CPU-only dev machine). `--skip-detr` / `enable_segmentation=False` is there for anyone who wants the faster (0.47s/image), slightly-lower-scoring path instead.

**Why the two earlier tables disagreed:** "how you consume the retrieved candidate pool" and "what you crop" are independent variables.
- Under **top1** (pick whichever candidate is closest to the query image), more/noisier candidates actively hurt — DETR's crops score *worse* than doing nothing, and *worse* than six blind random crops (bottom row) at the same variant count. It's specifically DETR's small, single-object-zoomed crops that drift from the whole-scene description a reference caption gives; large blind crops don't have that problem.
- Under **medoid** (pick whichever candidate the rest of the pool agrees with, via CLIP text-embedding similarity — ~20 lines, one extra `encode_text()` call), a bigger, more diverse pool is an asset, not a liability: segmentation crops become the best config measured anywhere in this project.

That reversal — "how you consume a retrieved set matters as much as how you retrieve it" — generalizes past this one captioning system; it's a property of RAG architectures broadly, not a COCO-specific quirk.

Reproduce:
```bash
docker build -f Dockerfile.eval -t rag-image-captioning:eval .

# top1 selection, all four crop configs (the original ablation)
docker run --rm \
  -v "/path/to/coco:/app/coco" -v "/path/to/chroma_db:/app/chroma_db" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:eval python scripts/evaluate.py \
    --num-eval-images 200 --seed 42 \
    --coco-img-dir /app/coco/val2017 --coco-ann-file /app/coco/annotations/captions_val2017.json \
    --source-chroma-db /app/chroma_db --eval-chroma-db /app/chroma_db_eval \
    --out /app/eval_output/eval_results.json --output-mode top1

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

## How this was found: two measurements that each told half the story (`ablation_table.csv`)

The table above didn't happen in one pass. First pass, top1 selection only:

| Config | BLEU-4 | METEOR | ROUGE-L | CIDEr | CLIPScore | s/image |
|---|---|---|---|---|---|---|
| generic-floor | 0.021 | 0.067 | 0.322 | 0.009 | 0.543 | 0.00 |
| top1-verbatim | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.24 |
| retrieval-only | 0.098 | 0.160 | 0.352 | 0.469 | 0.672 | 0.28 |
| +segmentation | 0.083 | 0.157 | 0.351 | 0.422 | 0.637 | 10.43 |
| +object-detection | 0.087 | 0.152 | 0.336 | 0.406 | 0.647 | 2.75 |
| all-seven | 0.074 | 0.151 | 0.337 | 0.395 | 0.634 | 11.87 |

`retrieval-only` and `top1-verbatim` are two independent code paths that should always agree (see `scripts/evaluate.py`'s built-in consistency check) — they do here, bit-for-bit, which is itself evidence the top1 selection logic is correct. `generic-floor` (a single fixed caption for every image, zero retrieval) is the score floor: what "no information" looks like on these metrics.

This table alone said "crops hurt, don't use them" — true only for the one candidate-selection strategy it tested. Testing a second strategy (medoid, see above) reversed the crop conclusion without changing the retrieval or the crops themselves. Neither table is wrong; each is a slice of the real, two-variable result at the top of this document.

*(An even earlier version of this table reported CIDEr ≈ 7×10⁻¹⁰ for the crops-on config — a real bug, not a real result: concatenating every crop's top-k retrieved captions into one long block confounded length-sensitive metrics with verbosity. Fixed by holding output length constant — one best candidate per config — before comparing; that fix is what produced the top1 numbers above.)*

Reproduce: see the top1 command in the section above.

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

N=53 Indian-context images (20 food + 33 Commons), each hand-viewed and given a real, from-scratch reference caption — not derived from the dataset's own template/scrape captions. Scored before vs. after augmentation with the same pycocoevalcap suite as the main ablation.

| Metric | Before | After |
|---|---|---|
| BLEU-4 | 0.0119 | 0.0183 |
| METEOR | 0.0469 | 0.0815 |
| ROUGE-L | 0.1545 | 0.1815 |
| CIDEr | 0.1128 | 0.2879 |

**Grown from an initial N=20 spot-check; still short of the ~150 that would make this a fully stable corpus-level CIDEr number, and still labeled as such rather than overclaimed.** At N=53, all four metrics — including BLEU-4, which was flatly ~0 and uninformative at N=20 — now move the same direction, with BLEU-4 showing real (if small) signal for the first time. METEOR/ROUGE-L/CIDEr all improve after augmentation, consistent with the distance-based measurement above and with the earlier N=20 pass; this is now real independent corroboration, not just "nothing is catastrophically broken." Scaling further toward N=150 remains the clearest way to firm up the Indian-context work's most differentiated result, and is the most valuable open item if this project continues. References live in `dist_shift_output/hand_labels.json`.

Reproduce:
```bash
docker run --rm \
  -v "/path/to/merged_images:/app/merged_images" \
  -v "/path/to/chroma_db_before_augmentation:/app/chroma_db_before" \
  -v "/path/to/chroma_db_augmented:/app/chroma_db_augmented" \
  -v "/path/to/dist_shift_output:/app/dist_shift_output" \
  rag-image-captioning:eval python scripts/score_hand_labeled.py
```

## Recall@K — retracted (`recall_at_k.csv`)

An earlier version of this document reported R@1=R@5=R@10=99.0% as evidence "the retriever is not the bottleneck." **That number was a self-match artifact, not a result, and has been withdrawn.** The query image's own embedding was never excluded from the index before searching, and every caption for an image in this system shares ONE stored CLIP *image* embedding (captions aren't independently text-embedded — this system retrieves nearest *images* and passes through whichever image's captions it lands on). Re-encoding an image and searching an index that already contains that exact image's embedding is close to guaranteed to find itself first; that's pipeline determinism, not retrieval quality.

Three variants, same 200-image sample, to make this explicit instead of asserting it:

| Variant | R@1 | R@5 | R@10 | What it shows |
|---|---|---|---|---|
| baseline (unheld-out, real queries) | 0.99 | 0.99 | 0.99 | The original, broken number — self-match |
| random control (unheld-out index, random-vector queries) | 0.00 | 0.00 | 0.00 | Confirms baseline's 99% is specifically the self-match leak, not a wiring bug elsewhere |
| held-out (query images fully excluded) | 0.00 | 0.00 | 0.00 | Once genuinely absent, there is no ground-truth "correct" item — COCO provides no cross-image relevance judgments for this task |

**Conclusion: there is no valid, non-trivial Recall@K claim to make about this system as originally framed**, and none is claimed anymore. What retrieval-quality evidence this project actually has is in the held-out ablation and distribution-shift work above, both of which score retrieved *text* against real references/distances under a genuine hold-out — not "did it find itself."

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

## Quantization: a precision-robustness result, not a memory result (`quantization_summary.csv`)

An earlier version of this document reported "50% memory saved for <1% recall loss" as a fp16 win. **Both halves of that claim needed correcting.**

**The memory number was theoretical, not measured**, and it isn't achievable with this project's actual storage layer: checked against Chroma's own docs ([Configure Collections](https://docs.trychroma.com/docs/collections/configure) lists the HNSW config surface — `space`, `ef_construction`, `ef_search` — no dtype/quantization parameter exists) and corroborating third-party comparisons ([Chroma vs FAISS](https://myengineeringpath.dev/tools/chroma-vs-faiss/), [Qdrant vs Chroma](https://www.kunalganglani.com/blog/qdrant-vs-chroma)), ChromaDB's HNSW index stores vectors as float32 internally with no native quantization support. Feeding it float16-rounded values wouldn't reduce what it stores or serves — it re-expands to float32 regardless. The MB figures below are pure bytes-per-element arithmetic on arrays pulled *out* of ChromaDB into numpy for this experiment, not a deployable result. Realizing an actual memory saving would require a different vector store (Qdrant's scalar quantization was the example that came up in research).

**The recall number was inflated by a trivial self-match.** The 200 query embeddings are themselves rows of the 100,000-embedding index, and the original version of this script didn't exclude a query's own row from its own candidate search — since a self-match survives quantization almost by definition, that guaranteed roughly 1 of the top-10 slots "for free" regardless of how good or bad the quantization actually was. Corrected (self-matches excluded from both the float32 ground truth and the quantized search):

| Representation | Theoretical memory (not achievable in ChromaDB) | Ranking Recall@10 vs. float32 (self-matches excluded) |
|---|---|---|
| float32 (baseline) | 204.8 MB | 100% |
| float16 | 102.4 MB (−50%, theoretical) | 91.55% (−8.45%) |
| int8 | 51.2 MB (−75%, theoretical) | 87.75% (−12.25%) |

**The honest finding: retrieval ranking is reasonably, not extremely, robust to float16 rounding** — roughly 1 in 12 of the top-10 results changes, a real cost, not the "near-free" result originally claimed. int8 changes roughly 1 in 8. Neither number currently translates into an actual memory saving for this project's ChromaDB-based storage.

Reproduce:
```bash
docker run --rm \
  -v "/path/to/chroma_db_scale_test:/app/chroma_db_scale_test" \
  -v "/path/to/eval_output:/app/eval_output" \
  rag-image-captioning:local python scripts/quantize_index.py \
    --chroma-db-dir /app/chroma_db_scale_test --max-embeddings 100000 --num-queries 200 --k 10
```
