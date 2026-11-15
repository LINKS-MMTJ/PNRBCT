# Experimental Protocol

This document describes the final experimental setup used for the paper and maps
every reported number to the file that contains it.

---

## 1. Dataset

| Item | Value |
|---|---|
| Annotated images | 845 (grid 238, horizontal 375, vertical 232) |
| Image-level split | train 676 / valid 85 / test 84 |
| End-to-end test images | 150 (50 per class, disjoint from the 845) |
| Grid partition | 6 x 8 = 48 cells per image |
| Grid resolution | 448 x 448 |

The end-to-end test set was corrected on 2026-10-26: three images that
overlapped with the 845 annotated pool were replaced, and all end-to-end
metrics were recomputed on the corrected 150-image set.

For evaluation, non-crack grids are downsampled to match the crack count
(1:1 protocol). Files are never deleted; downsampling is applied only when
building the evaluation subset.

---

## 2. Stage 1 — Binary CARE Transformer

### Training configuration

| Parameter | Value |
|---|---|
| Epochs | 150 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.03 |
| Batch size | 512 |
| LR schedule | 5-epoch linear warmup + cosine annealing to 1e-6 |
| Loss | Cross-entropy, label smoothing 0.1 |
| Class weighting | None (training set balanced 1:1) |
| Gradient clipping | 1.0 |
| Input size | 448 x 448 |
| Seed | 42 |

Model selection uses macro-F1 on the valid split. All metrics reported in this
document use the 1:1 evaluation protocol.

### Results (test set, 1:1 protocol)

| Metric | Value |
|---|---|
| Accuracy | 94.12% |
| Non-crack F1 | 94.25% |
| Crack F1 | 93.99% |
| Macro F1 | 94.12% |
| Confusion matrix | [[860, 33], [72, 821]] |

The 1:1 evaluation sets contain 15,492 training, 2,020 validation and 1,786
test grids (equal crack/non-crack counts in each split).

1:1 training / validation accuracy: 96.63% / 94.80%.

Files:
- `results/stage1/stage1_448_bal_eval_meta.json`
- `results/stage1/caret_s0_448_bal_train_metrics.csv`
- `results/verification/verify_s1_bal.json`

The training CSV logs the validation curve used during training; the 1:1
metrics above are recomputed by `eval_stage1.py` and stored in
`stage1_448_bal_eval_meta.json`.

---

## 3. Stage 2 — PNR

### Decision set

PNR only examines crack grids whose Stage-1 probability is below 0.85:

| Split | Crack cells | Gated cells | Noise | Valid crack | 1:1 training subset |
|---|---|---|---|---|---|
| train | 8,418 | 1,214 | 661 | 553 | 1,106 |
| valid | 1,088 | 185 | 96 | 89 | 178 |

The gated set is balanced 1:1 by downsampling the majority class with a fixed
seed (42). The loss is computed on the balanced subset; metrics are computed on
the full gated set, matching inference.

### Training configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 5e-3 |
| Batch size | 32 |
| Epochs | 60 |
| Dropout | 0.3 |
| Seed | 42 |

### Results (test split, 84 images, Table I)

| | Precision | Recall | F1 |
|---|---|---|---|
| Noise-grid detection | 0.5278 | 0.3423 | 0.4153 |
| Crack-grid misdetection | 0.4722 | 0.0381 | 0.0705 |

True-crack mis-removal: 34 / 893 = 3.81%.
Grid accuracy before / after denoising: 95.46% / 95.56%.

Table I uses the test split (84 images) and is stored in
`results/stage2/table1_test.json`. The `best_epoch_val` and `full_dataset`
entries in `pnr_eval_meta.json` are training-time validation/full-dataset
metrics with different scopes, and `verify_report.json` uses the full
train+valid actionable set; neither is the source of Table I.

Files:
- `results/stage2/table1_train.json`, `table1_valid.json`, `table1_test.json`
- `results/stage2/pnr_eval_meta.json`
- `results/stage2/pnr_dataset_stats.json`

---

## 4. Stage 3 — CDCC

### Architecture

```
48-dim binary grid (6 x 8)
  -> Conv2d(1 -> 8, kernel 2x2) + ReLU   (384 features)
  -> Linear(384 -> 256) + LayerNorm + ReLU + Dropout
  -> Linear(256 -> 256) + LayerNorm + ReLU + Dropout
  -> Linear(256 -> 3)
```

Parameters: 166,187.

### Training configuration

| Parameter | Value |
|---|---|
| Training images | 676 (internal 8:2 split: 540 / 136; the 136 is an internal validation split, not the manifest valid split) |
| Epochs | 80 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-3 |
| Batch size | 64 |
| Dropout | 0.25 |
| Seed | 42 |

### Results

End-to-end (150 images, with PNR):

| Metric | Value |
|---|---|
| Accuracy | 96.67% |
| Macro F1 | 0.9666 |
| Alligator | 98.0% |
| Transverse | 100.0% |
| Longitudinal | 92.0% |
| Confusion matrix | [[49, 1, 0], [0, 50, 0], [4, 0, 46]] |

Comparison on the internal validation split (136 samples):

| Model | Parameters | Accuracy | Macro F1 |
|---|---|---|---|
| CDCC | 166,187 | 95.59% | 95.09% |
| Position-agnostic MLP | 80,131 | 67.65% | 61.44% |

The position-agnostic baseline is a three-layer 48-256-256-3 MLP; only the
spatial order of the 48 input values is randomly permuted per sample.

Files:
- `results/stage3/e2e_cdcc_pnr_eval_meta.json`
- `results/stage3/e2e_cdcc_nopnr_eval_meta.json`
- `results/stage3/cdcc_pnr_eval_meta.json`
- `results/stage3/agnostic_pnr_agnostic_eval_meta.json`

---

## 5. Latency

| Module | Input | Latency |
|---|---|---|
| CARE Transformer | one 448 x 448 grid, 48 grids batched | 0.509 ms/grid |
| PNR | one 2 x 6 x 8 detection matrix | 0.068 ms/image |
| CDCC | one 48-dim vector | 0.142 ms/image |

Measured with CUDA events after warm-up on an RTX 5090. Absolute values depend
on the GPU; relative comparisons are stable.

The workspace-level `tools/bench_stages.py` regenerates
`results/timing/bench_stages_meta.json`.

---

## 6. Verification

The workspace-level `tools/verify_all.py` recomputes the data split, Stage-1
metrics and PNR metrics directly from the cached tensors and annotations, and
writes `results/verification/verify_report.json`. The current report was
generated on 2026-10-05, before the image-ID normalization; the normalization
does not change the split membership or the aggregate counts/metrics.

The workspace-level `tools/table1_by_split.py {train,valid,test}` recomputes
Table I for a given split.
