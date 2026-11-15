# PNRBCT

**Official public code repository for the paper**

**PNRBCT: A Position Aware Noise Removed Binary CARE Transformer for Crack Classification under Limited Training Data**

Xianglong Wang, Mamatjan Ubulkasim, Aoran Liu, Jin Zeng, Zhigan Xu\*, Ganlong Wang, Xinhua Dong, Hongmu Han, Zhongzheng Yan

\* Corresponding author: Zhigan Xu

See the repository root [README](../README.md) for the dataset download
instructions, overall workspace layout, and citation.

This repository contains the reference implementation of PNRBCT, a three-stage
pipeline for pavement crack classification:

1. **Stage 1 — Binary CARE Transformer.** Each pavement image is divided into a
   6 x 8 grid (48 cells). A CARE Transformer predicts, for every cell, whether it
   is a crack grid, producing a detection matrix `G ∈ {0,1}^{2×6×8}` whose two
   channels hold the binary label and the crack probability.
2. **Stage 2 — PNR (Position-Aware Noise Remover).** A lightweight convolutional
   model re-examines low-confidence crack cells (`p < 0.85`) and removes
   spatially inconsistent false positives ("noise grids") while keeping genuine
   cracks.
3. **Stage 3 — CDCC (Crack-Grid Distribution Classifier).** The denoised 6 x 8
   crack map is flattened into a 48-dimensional vector and classified into
   alligator, transverse, or longitudinal cracks.

The three models are trained independently and chained at inference time.

---

## Repository structure

```
.
├── models/                      # Shared model definitions
│   ├── care_model.py            # Binary CARE Transformer
│   ├── pnr_model.py             # Position-Aware Noise Remover
│   └── cdcc_model.py            # CDCC and position-agnostic MLP
├── stage1_care_transformer/
│   ├── train_stage1.py
│   └── eval_stage1.py
├── stage2_pnr/
│   ├── gen_g_cache.py           # Stage-1 inference cache
│   ├── build_pnr_dataset.py
│   └── train_pnr.py
├── stage3_cdcc/
│   ├── gen_cdcc_data.py
│   ├── train_cdcc.py
│   └── test_e2e.py              # End-to-end PNR -> CDCC test
├── data/
│   ├── split_v2_manifest.json   # Image-level split (845 images)
│   ├── annotations/             # Grid-level annotation JSONs
│   └── cache/                   # Stage-1 G cache and PNR dataset
├── checkpoints/                 # Trained weights
├── results/                     # Evaluation outputs
├── figures/
├── scripts/
│   └── run_stage1.sh
└── requirements.txt
```

Data-preparation and verification helpers are kept in the **workspace-level
`tools/` directory**, one level above this repository:

```
workspace/
├── CARETR_Project/     # this repository
└── tools/              # resplit_image_level.py, gen_e2e_grids.py, table1_by_split.py, ...
```

Run those helpers from the workspace root.

---

## Installation

```bash
pip install -r requirements.txt
```

The code was developed and evaluated with PyTorch 2.x and timm 1.x. All
experiments reported in the paper were run on a single NVIDIA GeForce RTX 5090
(32 GB).

---

## Data preparation

The raw pavement images are **not** distributed with this repository. To
reproduce the experiments, place the data as follows:

```
data/
├── images/
│   ├── grid/
│   │   ├── annotated/       # 845 annotated photos (split by the manifest)
│   │   └── e2e_test/        # 150 end-to-end test photos
│   ├── horizontal/
│   │   ├── annotated/
│   │   └── e2e_test/
│   └── vertical/
│       ├── annotated/
│       └── e2e_test/
└── annotations/
    └── {grid,horizontal,vertical}/{cls}/{id}.json
```

The `annotated/` folders contain **all** annotated images; the actual
train/valid/test assignment is defined by `data/split_v2_manifest.json`
(676 / 85 / 84 images). The `e2e_test/` folders contain the 150 end-to-end
test images and are never used for training.

Then, **from the workspace root**, generate the image-level split and the
448 x 448 grids:

```bash
# 1) Image-level split (676 / 85 / 84) -> data/split_v2_manifest.json
python tools/resplit_image_level.py --grid-size 448 --materialize

# 2) Grids for the 150 end-to-end test images
python tools/gen_e2e_grids.py --grid-size 448
```

The split is stratified by crack category and uses a fixed seed (42), so the
manifest is deterministic.

> **Leakage note.** The split is performed at the image level: no image appears
> in more than one of train/valid/test. The end-to-end test images are separate
> from the 845 annotated images.

---

## Reproducing the results

### Stage 1 — Binary CARE Transformer

```bash
bash scripts/run_stage1.sh          # training (paper configuration)
python stage1_care_transformer/eval_stage1.py
```

Key hyperparameters: 150 epochs, AdamW (lr 1e-4, weight decay 0.03), batch size
512, 5-epoch linear warmup followed by cosine annealing to 1e-6, label smoothing
0.1, gradient clipping 1.0. Non-crack grids are downsampled to 1:1 at training
and evaluation time.

### Stage 2 — PNR

```bash
python stage2_pnr/gen_g_cache.py
python stage2_pnr/build_pnr_dataset.py
python stage2_pnr/train_pnr.py

# from the workspace root:
python tools/table1_by_split.py test
```

PNR only makes decisions on crack grids with Stage-1 probability below 0.85;
high-confidence crack grids are kept unchanged.

### Stage 3 — CDCC and end-to-end evaluation

```bash
python stage3_cdcc/gen_cdcc_data.py --arm nopnr
python stage3_cdcc/gen_cdcc_data.py --arm pnr

python stage3_cdcc/train_cdcc.py --arm pnr   --model cdcc
python stage3_cdcc/train_cdcc.py --arm nopnr --model cdcc
python stage3_cdcc/train_cdcc.py --arm pnr   --model agnostic

python stage3_cdcc/test_e2e.py --arm pnr   --arch cdcc \
    --cdcc-model checkpoints/best_cdcc_pnr.pth
python stage3_cdcc/test_e2e.py --arm nopnr --arch cdcc \
    --cdcc-model checkpoints/best_cdcc_nopnr.pth
```

---

## Results

### Stage 1 — grid-level detection (test set, 1:1 protocol)

| Metric | Value |
|---|---|
| Evaluation grids (1:1) | 1,786 (893 crack / 893 non-crack) |
| Accuracy | 94.12% |
| Non-crack F1 | 94.25% |
| Crack F1 | 93.99% |
| Macro F1 | 94.12% |

### Stage 2 — PNR (84 test images, Table I)

| | Precision | Recall | F1 |
|---|---|---|---|
| Noise-grid detection | 0.5278 | 0.3423 | 0.4153 |
| Crack-grid misdetection | 0.4722 | 0.0381 | 0.0705 |

True-crack mis-removal rate: 3.81% (34 / 893).

### PNR ablation (150 end-to-end test images)

| | without PNR | with PNR |
|---|---|---|
| Macro F1 | 94.02% | **96.66%** |
| Accuracy | 94.00% | **96.67%** |
| Latency per image | 0.277 ms | 0.524 ms

### Stage 3 — CDCC (150 end-to-end test images, with PNR)

| Metric | Value |
|---|---|
| Accuracy | 96.67% |
| Macro F1 | 0.9666 |
| Alligator (grid) | 98.0% |
| Transverse (horizontal) | 100.0% |
| Longitudinal (vertical) | 92.0% |

### CDCC comparison (internal validation split, 136 samples)

| Model | Parameters | Accuracy | Macro F1 |
|---|---|---|---|
| CDCC | 166,187 | 95.59% | 95.09% |
| Position-agnostic MLP | 80,131 | 67.65% | 61.44% |

The position-agnostic baseline is a three-layer 48-256-256-3 MLP whose 48 input
values are randomly permuted per sample, so it cannot exploit the spatial
arrangement of the grid.

---

## Model and latency

| Module | Parameters | Latency |
|---|---|---|
| Binary CARE Transformer | 5.29 M | 0.509 ms per grid (48 grids batched) |
| PNR | 37,032 | 0.068 ms per image |
| CDCC | 166,187 | 0.142 ms per image |



---

## Checkpoints

| File | Description |
|---|---|
| `checkpoints/caret_s0_448_bal_best.pth` | Stage-1 CARE Transformer |
| `checkpoints/best_pnr.pth` | PNR |
| `checkpoints/best_cdcc_pnr.pth` | CDCC trained on PNR-cleaned grids |
| `checkpoints/best_cdcc_nopnr.pth` | CDCC trained without PNR |
| `checkpoints/best_agnostic_pnr_agnostic.pth` | Position-agnostic MLP baseline |

---

## Data availability

The pavement images are distributed with this project as GitHub Release assets
(see the repository root README for download and reassembly instructions).
The image-level split manifest and the grid-level annotations are included in
`data/`; the 448 x 448 grid images are not committed to Git and can be
regenerated from the original photos with the scripts above.

---

## Citation

```bibtex
@article{pnrbct2026,
  title   = {PNRBCT: A Position Aware Noise Removed Binary CARE Transformer
             for Crack Classification under Limited Training Data},
  author  = {Wang, Xianglong and Ubulkasim, Mamatjan and Liu, Aoran and
             Zeng, Jin and Xu, Zhigan and Wang, Ganlong and Dong, Xinhua and
             Han, Hongmu and Yan, Zhongzheng},
  year    = {2026}
}
```

---

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
