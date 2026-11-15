# PNRBCT

**Official public code repository for the paper**

> **PNRBCT: A Position Aware Noise Removed Binary CARE Transformer for Crack Classification under Limited Training Data**
>
> Xianglong Wang, Mamatjan Ubulkasim, Aoran Liu, Jin Zeng, Zhigan Xu\*, Ganlong Wang, Xinhua Dong, Hongmu Han, Zhongzheng Yan
>
> \* Corresponding author: Zhigan Xu

This repository provides the reference implementation, trained checkpoints,
evaluation results, and the pavement image dataset used in the paper. If you
use this code or dataset in your research, please cite the paper
(see [Citation](#citation)).

## Method overview

PNRBCT is a three-stage pipeline for pavement crack classification:

1. **Stage 1 — Binary CARE Transformer.** Each pavement image is divided into a
   6 x 8 grid (48 cells). A CARE Transformer predicts, for every cell, whether
   it is a crack grid, producing a detection matrix `G ∈ {0,1}^{2×6×8}` whose
   two channels hold the binary label and the crack probability.
2. **Stage 2 — PNR (Position-Aware Noise Remover).** A lightweight
   convolutional model re-examines low-confidence crack cells (`p < 0.85`) and
   removes spatially inconsistent false positives ("noise grids") while keeping
   genuine cracks.
3. **Stage 3 — CDCC (Crack-Grid Distribution Classifier).** The denoised 6 x 8
   crack map is flattened into a 48-dimensional vector and classified into
   alligator, transverse, or longitudinal cracks.

The three models are trained independently and chained at inference time.

## Repository layout

```
PNRBCT/
├── main.py                       # Pipeline dispatcher
├── tools/                        # Data preparation and verification helpers
├── CARETR_Project/               # Reference implementation
│   ├── models/                   # CARE Transformer, PNR, CDCC
│   ├── stage1_care_transformer/  # Stage-1 training and evaluation
│   ├── stage2_pnr/               # PNR cache, dataset, and training
│   ├── stage3_cdcc/              # CDCC data, training, and end-to-end test
│   ├── checkpoints/              # Trained weights
│   ├── results/                  # Reported evaluation outputs
│   ├── docs/EXPERIMENTS.md       # Full experimental protocol
│   ├── data/
│   │   ├── split_v2_manifest.json
│   │   ├── annotations/          # Grid-level annotation JSONs
│   │   └── cache/                # Stage-1 G cache and PNR dataset
│   ├── figures/
│   └── README.md
└── dataset/                      # Created after downloading the release assets
    ├── grid/
    ├── horizontal/
    └── vertical/
```

## Dataset

The raw pavement images (**~8.5 GB**, 995 JPG files) are distributed as GitHub
Release assets, because GitHub strongly recommends keeping a Git repository
below 5 GB. The 845 grid-level annotation JSON files are included in the
release and are also committed under `CARETR_Project/data/annotations/`.

| Class | Annotated images | End-to-end test images |
|---|---:|---:|
| Alligator (`grid`) | 238 | 50 |
| Transverse (`horizontal`) | 375 | 50 |
| Longitudinal (`vertical`) | 232 | 50 |
| **Total** | **845** | **150** |

The image-level split is stratified by crack category with a fixed seed (42):
676 training / 85 validation / 84 test images. The 150 end-to-end test images
are disjoint from the 845 annotated images.

### Download and reassemble

1. Download all `dataset_*.zip.part-*` files and `SHA256SUMS.txt` from the
   [Releases page](https://github.com/LINKS-MMTJ/PNRBCT/releases).
2. Place them in the repository root and verify the parts:

   ```bash
   sha256sum -c SHA256SUMS.txt
   ```

3. Concatenate the parts and extract the archives:

   ```bash
   cat dataset_grid.zip.part-*        > dataset_grid.zip
   cat dataset_horizontal.zip.part-*  > dataset_horizontal.zip
   cat dataset_vertical.zip.part-*    > dataset_vertical.zip

   unzip dataset_grid.zip
   unzip dataset_horizontal.zip
   unzip dataset_vertical.zip
   ```

   On Windows, run these commands in Git Bash, or concatenate the parts with
   PowerShell/7-Zip before extracting.

The archives extract to `dataset/{grid,horizontal,vertical}/`, each containing
`annotated/` (annotated images), `e2e_test/` (end-to-end test images), and the
matching `{class}/` annotation JSON directory.

The dataset is released for research and reproducibility purposes. Please cite
the paper if you use it.

## Installation

```bash
pip install -r CARETR_Project/requirements.txt
```

The code was developed and evaluated with PyTorch 2.x and timm 1.x. All
experiments reported in the paper were run on a single NVIDIA GeForce RTX 5090
(32 GB).

## Reproduction

All commands below are run from the repository root.

```bash
# 1. Build the image-level split and the 448 x 448 grids from dataset/
python tools/resplit_image_level.py --data-root dataset --grid-size 448 --materialize
python tools/gen_e2e_grids.py --data-root dataset --grid-size 448

# 2. Run the full pipeline (or a single stage: stage1-train, stage1-eval,
#    stage2, stage3, e2e)
python main.py all
```

Use `python main.py --help` for all stage and runtime options. See
[CARETR_Project/README.md](CARETR_Project/README.md) for stage-level commands
and [CARETR_Project/docs/EXPERIMENTS.md](CARETR_Project/docs/EXPERIMENTS.md) for
the complete experimental protocol.

## Main results

| Experiment | Metric | Value |
|---|---|---|
| Stage 1: grid-level detection (1:1 test set) | Accuracy / Macro F1 | 94.12% / 94.12% |
| Stage 2: PNR noise-grid detection | F1 | 0.4153 |
| Stage 2: true-crack mis-removal | Rate | 3.81% (34 / 893) |
| Stage 3: end-to-end, 150 test images | Accuracy | 96.67% |
| Stage 3: end-to-end, 150 test images | Macro F1 | 96.66% |
| PNR ablation (150 test images) | Macro F1 without / with PNR | 94.02% / **96.66%** |
| CDCC vs. position-agnostic MLP (136 samples) | Accuracy / Macro F1 (CDCC) | 95.59% / 95.09% |
| CDCC vs. position-agnostic MLP (136 samples) | Accuracy / Macro F1 (MLP) | 67.65% / 61.44% |

Per-class end-to-end accuracy with PNR: alligator 98.0%, transverse 100.0%,
longitudinal 92.0%.

## Checkpoints

| File | Description |
|---|---|
| `CARETR_Project/checkpoints/caret_s0_448_bal_best.pth` | Stage-1 CARE Transformer |
| `CARETR_Project/checkpoints/best_pnr.pth` | PNR |
| `CARETR_Project/checkpoints/best_cdcc_pnr.pth` | CDCC trained on PNR-cleaned grids |
| `CARETR_Project/checkpoints/best_cdcc_nopnr.pth` | CDCC trained without PNR |
| `CARETR_Project/checkpoints/best_agnostic_pnr_agnostic.pth` | Position-agnostic MLP baseline |

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

## License

The source code is released under the Apache License 2.0; see
[LICENSE](LICENSE). The dataset is released for research and reproducibility
purposes.

## Contact

Corresponding author: Zhigan Xu.
