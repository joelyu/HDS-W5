# Comparing Feature Extraction Methods for Peripheral Blood Cell Classification

## Healthcare Data Science Module 5 Assignment

To run just enough to render the Quarto report, go [here](#just-enough-to-render-quarto-document).

## Abstract

**PURPOSE** Automated peripheral blood cell classification supports haematological diagnosis, but the relative merits of pretrained deep learning backbones, handcrafted features and end-to-end fine-tuning remain unclear, particularly on generalisability and computational cost.

**METHODS** The three approaches were compared and trained on the [KU-Optofil dataset](https://zenodo.org/records/17333317) (31,484 images, 13 classes, 276 patients): pretrained deep learning (DinoBloom-S, ResNet-50, EfficientNet-B0, ViT-S/16) with XGBoost and linear probe classifiers; 65 handcrafted morphological features using convex hull or CellPose segmentation; end-to-end fine-tuning of ResNet-50 and EfficientNet-B0. Models were externally validated on the [Acevedo dataset](https://data.mendeley.com/datasets/snkd93bnjr/1) (17,092 images, 6 shared classes). SHAP values and attention maps provided interpretability.

**RESULTS** DinoBloom-S with XGBoost run on consumer hardware achieved the best performance among feature extraction approaches (macro F1 0.710 ± 0.007), comparable to fine-tuned models (macro F1 ~0.720) that required 40X more compute time on A100 GPUs. DinoBloom-S significantly outperformed ViT-S/16 (bootstrap Δ macro F1 95% CI excludes 0), highlighting the importance of domain-specific pretraining. On external validation, CellPose-based handcrafted features outperformed convex hull by 0.16 macro F1, thanks to better cell segmentation.

**CONCLUSION** Domain-specific self-supervised pretraining (DinoBloom-S) paired with a lightweight classifier provides near fine-tuned performance at a fraction of the computational cost. Segmentation quality plays a role for handcrafted feature generalisability.

## Datasets

**KU-Optofil PBC (primary)** — Yarıkan et al., *Scientific Data* 2026.
31,484 single-cell images, 13 classes, 276 patients, with patient-level metadata enabling leakage-free train/validation/test splits.

- **Download:** [Zenodo record (v2) 17333317](https://zenodo.org/records/17333317) · DOI [10.5281/zenodo.16542374](https://doi.org/10.5281/zenodo.16542374)
- Or run `python scripts/00_download_data.py` 

**Acevedo PBC (external validation)** — Acevedo et al., *Data in Brief* 2020.
~17,092 images, 8 classes, CellaVision DM96, Hospital Clínic de Barcelona. Mapped to the 6 classes shared with KU-Optofil for cross-dataset evaluation.

- **Download:** [Mendeley Data snkd93bnjr (v1)](https://data.mendeley.com/datasets/snkd93bnjr/1) · DOI [10.17632/snkd93bnjr.1](https://doi.org/10.17632/snkd93bnjr.1)
- Or run `python scripts/00b_download_acevedo.py` 


## Setup

### Dependencies

The project runs on **Python 3.12**, with `conda` or `pip` for building the environment.

```shell
pip install -r requirements.txt         # any OS/architecture

# OR

conda env create -f environment.yml     # macOS arm64-based
conda activate classileukotion
```

Key packages are:

| Package | Version | Used for |
|---|---|---|
| numpy | 2.4.4 | arrays / numerics (throughout) |
| pandas | 3.0.3 | metadata, tables, results I/O |
| scipy | 1.17.1 | convex hull, interpolation, stats |
| scikit-learn | 1.8.0 | linear probe, metrics, label encoding, PCA/KMeans |
| statsmodels | 0.14.6 | McNemar's test |
| xgboost | 3.2.0 | gradient-boosted classifier |
| optuna | 4.8.0 | hyperparameter tuning (TPE) |
| torch | 2.5.1 | backbone inference + fine-tuning |
| torchvision | 0.20.1 | ResNet-50 / EfficientNet-B0 + image transforms |
| timm | 1.0.27 | ViT-S/16 backbone |
| huggingface-hub | 1.15.0 | DinoBloom-S weight download |
| pillow | 12.2.0 | image loading |
| opencv-python | 4.13.0.92 | colour-space conversion, contours (handcrafted features) |
| scikit-image | 0.26.0 | nucleus segmentation, GLCM texture |
| cellpose | 4.1.1 | whole-cell segmentation (Cellpose-SAM) |
| matplotlib | 3.10.9 | all figures |
| umap-learn | 0.5.12 | UMAP feature embeddings (explainability) |
| shap | 0.51.0 | SHAP feature attributions (explainability) |
| openpyxl | 3.1.5 | xlsx comparison workbook (pandas Excel engine) |
| pytest | 9.0.3 | unit tests |


For CUDA GPU support (e.g. fine-tuning on an HPC), install the matching PyTorch build:
```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Pipeline 

#### Quick start

##### Just enough to render Quarto document

JSON results are git tracked, only extracted features in `*.npz` files are needed.

```shell
python scripts/00_download_data.py
python scripts/02_feature_extraction.py
```

##### Run everything from start to finish

One shell script to download all the data, train models, evaluate models, could take >1 day on consumer hardware.

```shell
./run_all.sh
```

#### Step by step breakdown

**1. Data**
```shell
python scripts/00_download_data.py            # KU-Optofil → data/raw/
python scripts/00b_download_acevedo.py        # Acevedo  → data/acevedo/   (external validation)
python scripts/01_data_exploration.py         # EDA plots + summary
```

**2. Feature extraction**
```shell
# Frozen backbone features (all five backbones)
python scripts/02_feature_extraction.py

# Handcrafted: first build the cell masks, then extract features under each segmentation
python scripts/02d_cellpose_masks.py                                   # CellPose masks (one-time, cached, resumable)
python scripts/02b_handcrafted_features.py                             # → handcrafted_features.npz          (convex hull)
python scripts/02b_handcrafted_features.py --segmentation cellpose --force   # → handcrafted_cellpose_features.npz (CellPose)
```

**3. Lightweight classifiers**
```shell
# All backbones + handcrafted variants, full feature set
python scripts/03_xgboost_training.py
python scripts/03b_linear_probe.py

# Handcrafted 2×2 ablation: Tavakoli-51 subset (vs the default full 65)
python scripts/03_xgboost_training.py  --backbone handcrafted          --feature-set tavakoli
python scripts/03_xgboost_training.py  --backbone handcrafted_cellpose --feature-set tavakoli
python scripts/03b_linear_probe.py     --backbone handcrafted          --feature-set tavakoli
python scripts/03b_linear_probe.py     --backbone handcrafted_cellpose --feature-set tavakoli
```

- `--five-class` collapses the 13 classes to the standard 5-class WBC differential for any of the above.
- The 51 feature set from Tavakoli was not used for the final report, but kept here for reference.

**4. End-to-end fine-tuning** 
Best done on [HPC](#hpc) or with hardware acceleration.

```shell
python scripts/04_fine_tune.py
```

**5. Evaluation, interpretation and external validation**
```shell
python scripts/05_evaluation.py               # comparison tables + McNemar/bootstrap significance
python scripts/06_explainability.py           # UMAP, SHAP, DinoBloom attention

# External validation on Acevedo (deep + handcrafted-convex-hull + linear run without masks;
# the handcrafted-CellPose arm needs the Acevedo masks below)
python scripts/02d_cellpose_masks.py --acevedo-dir data/acevedo/PBC_dataset_normal_DIB
python scripts/07_external_validation.py
```

## HPC

Fine-tuning (and, if preferred, the CPU jobs) can be offloaded to the CSD3 cluster. The `hpc/` folder contains a setup script and SLURM job files; per-backbone jobs can be submitted in parallel with `--backbone <name>`. Indicative resources:

| SLURM script | Job | Partition | Resources |
|---|-----|-----------|-----------|
| `01_feature_extraction` | Frozen backbone features | ampere (A100) | 1 GPU, 8 CPU, 32 GB |
| `01c_cellpose_masks` | CellPose whole-cell masks | ampere (A100) | 1 GPU, 8 CPU, 32 GB |
| `01b_handcrafted_features` | Handcrafted features (both seg.) | icelake | 32 CPU, 64 GB |
| `02_xgboost_training` | XGBoost + Optuna TPE | icelake | 32 CPU, 64 GB |
| `02b_linear_probe` | Linear probe | icelake | 8 CPU, 16 GB |
| `03_fine_tuning` | End-to-end fine-tuning (per backbone) | ampere (A100) | 1 GPU, 8 CPU, 32 GB |
| `04_external_validation` | Acevedo cross-dataset validation | ampere (A100) | 1 GPU, 8 CPU, 32 GB |
| `05_evaluation` | Comparison tables + significance tests | icelake | 8 CPU, 16 GB |
| `06_explainability` | UMAP, SHAP, attention maps | ampere (A100) | 1 GPU, 8 CPU, 32 GB |

> [!NOTE]
> Running `03_fine_tuning.py` on ResNet-50 and EfficientNet-B0 took ~10 hours (~6-7 hours for tuning, ~3 hours for validation), ViT-S/16 and DinoBloom-S failed to complete in the maximum wall time of 12 hours on the HPC.

The full pipeline can be submitted with dependency chaining via `bash hpc/submit_all.sh` (supports `--with-external`, `--with-finetune`, `--dry-run`).

## Repository structure

```
HDS_W5_cyy36/
├── scripts/
│   ├── config.py                       # Shared constants, class maps, data loaders
│   ├── 00_download_data.py             # Download + extract KU-Optofil from Zenodo (md5-checked)
│   ├── 00b_download_acevedo.py         # Download + extract Acevedo from Mendeley (sha256-checked)
│   ├── 01_data_exploration.py          # EDA: class distribution, sample grid, split stats
│   ├── 02_feature_extraction.py        # Frozen backbone features (ResNet-50, EfficientNet-B0, ViT-S/16, DinoBloom-S, DinoBloom-S multi-level)
│   ├── 02b_handcrafted_features.py     # Handcrafted features (Tavakoli-51 + extensions); --segmentation {convex_hull,dinobloom,cellpose}; dinobloom option kept as reference only
│   ├── 02c_dinobloom_cell_scores.py    # DinoBloom patch-token cellness maps (for --segmentation dinobloom; exploratory, kept as reference only)
│   ├── 02d_cellpose_masks.py           # CellPose whole-cell masks (for --segmentation cellpose; --acevedo-dir for external set)
│   ├── 03_xgboost_training.py          # XGBoost + Optuna TPE tuning; --feature-set {all,tavakoli}, --five-class
│   ├── 03b_linear_probe.py             # Linear probe (LogisticRegression); same flags
│   ├── 04_fine_tune.py                 # Optional: end-to-end fine-tuning (Optuna, AdamW, cosine LR, Hyperband pruning)
│   ├── 05_evaluation.py                # Comparison tables (xlsx), per-class F1, confusion matrices, clinical focus, McNemar + bootstrap significance
│   ├── 06_explainability.py            # UMAP, SHAP (handcrafted beeswarms with named features), DinoBloom attention
│   ├── 07_external_validation.py       # Cross-dataset validation on Acevedo (deep + handcrafted, XGBoost + linear)
│   ├── features.py                     # Handcrafted feature computation (GLCM, morphology, extract_cell_features)
│   ├── segmentation.py                 # Nucleus + cell-boundary segmentation (multi-Otsu, convex hull, CellPose, DinoBloom)
│   └── stats.py                        # McNemar's test, bootstrap CIs, Holm correction
├── tests/                              # pytest unit tests 
├── report/                             # Report files: .qmd, .bib, word and .pdf
├── hpc/                                # CSD3 SLURM scripts (optional)
│   ├── setup_csd3.sh                   # One-time environment + directory setup
│   ├── submit_all.sh                   # Submit full pipeline with dependency chaining
│   ├── 00_debug_validate.slurm         # Quick debug run (DinoBloom only, 5 trials)
│   ├── 01_feature_extraction.slurm     # Frozen backbone features (GPU)
│   ├── 01b_handcrafted_features.slurm  # Handcrafted features: convex hull + CellPose (CPU)
│   ├── 01c_cellpose_masks.slurm        # CellPose whole-cell masks (GPU); --acevedo for external set
│   ├── 02_xgboost_training.slurm       # XGBoost + Optuna TPE tuning (CPU)
│   ├── 02b_linear_probe.slurm          # Linear probe (CPU)
│   ├── 03_fine_tuning.slurm            # End-to-end fine-tuning (GPU, per-backbone)
│   ├── 04_external_validation.slurm    # External validation on Acevedo (GPU)
│   ├── 05_evaluation.slurm             # Comparison tables + statistical tests (CPU)
│   └── 06_explainability.slurm         # UMAP, SHAP, attention maps (GPU)
├── run_all.sh                          # Full local pipeline (--dry-run, --with-external, --with-finetune)
├── requirements.txt                    # Pip dependencies (OS/architecture agnostic)
├── environment.yml                     # Conda environment specification (for macOS-arm64)
├── .gitignore
└── README.md
```
