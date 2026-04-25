# CRISCross — Reproducibility Repository

> **This repository is a reproducibility archive for our publication.** It contains the exact, unmodified scripts, configurations, and intermediate results as they were run. No cleanup or refactoring has been applied.

> **Looking for the model architecture or a clean codebase?** Visit the canonical repository at **[github.com/domonik/CRISCross](https://github.com/domonik/CRISCross)**. That repo contains the proper model implementation, documentation, and is maintained for future development.

## What is CRISCross?

CRISCross is a deep learning framework for predicting CRISPR guide RNA off targets. It integrates genomic and epigenomic features (ATAC-seq, histone marks, AlphaGenome predictions) with sequence information to improve on-target activity predictions across cell types.

This repository enables full reproduction of every figure, table, and benchmark result reported in our publication.

## Directory Layout

| Path | Description |
|------|-------------|
| `CRISCross/` | Training pipeline — pretraining, fine-tuning, batch scripts, model source |
| `CRISPRPlots/` | Visualization scripts — figure generation, SHAP plots, statistical comparisons |
| `Results/` | Benchmark output — merged TSV/CSV files used by plotting scripts generated via CRISCross subdir |
| `datasets/` | Dataset files — guide coordinates, cell-type mappings |

## Reproducing the Results

The pipeline has two stages with a clear data dependency:

1. **Training** (`CRISCross/`) — Run pretraining and fine-tuning to produce benchmark results. See [`CRISCross/Readme.md`](CRISCross/Readme.md) for the full tutorial covering dataset preparation, pretraining, ablation studies, and fine-tuning.

2. **Figures** (`CRISPRPlots/`) — Generate all publication figures from the benchmark results. See [`CRISPRPlots/README.md`](CRISPRPlots/README.md) for the scripts, dependency order, and output descriptions.

The data flow: `CRISCross/` training → `Results/` → `CRISPRPlots/` figure scripts → `CRISPRPlots/Figures/`

## Environment

The Python environment used for this publication is specified in `CRISCross/environment.yml` and `CRISPRPlots/environment.yml`. Note that compatibility depends on your CUDA version and GPU hardware — it may require adjustment for your system.

## Citation

[Publication citation — to be added]
