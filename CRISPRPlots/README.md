# CRISPRPlots

Visualization scripts for generating publication figures from the CRISCross benchmark.
All figure-generation scripts produce both interactive HTML and publication-ready SVG outputs.
Input data lives in the sibling `../Results/` directory (produced by the CRISCross evaluation pipeline).

## Directory Layout

| Path | Description |
|------|-------------|
| `scripts/` | All plotting, data-gathering, and helper scripts |
| `Figures/` | Generated SVG/HTML figures (gitignored) |
| `Tables/` | Generated summary tables (gitignored) |
| `GenomicSummary/` | Pre-computed `.npy` arrays for distribution plots |
| `shap/sg*/` | SHAP pickle data per guide group |
| `WorkflowFigures/` | Schematic diagrams (created externally) |
| `SummaryStatsPerWinsize.tsv` | Per-guide, per-feature statistics across window sizes |
| `OldModlsfull.tsv` | Merged old-model results |

## Scripts

Scripts are listed in dependency order — run data preparation first, then figure generation.

### 1. Data Preparation (produce input files for figure scripts)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `gatherDistributions.py` | Compute summary statistics from raw genomic tensors | `../CRISCross/AGTensors3/` memmaps | `GenomicSummary/*.npy`, `SummaryStatsPerWinsize.tsv` |
| `gatherOldModelResults.py` | Parse old model benchmark text output | `.txt` benchmark logs | `.tsv` parsed results |
| `gatherTabPFNResults.py` | Parse TabPFN benchmark text output | `Tabpfn_results.txt` | `TabPFNResults.tsv` |
| `joinAllResults.py` | Merge all result sources into one file | Tensorboard CSVs, per-model CSVs | `../Results/FinalSummary.tsv` |

### 2. Shared Modules (imported by figure scripts)

| Script | Purpose |
|--------|---------|
| `helpers.py` | Significance brackets, star annotations, Wilcoxon + Bonferroni testing |
| `plotly_template.py` | Custom Plotly template, named colors, width constants |
| `plotly_shap_plotter.py` | Reusable `PlotlySHAPVisualizer` class (beeswarm, bar, force, dependence plots) |

### 3. Figure Generation — Genomic Distributions & Significance

(These depend on outputs from `gatherDistributions.py`.)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `plotSummaryFeatureDistr.py` | Feature distribution histograms + Mann-Whitney U test significance | `SummaryStatsPerWinsize.tsv` | `Figures/SummaryDistribution.{html,svg}`, `Tables/FeatureMWUTable.tsv` |
| `plotRankBiserial.py` | Violin plot of rank-biserial effect sizes for significant features | `Tables/FeatureMWUTable.tsv` ← from `plotSummaryFeatureDistr.py` | `Figures/RankBiserialComparison.{html,svg}` |
| `plotDistributions.py` | Genomic signal distributions around guide positions (EX vs AlphaGenome) | `GenomicSummary/*.npy` ← from `gatherDistributions.py` | `Figures/DistributionRes.{svg,html}`, zoomed insets |

### 4. Figure Generation — Model Comparison Figures

(These depend on `../Results/FinalSummary.tsv` from `joinAllResults.py`.)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `plotCrossAttnResults_v3.py` | Multi-panel bar charts: pretraining comparison, ablation, context-size | `../Results/FinalSummary.tsv` | `Figures/CrossAttn_Fig1.{html,svg}`, `CrossAttn_Fig2.{html,svg}` |
| `plotCRISPRAT.py` | CRISPRAT pretraining strategy comparison (T-Cell, K562, cross-cell-type) | `../Results/FinalSummary.tsv` | `Figures/CRISPRAT.{html,svg}` |
| `plotCrossCelltypeoldModels.py` | Cross-cell-type performance: old models vs CRISCross | `../Results/combined_cross_cell_results_3cell.csv`, `FinalSummary.tsv` | `Figures/CrossCellTypeOldModels.{html,svg}` |
| `plotOldModelResults.py` | Older models (CRISPR-IP, CnnCrispr, CRISPR-OFFT, CRISPert) across feature types | `../Results/ex_vs_base.csv`, `AG_vs_base.csv` | `Figures/OtherModels.{html,svg}` |
| `plotTabPFNResults.py` | TabPFN feature-set comparison (Baseline vs Experimental vs AlphaGenome) | `../Results/TabPFNResults.tsv` | `Figures/TabPFNResults.{html,svg}` |
| `modelSplitPerformance.py` | Prediction stability (AUC-PR std dev) across CV splits | `OldModlsfull.tsv`, `TabPFNResults.tsv` | `Figures/PredictionStdev.{html,svg}` |

### 5. Figure Generation — SHAP & Misc

(These use independent data sources.)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `plotShap.py` | SHAP beeswarm summary plots (top features + full supplement) | `shap/sg*/shap_{positive,negative}.pkl` | `Figures/shapVals.{svg,png,html}`, `Tables/SHAPSummary.tsv` |
| `shap/plot_shap_pos_neg.py` | Matplotlib-based SHAP bar/summary plots (separate pipeline from Plotly) | External SHAP pickle files | `shap/GLOBAL_shap_*.png` |
| `plotAverageValidationCurve.py` | Averaged validation AUC-PR over training steps | `validation_curves.tsv` | `Figures/ValidationCurve.{svg,html}` |
| `plotProbDistr.py` | Conditional binomial probability of mismatch counts in 23bp guides | (computed) | `Figures/BarChartDistribution.svg` |
| `plotCrossCellType.py` | Experimental cross-cell-type script (work in progress) | `FinalSummary.tsv` | (interactive display only) |

## Data Dependencies

The scripts depend on data produced by other tools in the CRISCross project:

- **`../Results/`** — Primary data source. Contains model benchmark results, tensorboard summaries, and merged files.
- **`../Results/FinalSummary.tsv`** — Central merged results file (created by `joinAllResults.py`). Used by most figure-generation scripts.
- **`GenomicSummary/`** — Pre-computed `.npy` arrays with per-position statistics (created by `gatherDistributions.py`).
- **`shap/sg*/`** — Pickled SHAP Explanation objects per guide group.
- **`../CRISCross/AGTensors3/`** — Raw genomic tensor memmap files (ATAC, H3K4me3, etc.).
- **`../datasets/`** — Dataset files (e.g., K562 DeepCRISPR with coordinates).

## Usage

Run scripts from the CRISPRPlots root directory:

```bash
python scripts/plotCrossAttnResults_v3.py
python scripts/plotShap.py
```

Outputs are written to `Figures/` (SVG + HTML) and `Tables/` (TSV).
Statistical comparisons use Wilcoxon signed-rank tests with Bonferroni correction.
