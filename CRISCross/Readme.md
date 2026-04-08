
## Loading the Pretrained CRISCross Model & using the corresponding Datamodule for you data

This section shows how to load the pretrained **CRISCross** model from released weights. 
This is either our best performing which was pretrained using **CRISPRAT** with AlphaGenome ATAC seq epigenetic features or the
sequence only model if you dont have access to the AlphaGenome ATAC seq tracks.
Further it shows how to use our Datamodule from a pandas Dataframe in order to train on your data


### Important Notes

The model architecture must match the pretraining configuration.  
The `dropout` value does not affect loading and can be set as needed.

---

### Loading the Model

Download the pretrained weights:

AlphaGenome ATAC seq model:

Sequence only model:

```python
import torch

model = CRISCross(
    vocab_size=5,
    dropout=dropout,   # arbitrary (e.g. same as training config)
    context_layers=3,
    hidden_dim=512,
    num_epi=1, # adjust to 0 if you want to use sequence only model
    output_size=1,
    windowsize=512,
    merge="early"
)

state_dict = torch.load("criscross_pretrained_state_dict.pt", map_location="cpu")
model.load_state_dict(state_dict)

```


## Using the `GenomicDataModule`





This section describes how to initialize and use the `GenomicDataModule` for loading genomic inputs and epigenetic features.

### Overview

The `GenomicDataModule` handles:
- Sequence loading from a reference genome
- Access to epigenetic signals stored as NumPy memmaps (AlphaGenome outputs)
- Dataset splitting into train/validation/test sets
- Batch generation for training and evaluation

---


```python
dm = GenomicDataModule(
    fasta_path="GRCh38.primary_assembly.genome.fa",
    bw_dir=bw_dir,
    epi_features=["ATAC"],   # or [] if not using epigenetics
    window_size=512,
    batch_size=batch_size,
    num_workers=10,
    num_samples=batch_size * 10,
    norm_epi=True,           # must be True if epi_features is ["ATAC"]
    use_energy=False,
    mode="np",
    df=df,
    val_guides=run_settings.iloc[train_test_split]["val_set"],
    test_guides=run_settings.iloc[train_test_split]["test_set"],
)
```

---

### Summary

- Use `"np"` mode for memmap-based loading  
- Match `window_size` with the model  
- Set `epi_features` and `norm_epi` consistently:
  - `"ATAC"` → `norm_epi=True`
  - no features → `epi_features=[]`, `norm_epi=False`  
- Provide correct train/validation/test splits via `df` and guide sets  


---

### Summary

1. Initialize `CRISCross` with the correct architecture  
2. Load the provided weights  


### Required Inputs

- `fasta_path`: Path to the reference genome (e.g. GRCh38)
- `bw_dir`: List of directories containing AlphaGenome-derived memmaps (one per chromosome)
- `epi_features`: List of epigenetic features (e.g. `["ATAC"]`)
- `window_size`: Must match the model configuration (e.g. 512)
- `df`: DataFrame containing guide / sample information
- `val_guides`, `test_guides`: Splits used for validation and testing
- `mode`: Must be `"np"` when using NumPy memmaps

---

### Important Configuration Rules

- `epi_features`:
  - Use `["ATAC"]` if the model uses epigenetic input
  - Use `[]` if no epigenetic features are used

- `norm_epi`:
  - Must be `True` when using ATAC features
  - Otherwise `False`

- `windowsize`:
  - Must match the model's `windowsize` (e.g. 512)

- `mode`:
  - Always set to `"np"` when using memmaps

- `use_energy`:
  - Ignored in this setup (can remain `False`)

---

### The dataframe

## Required DataFrame (`df`) for `GenomicDataModule`

The `df` input is a pandas DataFrame that defines all samples (e.g. guides) used for training, validation, and testing in the fine-tuning setup.

Each row represents one sample.

---

## Required Columns

### 1. Genomic Coordinates

- **`chr`**  
  Chromosome name (must match FASTA and memmap chromosome names)  
  Example: `"chr1"`, `"chr2"`, `"chrX"`

- **`start`**  
  Genomic start coordinate (integer)

- **`end`**  
  Genomic end coordinate (integer)

---

### 2. Guide Information

- **`Guide_sequence`**  
  Nucleotide sequence of the guide (string)

- **`GuideID`**  
  Unique identifier used for splitting into train/validation/test sets  
  Must match entries provided in:
  - `val_guides`
  - `test_guides`

---

### 3. Labels

- **`label`**  
  Target label for supervised training  
  - Binary classification: typically `0` or `1`

---

## Optional Columns


### 4. Epigenetic Track Selection (optional)

- **`epiDir`**  
  Specifies which epigenetic directory (from `bw_dir`) to use per sample  

If present:
- Values must correspond to entries in `bw_dir`
- Internally mapped to integer indices

If absent:
- A single epigenetic directory is used for all samples

---



## Example Schema

```python
df = pd.DataFrame({
    "chr": ["chr1", "chr2"],
    "start": [100000, 200000],
    "end": [100023, 200023],
    "Guide_sequence": ["ACGT...", "TGCA..."],
    "GuideID": ["guide_1", "guide_2"],
    "label": [1, 0],
})
```

Optional additions:

```python
df["epiDir"] = ["AGTensorsCL:0000624", "AGTensorsEFO:0002067"]
```

---
## Splitting Logic

- `test_guides` → defines the test set via matching `GuideID`
- Remaining samples are split into training and validation

---

### `val_guides` Behavior

| Value            | Behavior |
|------------------|----------|
| `None`           | No validation guides; validation mask is all zeros (no split) |
| `[]` (empty list)| Random 80/20 split is applied (~20% validation) |
| Non-empty list   | Deterministic validation set via matching `GuideID` |

---

### Summary

- Use guide lists → fixed splits  
- Use `[]` → random 80/20 split  
- Use `None` → no validation split via guides  

## Summary

To work with `GenomicDataModule`, your DataFrame must include:

- `chr`
- `start`
- `end`
- `Guide_sequence`
- `GuideID`
- `label`

Optional:

- `epiDir`
- `Score_norm`

All coordinate and chromosome values must be consistent with:
- The FASTA file (`GRCh38`)
- The AlphaGenome memmaps in `bw_dir`




# CRISPRAT Training Pipeline Tutorial

This guide walks through the experiments conducted for our publication: dataset preparation, pretraining, and fine-tuning.
The packages used in this publication are listed in `environment.yml`. Note that this environment may not work on all systems, as compatibility can depend on factors such as your CUDA version and hardware configuration.

---

## 1. Dataset Preparation

Before training, you must augment the T-cell dataset and download AlphaGenome predictions.

Run:

```bash
bash batchScripts/datasetCreation.sh
```

### What this does
- Adds required columns to the T-cell dataset  
- Downloads AlphaGenome predictions  
- Creates intermediate datasets  
- Produces the final dataset  
- Generates an `AGTensors/` directory containing NumPy memmaps for efficient data loading  

### Note on Dataset Coordinates
There is a minor indexing issue in the original dataset creation code:
- It **does not affect training data**
- It only results in incorrect coordinates in the final dataset
- Downstream scripts handle this correctly
- The issue is **fixed in the CRISPRAT version** of the training scripts (recommended for future work)

---

## 2. Pretraining Configuration

Generate all required pretraining configuration files:

```bash
bash batchScripts/createAllPretrainConfigs.sh
```

---

## 3. Pretraining

### Hardware Requirements
- SLURM cluster
- ≥ 4 GPUs per node
- ~40 GB VRAM per GPU

If your hardware differs:
- Adjust **batch size** and **gradient accumulation**
- Maintain a similar *effective batch size*

### Running Without SLURM (Task Arrays)

Most training scripts are implemented as **SLURM task arrays**, where each task processes a different configuration (e.g., feature type or context size).

If you want to reproduce this setup outside of SLURM:
- You must run the same script multiple times manually
- For each run, set the environment variable:

```bash
SLURM_ARRAY_TASK_ID=<task_id>
```


Example:

```bash
SLURM_ARRAY_TASK_ID=0 bash batchScripts/run_pretrain.sh
SLURM_ARRAY_TASK_ID=1 bash batchScripts/run_pretrain.sh
SLURM_ARRAY_TASK_ID=2 bash batchScripts/run_pretrain.sh
```

- Increment `SLURM_ARRAY_TASK_ID` for each configuration 
- The valid range depends on the array size defined in the SLURM script (see `#SBATCH --array=...`)

---

### 3.1 Main Pretraining (T-Cell Dataset)

```bash
sbatch batchScripts/run_pretrain.sh
sbatch batchScripts/run_pretrain_23.sh
```

- Uses SLURM task arrays
- Covers different:
  - Epigenetic feature types
  - Context sizes

---

### 3.2 Ablation Pretraining

```bash
sbatch batchScripts/run_pretrain_ablation.sh
```

---

### 3.3 AlphaGenome Data Preparation

Before running AlphaGenome-based experiments you need to download whole Alphagenome tracks as memmaps:

```bash
sbatch batchScripts/downloadK562AG.sh
sbatch batchScripts/downloadTCELLAG.sh
```

---

### 3.4 Artificial Pretraining

```bash
sbatch batchScripts/run_artificial_pretrain.sh
sbatch batchScripts/run_artificial_pretrain_withAG.sh
```

**Important:**
- Default configs require **nodes with 2× NVIDIA H200 GPUs**

---

### Recommendation

Pretraining is the most time-consuming step.  
Run all CRISPRAT-related jobs in parallel if resources allow.

---

## 4. Fine-Tuning

### 4.1 Generate Fine-Tuning Configurations

```bash
bash batchScripts/createAllFineTuningConfigs.sh
```

---

### 4.2 Fine-Tuning on T-Cell Dataset

```bash
sbatch batchScripts/run_fine_tune_no_pretrain.sh
sbatch batchScripts/run_fine_tune_l40s.sh
sbatch batchScripts/run_fine_tune_ablation.sh
```

---

### 4.3 Comparison with Previous Pretraining Strategy

```bash
sbatch batchScripts/run_fine_tune_artificial.sh
```

---

### 4.4 Cross Cell-Type Evaluation (CRISPRAT)

```bash
sbatch batchScripts/runCRISPRATTest.sh
```

---


### 4.5 Collect All Results

All Results were written in a RUNlogs directory in form of tensorboard logging. 
Collect them using the following command

```bash
python -m util.gatherFineTuningResults
```

This will create two files one directory up in the hirachy with all the results used to create the plots.

---

## Summary Workflow

1. Prepare dataset  
2. Generate pretraining configs  
3. Run pretraining (main + ablations + artificial)  
4. Generate fine-tuning configs  
5. Run fine-tuning (standard + comparisons + cross-cell-type)  

---

## Notes

- Ensure SLURM is properly configured before submitting jobs  
- Monitor GPU memory usage when adapting configs  
- Prefer the updated CRISPRAT scripts for new experiments  