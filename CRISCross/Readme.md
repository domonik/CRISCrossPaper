
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
