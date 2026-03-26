


## Setup

First it is necessary to add extra columns to the T-Cell dataset and download Alphagenome predictions for it. Therefore we will write several temporary datasets and a final one by executing the following bash script:

```console
bash batchScripts/datasetCreation.sh
```

This will also create an AGTensors directory containing numpy memmaps for efficient loading of the dataset

Note: Unfortunately the code that was used to run these experiments contains a minor indexing error which however has no impact on the used training data. The only thing that differes is that the final Dataset has wrong coordinates which are handled correctly by the following scripts. This issue is also fixed in the later used CRISPRAT version of the training scrips which we recommand to use for any further investigations.

Once this is accomplished we will create all the necessary pretrain configs via:


```console
bash batchScripts/createAllPretrainConfigs.sh
```


## Pretraining 

Note that the following steps require access to a SLURM cluster with at least 4 GPUs per node, each with approximately 40 GB of VRAM. If you do not have access to such a cluster, you will need to adjust the configuration files—particularly the gradient accumulation and batch size values—to achieve an effective batch size similar to the one used in our experiments. 

In order to run pretraining on the T-Cell dataset itself you need to run the following SLURM scripts.
These contain task arrays corresponding to each epigenetic feature type and context size.

```console
sbatch batchScripts/run_pretrain.sh
sbatch batchScripts/run_pretrain_23.sh
```

Next step will be running pretraining for our Ablation results:

```console
sbatch batchScripts/run_pretrain_ablation.sh
```

Since pretraining takes the majority of time it is recommended to run all CRISPRAT tasks in parallel.

Note that those however require nodes with two Nvidia-H200 GPUs attached when run using the default configs.
Before starting the job that uses AlphaGenome download the corresponding tracks

```console
sbatch batchScripts/downloadK562AG.sh
sbatch batchScripts/downloadTCELLAG.sh
```

```console
sbatch batchScripts/run_artificial_pretrain.sh
sbatch batchScripts/run_artificial_pretrain_withAG.sh
```


## Fine Tuning

first we create all fine tuning configs



```console
bash batchScripts/createAllFineTuningConfigs.sh
```


Then we run fine tuning for the T-cell dataset

```console
sbatch batchScripts/run_fine_tune_no_pretrain.sh
sbatch batchScripts/run_fine_tune_l40s.sh
sbatch batchScripts/run_fine_tune_ablation.sh
```

Afterwards we fine tune the comparison of CRISPRAT to the previous pretraining strategy. 

```console
sbatch batchScripts/run_fine_tune_artificial.sh
sbatch batchScripts/run_fine_tune_l40s.sh
```