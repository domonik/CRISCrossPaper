import json
import itertools
from src.pretrain import short_hash
import os
from src.Datasets import EPI_FEATURES


parent_dir = "PretrainingPaperFixed"
# Base/default parameters (copied from your idx is None branch)
base_params = {
    "batch_size": 256,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.3,
    "lr": 1e-4,
    "patience": 20,
    "experiment": "FineTuningPaperLeaveOneOutFixed",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "run_settings": "runSettings/RunSettingsLeaveOneOut.tsv",
    "dataset": "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv",
}

# Full feature list
all_features = [
    "EX_ATAC",
    "EX_H3K4me1",
    "EX_H3K4me3",
    #"EX_H3K9ac",      # excluded from some sets
    "EX_H3K9me3",
    "EX_H3K27ac",
    "EX_H3K27me3",
    "EX_H3K36me3",
]

# Feature groups
ex_features_no_h3k9ac = [
    f for f in all_features
    if f.startswith("EX_") and f != "EX_H3K9ac"
]

ag_features = [
    f[3:] for f in all_features
]

seeds = [i * 42 for i in range(10)]

splits = list(range(17))

feature_sets = [
    ag_features,
    ex_features_no_h3k9ac,
    #all_features,
    #missing_ag,
    [],  # empty list case
    #["CHIP_HISTONE"]

]

windowsizes = [23, 32, 128, 512]
runs = [i * 42 for i in range(0, 1)]

configs_l40s = []
configs_h200 = []

# for split in splits:
#     for run in runs:
#         params = dict(
#                 windowsize=23,
#                 epi_features=[],
#                 seed=seeds,
#                 split = split,                    
#                 **base_params
#                 )
#         params["experiment"] = f"{base_params["experiment"]}/NoPretrain/{run}"
#         params["merge"] = None
#         configs_l40s.append(params)
exindices = []
for idx, (windowsize, epi_features, split, run) in enumerate(itertools.product(windowsizes, feature_sets, splits, runs)):
    params = base_params.copy()
    params["windowsize"] = windowsize
    params["epi_features"] = epi_features
    params["seed"] = seeds
    params["split"] = split
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)
    epi_hash = short_hash(epi_features, num_epi)

    if len(epi_features) == 0:
            params["merge"] = None
            params["experiment"] = f"{base_params["experiment"]}/None/{run}"
            mode = "None"
    else:
        # Decide based on feature types
        if all(f.startswith("EX_") for f in epi_features):
            params["experiment"] = f"{base_params["experiment"]}/EX/{run}"
            mode = "EX"
            exindices.append(idx)
        elif len(epi_features) == 2 and "DNASE" in epi_features:
            params["experiment"] = f"{base_params["experiment"]}/MissingAG/{run}"
            mode="MissingAG"
        elif len(epi_features) == 1:
            params["experiment"] = f"{base_params["experiment"]}/AGNoATAC/{run}"
            mode = "AGNoATAC"
        elif all(not f.startswith("EX_") for f in epi_features):
            params["experiment"] = f"{base_params["experiment"]}/AG/{run}"
            mode = "AG"
        else:
            # mixed set, you can choose a name or fallback
            params["experiment"] = f"{base_params["experiment"]}/Mixed/{run}"
            print(windowsize)
            mode ="Mixed"


    base_dir = f"RUNlogs/{parent_dir}/{mode}/test_split0/ctl{params['context_layers']}_bs512_ws{params["windowsize"]}_ue{num_epi}_seed{run}_hash{epi_hash}" 
    chkpt = os.path.join(base_dir, "run_", "vv0", "checkpoints", "last.ckpt")
    assert os.path.exists(chkpt), f"Checkpoint not found:\n{chkpt}"
    params["chkpt"] = chkpt



    if windowsize >= 512 and params["context_layers"] > 3:
        params["accumulate_grad_batches"] = 2
        params["batch_size"] = 128
    
    configs_l40s.append(params)

print(",".join(map(str, exindices)))
# Write to JSON
with open("configs/run_configs_l40s.json", "w") as f:
    json.dump(configs_l40s, f, indent=2)

with open("configs/run_configs_h200.json", "w") as f:
    json.dump(configs_h200, f, indent=2)

print(
    f"Wrote {len(configs_l40s)} configs to run_configs_l40s.json and "
    f"{len(configs_h200)} configs to run_configs_h200.json"
)