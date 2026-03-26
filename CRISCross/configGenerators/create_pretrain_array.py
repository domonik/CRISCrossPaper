import json
import itertools
import os
from src.pretrain import short_hash
from src.Datasets import EPI_FEATURES

# Base/default parameters (copied from your idx is None branch)
base_params = {
    "batch_size": 512,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.3,
    "lr": 1e-4,
    "patience": 50000,
    "seed": 0,
    "split": 0,
    "experiment": "PretrainingPaperFixed",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "accumulate_grad_batches": 1,
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

missing_ag = ["RNA_SEQ", "DNASE"]

feature_sets = [
    ex_features_no_h3k9ac,
    ag_features,
    #all_features,
    #missing_ag,
    [],  # empty list case
    #["CHIP_HISTONE"],
    #["ATAC"]
]

windowsizes = [512, 128, 23]
seeds = [0]
loop = list(range(0, 1))
configs = []
for iteration, seed, windowsize, epi_features in itertools.product(loop, seeds, windowsizes, feature_sets):
    params = base_params.copy()
    params["windowsize"] = windowsize
    params["epi_features"] = epi_features
    params["seed"] = seed
    if windowsize <= 32:
        params["accumulate_grad_batches"] = 4
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)


    epi_hash = short_hash(epi_features, num_epi)

    # Special rule: empty epi_features → merge = None
    if len(epi_features) == 0:
        params["merge"] = None
        params["experiment"] = f"{base_params["experiment"]}/None"
    else:
        # Decide based on feature types
        if all(f.startswith("EX_") for f in epi_features):
            params["experiment"] = f"{base_params["experiment"]}/EX"
        elif len(epi_features) == 2 and "DNASE" in epi_features:
            params["experiment"] = f"{base_params["experiment"]}/MissingAG"
        elif len(epi_features) == 1 and "ATAC" in epi_features:
            params["experiment"] = f"{base_params["experiment"]}/AGNoCHIP"
            if windowsize != 512:
                # Dont run this we only compare on 512
                continue
        elif len(epi_features) == 1:
            params["experiment"] = f"{base_params["experiment"]}/AGNoATAC"
            if windowsize != 512:
                # Dont run this we only compare on 512
                continue
        elif all(not f.startswith("EX_") for f in epi_features):
            params["experiment"] = f"{base_params["experiment"]}/AG"

        else:
            # mixed set, you can choose a name or fallback
            params["experiment"] = f"{base_params["experiment"]}/Mixed"
    if iteration != 0:
        config = params
        base_dir = f"RUNlogs/{params['experiment']}/test_split{params['split']}/ctl{params['context_layers']}_bs{params['batch_size']}_ws{params["windowsize"]}_ue{num_epi}_seed{params["seed"]}_hash{epi_hash}" 
        chkpt = os.path.join(base_dir, "run_", f"vv{iteration-1}", "checkpoints", "last.ckpt")
        print(os.path.exists(chkpt), chkpt)
        params["chkpt"] = chkpt
    configs.append(params)

# Write to JSON
with open("configs/pretrain_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configurations to pretrain_configs.json")
