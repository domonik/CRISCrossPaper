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
    "experiment": "PretrainingShuffled",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "accumulate_grad_batches": 1,
}

FEATURES = {
    "ATAC": "ATAC",
    "H3K27ac": "H3K27ac",
    "H3K27me3": "H3K27me3",
    "H3K36me3": "H3K36me3",
    "H3K4me1": "H3K4me1",
    "H3K4me3": "H3K4me3",
    "H3K9me3": "H3K9me3",
}

RNA_KEYS = {"RNA_PLUS", "RNA_MINUS"}
HISTONE_KEYS = {key for key in FEATURES if key.startswith("H3K")}

feature_sets = {}

#feature_sets["full"] = list(FEATURES.values())

# Ablate RNA-seq (both strands together)
feature_sets["ATAC"] = [
    FEATURES[k] for k in FEATURES if k not in HISTONE_KEYS
]

# Full model

windowsizes = [23]
seeds = [0]
loop = list(range(0, 1))
configs = []
for epiDir in ["ShuffledAGTensors", "AGTensorsWrongCellType"]:

    for iteration, seed, windowsize, feature_key in itertools.product(loop,seeds, windowsizes, feature_sets.keys()):
        params = base_params.copy()
        params["windowsize"] = windowsize
        epi_features = feature_sets[feature_key]
        params["epi_features"] = epi_features
        params["seed"] = seed
        params["epi_dir"] = epiDir
        #if windowsize == 23:
        #    params["accumulate_grad_batches"] = 4
        print(epi_features)
        num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)


        epi_hash = short_hash(epi_features, num_epi)
        params["experiment"] = f"{base_params["experiment"]}/{epiDir}"

        if iteration != 0:
            config = params
            base_dir = f"RUNlogs/{params['experiment']}/test_split{params['split']}/ctl{params['context_layers']}_bs{params['batch_size']}_ws{params["windowsize"]}_ue{num_epi}_seed{params["seed"]}_hash{epi_hash}" 
            chkpt = os.path.join(base_dir, "run_", f"vv{iteration-1}", "checkpoints", "last.ckpt")
            print(os.path.exists(chkpt), chkpt)
            params["chkpt"] = chkpt
        configs.append(params)

# Write to JSON
with open("configs/shuffled_pretrain_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configurations to shuffled_pretrain_configs.json")
