import json
import itertools
from src.pretrain import short_hash
import os
from src.Datasets import EPI_FEATURES


parent_dir = "PretrainingPaperAblationOnlyOverlap"
# Base/default parameters (copied from your idx is None branch)
base_params = {
    "batch_size": 256,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.3,
    "lr": 1e-4,
    "patience": 20,
    "experiment": "FineTuningAblationLeaveOneOutOnlyOverlap",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "run_settings": "runSettings/RunSettingsLeaveOneOut.tsv",
    "dataset": "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv",
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
feature_sets["abl_no_Histone"] = [
    FEATURES[k] for k in FEATURES if k not in HISTONE_KEYS
]
feature_sets["abl_no_ATAC"] = [FEATURES[k] for k in FEATURES if not k.startswith("ATAC")]


windowsizes = [23]
runs = [i * 42 for i in range(0, 1)]
splits = list(range(17))
configs_l40s = []
configs_h200 = []
seeds = list(range(10))
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

for windowsize, feature_key, split, run in itertools.product(windowsizes, feature_sets.keys(), splits, runs):
    params = base_params.copy()
    params["windowsize"] = windowsize
    epi_features = feature_sets[feature_key]

    params["epi_features"] = epi_features
    params["seed"] = seeds
    params["split"] = split
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)
    epi_hash = short_hash(epi_features, num_epi)

    params["experiment"] = f"{base_params["experiment"]}/{feature_key}"



    base_dir = f"RUNlogs/{parent_dir}/{feature_key}/test_split0/ctl{params['context_layers']}_bs512_ws{params["windowsize"]}_ue{num_epi}_seed{run}_hash{epi_hash}" 
    chkpt = os.path.join(base_dir, "run_", "vv0", "checkpoints", "last.ckpt")
    assert os.path.exists(chkpt), f"Checkpoint not found:\n{chkpt}"
    params["chkpt"] = chkpt



    if windowsize >= 512 and params["context_layers"] > 3:
        params["accumulate_grad_batches"] = 2
        params["batch_size"] = 128
    
    configs_l40s.append(params)


# Write to JSON
with open("configs/run_configs_ablation_l40s.json", "w") as f:
    json.dump(configs_l40s, f, indent=2)

with open("configs/run_configs_ablation_h200.json", "w") as f:
    json.dump(configs_h200, f, indent=2)

print(
    f"Wrote {len(configs_l40s)} configs to run_configs_ablation_l40s.json and "
    f"{len(configs_h200)} configs to run_configs_ablation_h200.json"
)