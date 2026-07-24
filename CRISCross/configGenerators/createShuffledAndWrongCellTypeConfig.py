import json
import itertools
import os
from src.pretrain import short_hash

# Base/default parameters
base_params = {
    "batch_size": 256,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.3,
    "lr": 1e-4,
    "patience": 20,
    "experiment": "FineTuningShuffled/AG/0",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "run_settings": "runSettings/RunSettingsLeaveOneOut.tsv",
    "dataset": "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv",
}



seeds = [i * 42 for i in range(10)]
splits = list(range(17))
runs = [0]  # or [i * 42 for i in range(1)] if you prefer

configs = []
parent_dir = "PretrainingShuffled"

for epiDir in ["ShuffledAGTensors", "AGTensorsWrongCellType"]:
    for split, run in itertools.product(splits, runs):
        params = base_params.copy()

        params["windowsize"] = 23
        params["epi_features"] = ["ATAC"]
        num_epi = 1
        params["seed"] = seeds
        params["split"] = split
        params["experiment"] = f"ShuffledAndWrongCellType/{epiDir}"
        params["epi_dir"] = epiDir
        
        epi_hash = short_hash(["ATAC"], num_epi)

        base_dir = f"RUNlogs/{parent_dir}/{epiDir}/test_split0/ctl{params['context_layers']}_bs512_ws{params["windowsize"]}_ue{num_epi}_seed{run}_hash{epi_hash}" 
        chkpt = os.path.join(base_dir, "run_", "vv0", "checkpoints", "last.ckpt")
        assert os.path.exists(chkpt), f"Checkpoint not found:\n{chkpt}"
        params["chkpt"] = chkpt

        configs.append(params)

with open("configs/run_configs_shuffled.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configs to configs/run_configs_shuffled.json")