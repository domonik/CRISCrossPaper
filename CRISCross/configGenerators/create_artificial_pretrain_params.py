import json
from itertools import product
from src.pretrainArtificial import short_hash
from src.Datasets import EPI_FEATURES
import os

# Base parameters
base_params = {
    "batch_size": 1024,
    "context_layers": 3,
    "hidden_dim": 512,
    "embed_size": 32,
    "dropout": 0.2,
    "lr": 1e-4,
    "patience": 100000,
    "seed": 0,
    "split": 1,
    "experiment": "PretrainingArtificialPaperFixed",
    "regression": False,
    "windowsize": 512,
    "model_type": "crosscrispr",
}

# Epi features
epi_features_list = [
    []  # empty list
]

# Use_energy options
use_energy_list = [True]
loop = list(range(0,5))
window_sizes = [512]
seeds = [0]
# Generate all combinations
all_combinations = []
l_40_combs = []
for iteration, seed, epi_features, use_energy, ws in product(loop, seeds, epi_features_list, use_energy_list, window_sizes):
    params = base_params.copy()
    params["epi_features"] = epi_features
    params["use_energy"] = use_energy
    params["merge"] = None if not epi_features else "early"
    params["windowsize"] = ws
    params["seed"] = seed
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)

    epi_hash = short_hash(epi_features, num_epi)
    params["num_epi"] = num_epi
    if iteration != 0:
        config = params
        base_dir = f"RUNlogs/{config['experiment']}/test_split{config['split']}/ctl{config['context_layers']}_bs{config['batch_size']}_ws{config["windowsize"]}_ue{config['num_epi']}_seed{config['seed']}_energy{config["use_energy"]}_hash{epi_hash}" 
        chkpt = os.path.join(base_dir, "run_", f"vv{iteration-1}", "checkpoints", "last.ckpt")
        print(os.path.exists(chkpt), chkpt)
        params["chkpt"] = chkpt
    if ws == 512:
        all_combinations.append(params)
    elif (len(epi_features) == 0):
        l_40_combs.append(params)
        



# Save to JSON
with open("configs/artificial_param_combinations.json", "w") as f:
    json.dump(all_combinations, f, indent=4)

with open("configs/artificial_param_combinationsl40.json", "w") as f:
    json.dump(l_40_combs, f, indent=4)

print(f"Generated {len(all_combinations)} parameter combinations for h200.")
print(f"Generated {len(l_40_combs)} parameter combinations for l40.")
