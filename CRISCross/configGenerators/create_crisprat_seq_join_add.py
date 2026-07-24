import json
from itertools import product
from src.pretrainArtificial import short_hash
from src.Datasets import EPI_FEATURES
import os

# Base parameters
base_params = {
    "batch_size": 1024 //4,
    "context_layers": 3,
    "hidden_dim": 512,
    "embed_size": 32,
    "dropout": 0.2,
    "lr": 1e-4,
    "patience": 100000,
    "seed": 0,
    "split": 1,
    "experiment": "PretrainingArtificialPaperSeqJoin",
    "regression": False,
    "windowsize": 512,
    "model_type": "crosscrispr",
    "bw_dir": ["AGTensorsCL:0000624"],
    "epi_mode": "np"
}

# Epi features
epi_features_list = [
    [
    ],
]

# Use_energy options
use_energy_list = [True]
loop = list(range(0,5))
join_methods = ["add", "concat"]
seeds = [0]
# Generate all combinations
all_combinations = []
l_40_combs = []
for join_method, iteration, seed, epi_features, use_energy in product(join_methods, loop, seeds, epi_features_list, use_energy_list):
    params = base_params.copy()
    params["epi_features"] = epi_features
    params["use_energy"] = use_energy
    params["merge"] = None if not epi_features else "early"
    params["windowsize"] = 512
    params["join_method"] = join_method
    params["seed"] = seed
    params["experiment"] = f"{base_params['experiment']}_{join_method}"
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)

    epi_hash = short_hash(epi_features, num_epi)
    params["num_epi"] = num_epi
    if iteration != 0:
        config = params
        bs = 1024 if iteration == 1 else config['batch_size']

        base_dir = f"RUNlogs/{config['experiment']}/test_split{config['split']}/ctl{config['context_layers']}_bs{bs}_ws{config["windowsize"]}_ue{config['num_epi']}_seed{config['seed']}_energy{config["use_energy"]}_hash{epi_hash}" 
        chkpt = os.path.join(base_dir, "run_", f"vv{iteration-1}", "checkpoints", "last.ckpt")
        print(os.path.exists(chkpt), chkpt)
        params["chkpt"] = chkpt
    all_combinations.append(params)

        



# Save to JSON
with open("configs/artificial_SEQ_param_JoinMethods.json", "w") as f:
    json.dump(all_combinations, f, indent=4)



print(f"Generated {len(all_combinations)} parameter combinations for h200.")
print(f"Generated {len(l_40_combs)} parameter combinations for l40.")
