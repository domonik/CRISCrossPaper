import json
import itertools
from src.pretrain import get_base 

from src.pretrain import short_hash
import os
from src.Datasets import EPI_FEATURES
import pandas as pd
import numpy as np

# Base/default parameters (copied from your idx is None branch)
base_params = {
    "batch_size": 128,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.2,
    "lr": 1e-4,
    "patience": 10,
    "experiment": "CrossCellTypeTest",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "accumulate_grad_batches": 2,
    "epi_mode": "np",
    "max_epochs": 20 * 2,
    "swa_epoch_start": 6*2
    
}

epi_features = [
        "ATAC",
    ]

# Feature groups

RUNSETTINGSTCELL = "runSettings/RunSettingsFullTCell.tsv"

data = {
    "val_set": [None],
    "test_set": [[]],
    "exclude": [[]],
}
data_df = pd.DataFrame(data)
data_df.to_csv(RUNSETTINGSTCELL, sep="\t")


datasets = {"T_cell": "datasets/TCellDatasetWithCorrectCoords.tsv", "HEK293T": "Hek293WithextendedSequences.tsv", "K562": "../datasets/k562_deepcrispr_withCoords_hg38.csv"}

k562 = pd.read_csv(datasets["K562"], sep=",")
#k562["GuideID"] = k562["GuideID"] + "_k562"
k562["epiDir"] = "AGTensorsEFO:0002067"

k562 = k562.rename(
    {
        "sgRNA": "Guide_sequence",
        "chrom": "chr",
        "strand": "Strand",
        "ID": "GuideID"
    },
    axis=1
)

k562["GuideID"] = pd.factorize(k562["Guide_sequence"])[0]
k562.to_csv("datasets/K562WithEpidir.tsv", sep="\t", index=False)
datasets["K562"] = "datasets/K562WithEpidir.tsv"

RUNSETTINGSK562 = "runSettings/RunSettingsFullK562.tsv"
data = {
    "val_set": [[]],
    "test_set": [k562["GuideID"].unique().tolist()],
    "exclude": [[]],
}
data_df = pd.DataFrame(data)
data_df.to_csv(RUNSETTINGSK562, sep="\t")


# Load T_cell dataset
t_cell = pd.read_csv(datasets["T_cell"], sep="\t")
t_cell = t_cell.drop(["extended_off_target"], axis=1)
t_cell["GuideID"] = t_cell["GuideID"] + "_tcell"
t_cell["epiDir"] = "AGTensorsCL:0000624"
# Create directories for joined datasets and run settings
joined_datasets_dir = "joined_datasets"
run_settings_dir = "run_settings"
os.makedirs(joined_datasets_dir, exist_ok=True)
os.makedirs(run_settings_dir, exist_ok=True)

seeds = [i * 42 for i in range(50)]

CHKPTS = {
    "Arti-Ws512-AG": os.path.join("RUNlogs/PretrainingArtificialPaperFixed/test_split1/ctl3_bs1024_ws512_ue0_seed0_energyTrue_hash/run_/vv1",   "checkpoints", "last.ckpt"),
    "Arti-Ws512+AG": os.path.join("RUNlogs/PretrainingArtificialPaperAG2/test_split1/ctl3_bs1024_ws512_ue1_seed0_energyTrue_hash2/run_/vv1",  "checkpoints", "last.ckpt"),
}


if __name__ == "__main__":
    fine_tune_configs = []
    test_configs = []
    idx = 0
    for seed in seeds:
        # Create a different 80/20 split for each seed (20% of rows)
        np.random.seed(seed)
        for chkpt_key in CHKPTS.keys():
            dataset_key = "T_cell"

            params = base_params.copy()
            ws = int(CHKPTS[chkpt_key].split("_ws")[-1].split("_")[0])
            params["epi_features"] = epi_features if "+AG" in chkpt_key else []
            params["merge"] = "early" if "+AG" in chkpt_key else None
            params["seed"] = [seed]
            params["split"] = 0
            params["bw_dir"] = ["AGTensorsCL:0000624", "AGTensorsEFO:0002067"]
            params["fit"] = True
            params["windowsize"] = ws
            params["run_settings"] = RUNSETTINGSTCELL
            params["dataset"] = datasets["T_cell"]
            params["IDX"] = idx
            num_epi = sum(EPI_FEATURES[key][-1] for key in params["epi_features"])
            params["num_epi"] = num_epi
            chkpt = CHKPTS[chkpt_key]


            assert os.path.exists(chkpt), f"Checkpoint not found:\n{chkpt}"
            params["chkpt"] = chkpt
            mode = chkpt_key
            params["experiment"] = base_params["experiment"] + f"/{mode}_ccFineTuning/{dataset_key}_K562Val_seed{seed}" 
            
            
            sparam = params.copy()
            sparam["seed"] = seed
            base_dir, version = get_base(sparam)
            test_params = params.copy()
            test_params["seed"] = seed
            test_params["chkpt"] = os.path.join(base_dir, "run_/vv0/checkpoints/swa_model.ckpt")
            test_params["fit"] = False
            test_params["dataset"] = datasets["K562"]  # Use the 80% test split file
            test_params["run_settings"] = RUNSETTINGSK562
            test_params["experiment"] = base_params["experiment"] + f"/{mode}_ccTesting/K562Test_seed{seed}"
            test_params["bw_dir"] = ["AGTensorsCL:0000624", "AGTensorsEFO:0002067"]



            fine_tune_configs.append(params)
            test_configs.append(test_params)
            idx += 1

    # Write to JSON
    with open("configs/fineTuningCRISPRATAG.json", "w") as f:
        json.dump(fine_tune_configs, f, indent=2)

    with open("configs/TestCRISPRATAG.json", "w") as f:
        json.dump(test_configs, f, indent=2)

    print(
        f"Wrote {len(fine_tune_configs)} configs to fineTuningCRISPRATAG.json"
    )
    print(
        f"Wrote {len(test_configs)} configs to TestCRISPRATAG.json"
    )
