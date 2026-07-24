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
    "experiment": "CrossCellTypeTestIPSC",
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



datasets = {"T_cell": "datasets/TCellDatasetWithCorrectCoords.tsv", "IPSCs": "../datasets/ipscs_dataset_invivo_full.csv"}

ipsc = pd.read_csv(datasets["IPSCs"], sep=",")
#k562["GuideID"] = k562["GuideID"] + "_k562"
ipsc["epiDir"] = "AGTensorsEFO:0009747"

ipsc = ipsc.rename(
    {
        "sgRNA": "Guide_sequence",
        "chrom": "chr",
        "strand": "Strand",
        "ID": "GuideID"
    },
    axis=1
)

ipsc["GuideID"] = pd.factorize(ipsc["Guide_sequence"])[0]
ipsc.to_csv("datasets/IPScWithEpidir.tsv", sep="\t", index=False)
datasets["IPSC"] = "datasets/IPScWithEpidir.tsv"

RUNSETTINGSIPSC= "runSettings/RunSettingsFullIPSC.tsv"
data = {
    "val_set": [[]],
    "test_set": [ipsc["GuideID"].unique().tolist()],
    "exclude": [[]],
}
data_df = pd.DataFrame(data)
data_df.to_csv(RUNSETTINGSIPSC, sep="\t")


# Load T_cell dataset
t_cell = pd.read_csv(datasets["T_cell"], sep="\t")
t_cell = t_cell.drop(["extended_off_target"], axis=1)
t_cell["GuideID"] = t_cell["GuideID"] + "_tcell"
t_cell["epiDir"] = "AGTensorsCL:0000624"
# Create directories for joined datasets and run settings


seeds = [i * 42 for i in range(25)]

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
            params["bw_dir"] = ["AGTensorsCL:0000624", "AGTensorsEFO:0002067", "AGTensorsEFO:0009747"]
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
            params["experiment"] = base_params["experiment"] + f"/{mode}_ccFineTuning/{dataset_key}_IPSC_seed{seed}" 
            
            
            sparam = params.copy()
            sparam["seed"] = seed
            base_dir, version = get_base(sparam)
            test_params = params.copy()
            test_params["seed"] = seed
            test_params["chkpt"] = os.path.join(base_dir, "run_/vv0/checkpoints/swa_model.ckpt")
            test_params["fit"] = False
            test_params["dataset"] = datasets["IPSC"]  # Use the 80% test split file
            test_params["run_settings"] = RUNSETTINGSIPSC
            test_params["experiment"] = base_params["experiment"] + f"/{mode}_ccTesting/IPSCTest_seed{seed}"
            test_params["bw_dir"] = ["AGTensorsCL:0000624", "AGTensorsEFO:0002067", "AGTensorsEFO:0009747"]



            fine_tune_configs.append(params)
            test_configs.append(test_params)
            idx += 1

    # Write to JSON
    with open("configs/fineTuningCRISPRATAGIPSC.json", "w") as f:
        json.dump(fine_tune_configs, f, indent=2)

    with open("configs/TestCRISPRATAGIPSC.json", "w") as f:
        json.dump(test_configs, f, indent=2)

    print(
        f"Wrote {len(fine_tune_configs)} configs to fineTuningCRISPRATAGIPSC.json"
    )
    print(
        f"Wrote {len(test_configs)} configs to TestCRISPRATAGIPSC.json"
    )
