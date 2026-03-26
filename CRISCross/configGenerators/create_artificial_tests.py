import json
import itertools
from src.pretrain import short_hash
import os
from src.Datasets import EPI_FEATURES
import pandas as pd
# Base/default parameters (copied from your idx is None branch)
base_params = {
    "batch_size": 128,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.2,
    "lr": 1e-4,
    "patience": 20,
    "experiment": "ArtificialFineTunePaper",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "accumulate_grad_batches": 2,
    "epi_mode": "np",
    "bw_dir": "AGTensorsEFO:0002067",
    "fit": True,
}

FINETUNINGK562 = "runSettings/RunSettingsFineTuneK562.tsv"

datasets = {"T_cell": "datasets/TCellDatasetWithCorrectCoords.tsv", "K562": "../datasets/k562_deepcrispr_withCoords_hg38.csv"}

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
k562.to_csv("datasets/K562WithIDs.tsv", sep="\t", index=False)
datasets["K562"] = "datasets/K562WithIDs.tsv"
unique_guides = k562["GuideID"].unique().tolist()
data = {
    "val_set": [[] for _ in range(len(unique_guides))],
    "test_set": [[guide] for guide in unique_guides],
    "exclude": [[] for _ in range(len(unique_guides))],
}

data_df = pd.DataFrame(data)
data_df.to_csv(FINETUNINGK562, sep="\t")


seeds = [i * 42 for i in range(10)]


CHKPTS = {
    "Arti-Ws512-Epi": os.path.join("RUNlogs/PretrainingArtificialPaperFixed/test_split1/ctl3_bs1024_ws512_ue0_seed0_energyTrue_hash/run_/vv1",   "checkpoints", "last.ckpt"),
    "CRISPert-Ws512-Epi": os.path.join("RUNlogs/PretrainingPaperFixed/None/test_split0/ctl3_bs512_ws512_ue0_seed0_hashd41d8cd98f/run_/vv0",  "checkpoints", "last.ckpt"),

}

run_settings = {"T_cell": "runSettings/RunSettingsLeaveOneOut.tsv", "K562": "runSettings/RunSettingsFineTuneK562.tsv"}

if __name__ == "__main__":
    configs = []
    for  chkpt_key, dataset_key in itertools.product(CHKPTS.keys(), datasets.keys()):
        rs  = run_settings[dataset_key]

        rdf = pd.read_csv(rs, sep="\t")
        splits = list(range(len(rdf)))
        for split in splits:
            params = base_params.copy()
            ws = int(CHKPTS[chkpt_key].split("_ws")[-1].split("_")[0])
            params["epi_features"] = []
            params["merge"] = None
            params["seed"] = seeds
            params["split"] = split
            params["windowsize"] = ws
            params["run_settings"] = run_settings[dataset_key]
            params["dataset"] = datasets[dataset_key]   
            num_epi = 0
            epi_hash = short_hash([], num_epi)
            chkpt = CHKPTS[chkpt_key]


            assert os.path.exists(chkpt), f"Checkpoint not found:\n{chkpt}"
            params["chkpt"] = chkpt
            mode = chkpt_key
            params["experiment"] = base_params["experiment"] + f"/{mode}/{dataset_key}" 



            configs.append(params)

    # Write to JSON
    with open("configs/run_artifical_configs_l40s.json", "w") as f:
        json.dump(configs, f, indent=2)



    print(
        f"Wrote {len(configs)} configs to run_artifical_configs_l40s.json and "
    )