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
    "experiment": "PretrainingPaperCrossCellMLMNewScript",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "accumulate_grad_batches": 1,
    "dataset": "datasets/TcellAndK562StackedCorrCoords.tsv",
    "bw_dir": ["AGTensorsCL:0000624", "AGTensorsEFO:0002067", "AGTensorsEFO:0009747"]
}

FEATURES = {
    "ATAC": ["ATAC"],
    "Sequence": []
}


DATASETS = {
    "TCellPlusK562": "datasets/TcellAndK562StackedCorrCoords.tsv",
    "TCell": "datasets/TCellDatasetWithCorrectCoords.tsv"
}
# Full model

windowsizes = [512]
seeds = [0]
loop = list(range(0, 1))
configs = []
for iteration, seed, windowsize, feature_key, dataset_key in itertools.product(loop,seeds, windowsizes, FEATURES.keys(), DATASETS.keys()):
    params = base_params.copy()
    params["windowsize"] = windowsize
    epi_features = FEATURES[feature_key]
    params["epi_features"] = epi_features
    params["seed"] = seed
    params["dataset"] = DATASETS[dataset_key]
    #if windowsize == 23:
    #    params["accumulate_grad_batches"] = 4
    print(epi_features)
    num_epi = sum(EPI_FEATURES[key][-1] for key in epi_features)


    epi_hash = short_hash(epi_features, num_epi)
    params["experiment"] = f"{base_params["experiment"]}/{feature_key}_{dataset_key}"

    if iteration != 0:
        config = params
        base_dir = f"RUNlogs/{params['experiment']}/test_split{params['split']}/ctl{params['context_layers']}_bs{params['batch_size']}_ws{params["windowsize"]}_ue{num_epi}_seed{params["seed"]}_hash{epi_hash}" 
        chkpt = os.path.join(base_dir, "run_", f"vv{iteration-1}", "checkpoints", "last.ckpt")
        print(os.path.exists(chkpt), chkpt)
        params["chkpt"] = chkpt
    configs.append(params)

# Write to JSON
with open("configs/crosscell_mlm_pretrain_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print(f"Wrote {len(configs)} configurations to crosscell_mlm_pretrain_configs.json")
