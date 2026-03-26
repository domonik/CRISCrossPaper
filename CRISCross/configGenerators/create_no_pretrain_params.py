import json

base_params = {
    "batch_size": 256,
    "context_layers": 3,
    "embed_size": 32,
    "hidden_dim": 512,
    "dropout": 0.3,
    "lr": 1e-4,
    "patience": 20,
    "experiment": "FineTuningPaperNoPretrain",
    "regression": False,
    "merge": "early",
    "model_type": "crosscrispr",
    "run_settings": "RunSettingsLeaveOneOut.tsv",
    "dataset": "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv",
}




windowsizes = [23, 128, 512]
runs = [i * 42 for i in range(0, 1)]
seeds = [i * 42 for i in range(10)]

splits = list(range(17))

configs_l40s = []
configs_h200 = []

for split in splits:
    for run in runs:
        params = dict(
                windowsize=23,
                epi_features=[],
                seed=seeds,
                split = split,                    
                **base_params
                )
        params["experiment"] = f"{base_params["experiment"]}/NoPretrain/{run}"
        params["merge"] = None
        configs_l40s.append(params)
        
        
with open("configs/run_configs_no_pretrain.json", "w") as f:
    json.dump(configs_l40s, f, indent=2)
    
    
print(
    f"Wrote {len(configs_l40s)} configs to run_configs_no_pretrain.json"
)
