"""
Create a unified table from all old model results in OldModelResults/.

Output columns (matching joinAllResults.py shared schema):
    model | dataset | mode | split | seed | AUCPR

Mappings:
  - *_k562_results.csv    → dataset=K562,   mode=Base, split=0
  - *_ipscs_results.csv   → dataset=IPSC,   mode=Base, split=0
  - *_LOO2bit_*.csv       → dataset=T-Cell, mode=feature_set (base→Base, ex→EX, ag→AG),
    split=test_group mapped via sgRNA grouping from joinAllResults.py
"""

import os
import pandas as pd

RESULTS_DIR = "../Results/OldModelResults"
OUTPUT_FILE = "../Results/OldModelsUnified.tsv"
NOPRETRAIN_FILE = "../Results/CRISPert_nopretraining.csv"

# sgRNA → split index (same grouping as joinAllResults.py)
GROUPED_SG = [
    ["sg28", "sg23", "sg3"],
    ["sg2", "sg24", "sg8"],
    ["sg1", "sg6", "sg7"],
    ["sg12", "sg18", "sg10"],
    ["sg13", "sg26", "sg5"],
    ["sg19", "sg16"],
]
SGRNA_TO_SPLIT = {}
for split_idx, group in enumerate(GROUPED_SG):
    for sg in group:
        SGRNA_TO_SPLIT[sg] = split_idx

# feature_set → mode mapping for LOO2bit files
FEATURE_TO_MODE = {"base": "Base", "ex": "EX", "ag": "AG"}

OLD_MODELS = ["cnnCRISPR", "CRISPert", "crisprIP", "CRISPROfft"]


def load_k562_ipscs(dfs: list[pd.DataFrame]):
    """Load *_k562_results.csv and *_ipscs_results.csv files."""
    for fpath in sorted(os.listdir(RESULTS_DIR)):
        if fpath.endswith("_k562_results.csv"):
            model = fpath.split("_k562")[0]
            df = pd.read_csv(os.path.join(RESULTS_DIR, fpath))
            df["model"] = model
            df["dataset"] = "K562"
            df["mode"] = "Base"
            df["split"] = 0
            df["AUCPR"] = df["aucpr"]
            df = df[["model", "dataset", "mode", "split", "seed", "AUCPR"]]
            dfs.append(df)

        elif fpath.endswith("_ipscs_results.csv"):
            model = fpath.split("_ipscs")[0]
            df = pd.read_csv(os.path.join(RESULTS_DIR, fpath))
            df["model"] = model
            df["dataset"] = "IPSC"
            df["mode"] = "Base"
            df["split"] = 0
            df["AUCPR"] = df["aucpr"]
            df = df[["model", "dataset", "mode", "split", "seed", "AUCPR"]]
            dfs.append(df)


def load_loo2bit(dfs: list[pd.DataFrame]):
    """Load *_LOO2bit_per_sgRNA_aucpr.csv files (keep all rows)."""
    for fpath in sorted(os.listdir(RESULTS_DIR)):
        if "LOO2bit" in fpath and fpath.endswith("_per_sgRNA_aucpr.csv"):
            model = fpath.split("_LOO2bit")[0]
            df = pd.read_csv(os.path.join(RESULTS_DIR, fpath))

            df["model"] = model
            df["dataset"] = "T-Cell"
            df["mode"] = df["feature_set"].map(FEATURE_TO_MODE)
            df["split"] = df["test_group"].map(SGRNA_TO_SPLIT)
            df["AUCPR"] = df["aucpr"]
            df = df[["model", "dataset", "mode", "split", "seed", "AUCPR"]]

            # Validate no unmapped values
            assert df["mode"].notna().all(), f"Unmapped feature_set in {fpath}"
            assert df["split"].notna().all(), f"Unmapped test_group in {fpath}"

            dfs.append(df)


def load_crispert_pretrain_ablation(dfs: list[pd.DataFrame]):
    """Restore the CRISPert pretrain-ablation pair for Figure 1 panel a:
    'CRISPert no pretrain' (trained from scratch) vs 'CRISPert original'
    (published, pretrained baseline == the T-Cell 'base' feature-set run
    already loaded by load_loo2bit). Tagged window_size=23 and given their
    own mode labels so they line up with plotCrossAttentionResultsV4.py's
    no_pretrain_comp filter without mixing into the generic Base/Ex/AG
    epi-mode comparisons (those stay window_size=NaN, dataset-agnostic)."""
    df_np = pd.read_csv(NOPRETRAIN_FILE)
    df_np["model"] = "CRISPert"
    df_np["dataset"] = "T-Cell"
    df_np["mode"] = "CRISPert no pretrain"
    df_np["split"] = df_np["group"].map(SGRNA_TO_SPLIT)
    df_np["AUCPR"] = df_np["AUPRC"]
    df_np["window_size"] = 23
    df_np = df_np[["model", "dataset", "mode", "split", "seed", "AUCPR", "window_size"]]
    assert df_np["split"].notna().all(), "Unmapped sgRNA group in CRISPert_nopretraining.csv"
    dfs.append(df_np)

    loo2bit_path = os.path.join(RESULTS_DIR, "CRISPert_LOO2bit_per_sgRNA_aucpr.csv")
    df_pt = pd.read_csv(loo2bit_path)
    df_pt = df_pt[df_pt["feature_set"] == "base"].copy()
    df_pt["model"] = "CRISPert"
    df_pt["dataset"] = "T-Cell"
    df_pt["mode"] = "CRISPert original"
    df_pt["split"] = df_pt["test_group"].map(SGRNA_TO_SPLIT)
    df_pt["AUCPR"] = df_pt["aucpr"]
    df_pt["window_size"] = 23
    df_pt = df_pt[["model", "dataset", "mode", "split", "seed", "AUCPR", "window_size"]]
    assert df_pt["split"].notna().all(), "Unmapped test_group in CRISPert_LOO2bit_per_sgRNA_aucpr.csv"
    dfs.append(df_pt)


def main():
    dfs: list[pd.DataFrame] = []
    load_k562_ipscs(dfs)
    load_loo2bit(dfs)
    load_crispert_pretrain_ablation(dfs)

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values(["model", "dataset", "mode", "split", "seed"]).reset_index(drop=True)

    result.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Wrote {len(result)} rows to {OUTPUT_FILE}")
    print(f"  Models:       {sorted(result['model'].unique())}")
    print(f"  Datasets:     {sorted(result['dataset'].unique())}")
    print(f"  Modes:        {sorted(result['mode'].unique())}")
    print(f"  Splits:       {sorted(result['split'].unique())}")
    print(f"  Seeds range:  {result['seed'].min()} - {result['seed'].max()}")


if __name__ == "__main__":
    main()
