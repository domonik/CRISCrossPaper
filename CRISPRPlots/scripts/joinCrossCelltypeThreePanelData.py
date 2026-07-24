import pandas as pd

AG_STRING = "Sequence & AlphaGenome ATAC-seq"
SEQ_STRING = "Sequence only"

# AllExperiments_PerGuide.tsv "dataset" values -> combined_results.csv "dataset" keys.
# The two MLM-trained datasets (CrossCellTypeTestMLMFull / CrossCellTypeTestMLMTCell)
# are intentionally excluded here.
CRISCROSS_DATASET_MAP = {
    "CrossCellTypeTest": "k562_deepcrispr",
    "CrossCellTypeTestIPSC": "ipscs_full",
    "CrossCellTypeTestIVK562": "k562_full",
}

# Per-dataset guide tables: give the GuideID -> Guide_sequence mapping used by
# AllExperiments_PerGuide.tsv, and (via the "label" column) the per-guide
# positive/sample counts, i.e. the dataset imbalance.
GUIDE_TABLE_FILES = {
    "k562_deepcrispr": "../CRISCross/datasets/K562WithEpidir.tsv",
    "ipscs_full": "../CRISCross/datasets/IPScWithEpidir.tsv",
    "k562_full": "../CRISCross/datasets/K562IvFullWithEpidir.tsv",
}

# Raw "model" values in per_guide_precision_recall_k1-200_updated.csv ->
# correct display names, matching the names used throughout the rest of
# the codebase (e.g. MODEL_NAMES in plotCrossCelltypeRecallAt90.py).
# CRISOT_changeseq is intentionally left unmapped: it's dropped below as
# redundant with CRISOT_tcell (-> "CRISOT").
MODEL_DISPLAY_NAME = {
    "CRISOT_tcell": "CRISOT",
    "CRISPROfft": "CRISPR-OFFT",
    "crisprIP": "CRISPR-IP",
    "cnnCRISPR": "CnnCrispr",
    "CrisprBERT_Osari": "CrisprBERT",
    "CRISPert": "CRISPert",
}

PRECISION_COLS = [
    f"precision_at_{k}" for k in range(1, 201)
]

RECALL_COLS = [
    f"recall_at_{k}" for k in range(1, 201)
]


def load_guide_tables():
    tables = []
    for dataset, file in GUIDE_TABLE_FILES.items():
        df = pd.read_csv(file, sep="\t")
        g = df.groupby("GuideID").agg(
            guide_id=("Guide_sequence", "first"),
            n_positives=("label", "sum"),
            n_samples=("label", "count"),
        ).reset_index()
        g["dataset"] = dataset
        tables.append(g)
    return pd.concat(tables, ignore_index=True)


def load_criscross(file, guide_tables):
    df = pd.read_csv(file, sep="\t")
    df = df[df["dataset"].isin(CRISCROSS_DATASET_MAP)].copy()
    df["dataset"] = df["dataset"].map(CRISCROSS_DATASET_MAP)
    df["features"] = df["condition"].map({"+AG": AG_STRING, "-AG": SEQ_STRING})
    df = df.rename(columns={"recall_at_90pct_precision": "recall_at_precision90"})

    # AllExperiments_PerGuide.tsv's GuideID column is object-dtype (other
    # datasets in that file use "sgN" labels), but for the three datasets
    # kept here it always holds plain integers as strings; cast to int so it
    # matches guide_tables' int64 GuideID for the merge below.
    df["GuideID"] = df["GuideID"].astype(int)

    # ipscs_full: keep only the guide with 53 positives, which is GuideID 0
    # in AllExperiments_PerGuide.tsv.
    df = df[(df["dataset"] != "ipscs_full") | (df["GuideID"] == 0)]

    # Average over seeds per guide, so CRISCross rows are at the same
    # granularity (one row per model/dataset/guide) as the CSV sources.
    agg_cols = ["aucpr"] + PRECISION_COLS + RECALL_COLS + ["recall_at_precision90"]
    df = df.groupby(
        ["dataset", "features", "GuideID"], as_index=False
    )[agg_cols].mean()

    # Replace the numeric GuideID with the actual guide sequence and attach
    # the per-guide positive/sample counts from the raw dataset tables.
    df = df.merge(guide_tables, on=["dataset", "GuideID"], how="left")
    assert df["guide_id"].notna().all(), "Unmapped GuideID(s) found"
    df = df.drop(columns=["GuideID"])

    df["model"] = "CRISCross"
    df["model_group"] = "CRISCross"
    return df


def load_csv_models(file, model_group):
    df = pd.read_csv(file)
    df = df.rename(columns={"aucpr_mean_over_seeds": "aucpr", "sgRNA": "guide_id"})
    df["features"] = SEQ_STRING
    df["model_group"] = model_group

    # CRISOT_changeseq is redundant with CRISOT_tcell (displayed as
    # "CRISOT"); drop it so every downstream consumer sees one CRISOT series.
    df = df[df["model"] != "CRISOT_changeseq"]
    df["model"] = df["model"].map(MODEL_DISPLAY_NAME)
    assert df["model"].notna().all(), "Unmapped model name(s) found"

    # ipscs_full: keep only the guide with 53 positives.
    df = df[(df["dataset"] != "ipscs_full") | (df["n_positives"] == 52)]

    keep = ["model", "model_group", "dataset", "features", "guide_id",
            "n_positives", "n_samples", "aucpr"] \
        + PRECISION_COLS + RECALL_COLS + ["recall_at_precision90"]
    df = df[[k for k in keep if k in df.columns]]
    return df


def main():
    guide_tables = load_guide_tables()

    cross = load_criscross("AllExperiments_PerGuide.tsv", guide_tables)
    old = load_csv_models("per_guide_precision_recall_k1-200_updated.csv", "old")
    #new = load_csv_models("combined_results_new_models.csv", "new")

    keep = ["model", "model_group", "dataset", "features", "guide_id",
            "n_positives", "n_samples", "aucpr"] \
        + PRECISION_COLS + RECALL_COLS + ["recall_at_precision90"]
    combined = pd.concat(
        [
            cross.reindex(columns=keep),
            old.reindex(columns=keep),
            #new.reindex(columns=keep),
        ],
        ignore_index=True,
    )
    combined["positive_rate"] = combined["n_positives"] / combined["n_samples"]
    combined = combined.sort_values(
        ["dataset", "model_group", "model", "features", "guide_id"]
    ).reset_index(drop=True)

    out_file = "CrossCelltypeThreePanelCombined.tsv"
    combined.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {len(combined)} rows to {out_file}")


if __name__ == "__main__":
    main()
