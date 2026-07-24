#!/usr/bin/env python3
"""Combine old model top-K results (topk_summary.csv) with CRISCross results (FinalSummary.tsv)
into a unified TSV file.

- Old models: already aggregated means from topk_summary.csv
- CRISCross: aggregated per (pretrain_config, dataset, window_size, feature_mode) from FinalSummary.tsv
- Only common K-values: 10, 50, 100, 500, 1000
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "Results")
OLD_CSV = os.path.join(BASE_DIR, "OldModelResults/topk_summary.csv")
FINAL_TSV = os.path.join(BASE_DIR, "FinalSummary.tsv")
OUTPUT_TSV = os.path.join(BASE_DIR, "combined_topk_summary.tsv")

K_VALUES = [10, 50, 100, 500, 1000]

# Dataset name normalization
DATASET_MAP = {"Tcell": "T-Cell", "ipscs": "IPSC", "k562": "K562"}


def load_old_models():
    """Load old model results from topk_summary.csv, keep only common K-values."""
    df = pd.read_csv(OLD_CSV)

    # Normalize dataset names
    df["dataset"] = df["dataset"].map(DATASET_MAP)

    rows = []
    for _, r in df.iterrows():
        row = {
            "model": r["model"],
            "dataset": r["dataset"],
            "feature_set": r.get("feature_set", ""),
            "window_size": "",
        }
        for k in K_VALUES:
            mean_col = f"precision@{k}__mean"
            n_col = f"precision@{k}__n_guides"
            row[f"precision@{k}__mean"] = r.get(mean_col, np.nan)
            row[f"precision@{k}__std"] = np.nan  # Old data is already aggregated
            row[f"precision@{k}__n"] = r.get(n_col, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def load_criscross():
    """Load CRISCross results from FinalSummary.tsv, aggregate per group."""
    df = pd.read_csv(FINAL_TSV, sep="\t")

    # Filter to CRISCross model only
    df = df[df["model"] == "CRISCross"].copy()

    # Build model name from pretrain config
    df["model"] = "CRISCross-" + df["run"].str.split("/").str[0]

    # Normalize dataset
    df["dataset"] = df["dataset"].replace(DATASET_MAP)

    # Feature set: mode column for T-Cell, empty for others
    df["feature_set"] = df.apply(
        lambda r: r["mode"] if r["dataset"] == "T-Cell" else "", axis=1
    )

    # Window size as string
    df["window_size"] = df["window_size"].astype(str)

    # Map precision columns
    prec_map = {
        "test_precision@10": "precision@10",
        "test_precision@50": "precision@50",
        "test_precision@100": "precision@100",
        "test_precision@500": "precision@500",
        "test_precision@1000": "precision@1000",
    }
    df.rename(columns=prec_map, inplace=True)

    # Group and aggregate
    group_cols = ["model", "dataset", "feature_set", "window_size"]
    grouped = df.groupby(group_cols)

    rows = []
    for name, grp in grouped:
        row = dict(zip(group_cols, name))
        for k in K_VALUES:
            col = f"precision@{k}"
            vals = grp[col].dropna()
            row[f"precision@{k}__mean"] = vals.mean() if len(vals) > 0 else np.nan
            row[f"precision@{k}__std"] = vals.std() if len(vals) > 1 else 0.0
            row[f"precision@{k}__n"] = len(vals)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    old_df = load_old_models()
    cross_df = load_criscross()

    # Define output column order
    header_cols = ["model", "dataset", "feature_set", "window_size"]
    metric_cols = []
    for k in K_VALUES:
        metric_cols.extend(
            [f"precision@{k}__mean", f"precision@{k}__std", f"precision@{k}__n"]
        )
    all_cols = header_cols + metric_cols

    # Combine
    combined = pd.concat([old_df[all_cols], cross_df[all_cols]], ignore_index=True)

    # Sort for readability: old models first, then CRISCross; within each by dataset
    combined["sort_key"] = combined["model"].apply(
        lambda m: (0, m) if not m.startswith("CRISCross") else (1, m)
    )
    combined.sort_values(["sort_key", "dataset", "model", "feature_set"], inplace=True)
    combined.drop(columns=["sort_key"], inplace=True)

    # Write output
    combined.to_csv(OUTPUT_TSV, sep="\t", index=False)
    print(f"Written {len(combined)} rows to {OUTPUT_TSV}")
    print(f"  Old models: {len(old_df)} rows")
    print(f"  CRISCross:  {len(cross_df)} rows")
    print()
    print("Models included:")
    print(combined["model"].unique())
    print()
    print("Datasets:", combined["dataset"].unique().tolist())


if __name__ == "__main__":
    main()
