#!/usr/bin/env python3
"""
Collect AUCPR per GuideID from Arti-Ws512 ccTesting results, plus a pooled
(GuideID-independent) AUCPR summary, for three experiments.

Walks Arti-Ws512+AG_ccTesting and Arti-Ws512-AG_ccTesting under each
experiment's RUNlogs directory, finds every final_predictions.tsv, and
computes:

  1. Per-GuideID AUCPR, precision@k, recall@k, per seed, per AG-condition,
     per experiment -> one joined table across all experiments (with a
     "dataset" column)

  2. Pooled AUCPR per seed, per AG-condition, per experiment, where all
     predictions/labels for a given (condition, seed) are pooled together
     ACROSS guides before computing AUCPR (i.e. GuideID is ignored, not
     averaged over) -> a second joined table across all experiments

Outputs:
  ../CRISPRPlots/AllExperiments_PerGuide.tsv
  ../CRISPRPlots/AllExperiments_PooledByCondition.tsv
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

# k values for precision@k
KS = list(range(1, 201))

# precision threshold for recall@precision
TARGET_PRECISION = 0.90


def average_precision_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute the area under the precision-recall curve (average precision).
    Matches sklearn's default step-interpolation approach.
    """
    # Sort by descending score
    sorted_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Cumulative positives
    n_pos = np.cumsum(y_true_sorted)
    # Total examples considered at each step (1-indexed)
    n_examined = np.arange(1, len(y_true_sorted) + 1)

    # Precision at each step
    precision = n_pos / n_examined

    # Only count precision changes at positions where y_true == 1 (step interpolation)
    # AP = sum(precision[k] * (recall[k] - recall[k-1])) for k where y_true[k] == 1
    # Since recall[k] - recall[k-1] = 1/total_positives when y_true[k] == 1,
    # AP = sum(precision[k]) for k where y_true[k] == 1 / total_positives
    total_pos = y_true_sorted.sum()
    if total_pos == 0:
        return float("nan")

    ap = np.sum(precision[y_true_sorted == 1]) / total_pos
    return float(ap)


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Precision among the top-k highest-scored examples.
    Returns NaN if there are fewer than k examples total (top-k is undefined).
    """
    n_total = len(y_true)
    if n_total < k:
        return float("nan")

    sorted_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    top_k_labels = y_true[sorted_indices[:k]]
    return float(top_k_labels.sum() / k)


def recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Recall among the top-k highest-scored examples (fraction of all positives
    that are captured within the top-k). Returns NaN if there are fewer than
    k examples total (top-k is undefined) or if there are no positives.
    """
    n_total = len(y_true)
    if n_total < k:
        return float("nan")

    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan")

    sorted_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    top_k_labels = y_true[sorted_indices[:k]]
    return float(top_k_labels.sum() / total_pos)


def recall_at_precision(y_true: np.ndarray, y_scores: np.ndarray, target_precision: float) -> float:
    """
    Highest achievable recall among thresholds (sweeping the sorted scores)
    where precision >= target_precision. Returns NaN if that precision level
    is never reached.
    """
    total_pos = y_true.sum()
    if total_pos == 0:
        return float("nan")

    sorted_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_true_sorted = y_true[sorted_indices]

    n_pos_cum = np.cumsum(y_true_sorted)
    n_examined = np.arange(1, len(y_true_sorted) + 1)

    precision = n_pos_cum / n_examined
    recall = n_pos_cum / total_pos

    valid = precision >= target_precision
    if not valid.any():
        return float("nan")

    return float(recall[valid].max())


def compute_metrics(labels: np.ndarray, preds: np.ndarray, ks: list[int]) -> dict:
    """
    Compute AUCPR, precision@k and recall@k (for each k in ks), and
    recall@90%precision for one set of labels/predictions. Returns a dict of
    metric_name -> value. If the group has no positives or no negatives, all
    metrics are NaN.
    """
    n_pos = int(labels.sum())
    n_total = len(labels)

    metrics = {}

    if n_pos == 0 or n_pos == n_total:
        metrics["aucpr"] = float("nan")
        for k in ks:
            metrics[f"precision_at_{k}"] = float("nan")
            metrics[f"recall_at_{k}"] = float("nan")
        metrics[f"recall_at_{int(TARGET_PRECISION * 100)}pct_precision"] = float("nan")
        return metrics

    metrics["aucpr"] = average_precision_score(labels, preds)
    for k in ks:
        metrics[f"precision_at_{k}"] = precision_at_k(labels, preds, k)
        metrics[f"recall_at_{k}"] = recall_at_k(labels, preds, k)
    metrics[f"recall_at_{int(TARGET_PRECISION * 100)}pct_precision"] = recall_at_precision(
        labels, preds, TARGET_PRECISION
    )

    return metrics


SEED_RE = re.compile(r"seed(\d+)")


def extract_seed(dirname: str) -> str:
    """Extract seed value from a directory name containing 'seed<N>', e.g.
    'K562Test_seed126' or 'ctl3_bs128_ws512_ue0_seed126_hashd41d8cd98f'."""
    match = SEED_RE.search(dirname)
    if match is None:
        raise ValueError(f"No seed found in directory name: {dirname}")
    return match.group(1)


def extract_condition(top_dir: str) -> str:
    """Extract +AG / -AG from parent directory name. Defaults to -AG when neither is present."""
    if "+AG" in top_dir:
        return "+AG"
    return "-AG"


def find_predictions(base: str) -> list[str]:
    """Recursively find all final_predictions.tsv under base."""
    pattern = os.path.join(base, "**", "final_predictions.tsv")
    return glob.glob(pattern, recursive=True)


def collect_raw(test_dirs: list[str]) -> pd.DataFrame:
    """
    Walk the given test dirs and return a long-format dataframe with one row
    per prediction: condition, seed, GuideID, label, prediction.
    """
    rows = []

    for test_dir in test_dirs:
        condition = extract_condition(test_dir)
        files = find_predictions(test_dir)
        print(f"{condition}: found {len(files)} prediction files")

        for fpath in files:
            # Determine seed from the deepest matching parent directory
            parts = fpath.split(os.sep)
            seed_dirname = None
            for p in reversed(parts):
                if SEED_RE.search(p):
                    seed_dirname = p
                    break
            if seed_dirname is None:
                print(f"WARNING: Could not determine seed from path: {fpath}", file=sys.stderr)
                continue
            seed = int(extract_seed(seed_dirname))

            df = pd.read_csv(fpath, sep="\t", index_col=0)

            chunk = pd.DataFrame({
                "condition": condition,
                "seed": seed,
                "GuideID": df["GuideID"].values,
                "label": df["label"].astype(float).values,
                "prediction": df["predictions"].astype(float).values,
            })
            rows.append(chunk)

    if not rows:
        return pd.DataFrame(columns=["condition", "seed", "GuideID", "label", "prediction"])

    return pd.concat(rows, ignore_index=True)


def per_guide_aucpr(raw_df: pd.DataFrame) -> pd.DataFrame:
    """AUCPR, precision@k, and recall@90%precision per condition/seed/GuideID."""
    results = []
    for (condition, seed, guide_id), group in raw_df.groupby(["condition", "seed", "GuideID"]):
        labels = group["label"].values
        preds = group["prediction"].values

        metrics = compute_metrics(labels, preds, KS)

        results.append({
            "condition": condition,
            "seed": seed,
            "GuideID": guide_id,
            **metrics,
        })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["condition", "seed", "GuideID"]).reset_index(drop=True)
    return out_df


def pooled_aucpr(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    AUCPR, precision@k, and recall@90%precision per condition/seed, pooling
    all predictions/labels across guides (GuideID is ignored, not averaged over).
    """
    results = []
    for (condition, seed), group in raw_df.groupby(["condition", "seed"]):
        labels = group["label"].values
        preds = group["prediction"].values

        metrics = compute_metrics(labels, preds, KS)

        results.append({
            "condition": condition,
            "seed": seed,
            **metrics,
        })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["condition", "seed"]).reset_index(drop=True)
    return out_df


def run_experiment(dataset_name: str, test_dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect raw predictions for one experiment and return (per_guide_df, pooled_df),
    each tagged with a 'dataset' column."""
    raw_df = collect_raw(test_dirs)

    per_guide_df = per_guide_aucpr(raw_df)
    per_guide_df.insert(0, "dataset", dataset_name)

    pooled_df = pooled_aucpr(raw_df)
    pooled_df.insert(0, "dataset", dataset_name)

    return per_guide_df, pooled_df


if __name__ == "__main__":

    experiments = {
        "LOSOFineTuneCRISPRATTCell": [
            "RUNlogs/ArtificialFineTunePaper/Arti-Ws512-Epi/T_cell/",
        ],
        "LOSOFineTuneMLMTCell": [
            "RUNlogs/ArtificialFineTunePaper/CRISPert-Ws512-Epi/T_cell/",
        ],
        "LOSOFineTuneCRISPRATK562": [
            "RUNlogs/ArtificialFineTunePaper/Arti-Ws512-Epi/K562/",
        ],
        "LOSOFineTuneMLMK562": [
            "RUNlogs/ArtificialFineTunePaper/CRISPert-Ws512-Epi/K562/",
        ],
        "CrossCellTypeTest": [
            "RUNlogs/CrossCellTypeTest/Arti-Ws512+AG_ccTesting",
            "RUNlogs/CrossCellTypeTest/Arti-Ws512-AG_ccTesting",
        ],
        "CrossCellTypeTestIPSC": [
            "RUNlogs/CrossCellTypeTestIPSC/Arti-Ws512+AG_ccTesting",
            "RUNlogs/CrossCellTypeTestIPSC/Arti-Ws512-AG_ccTesting",
        ],
        "CrossCellTypeTestIVK562": [
            "RUNlogs/CrossCellTypeTestIVK562/Arti-Ws512+AG_ccTesting",
            "RUNlogs/CrossCellTypeTestIVK562/Arti-Ws512-AG_ccTesting",
        ],
        "CrossCellTypeTestMLMFull": [
            "RUNlogs/CrossCellTypeTestAblation/MLMFull-Ws512-AG_ccTesting",
            "RUNlogs/CrossCellTypeTestAblation/MLMFull-Ws512+AG_ccTesting",
        ],
        "CrossCellTypeTestMLMTCell": [
            "RUNlogs/CrossCellTypeTestAblation/MLMTCell-Ws512-AG_ccTesting",
            "RUNlogs/CrossCellTypeTestAblation/MLMTCell-Ws512+AG_ccTesting",
        ],
        "CrossCellTypeCrossAttentionAblation": [
            "RUNlogs/CrossCellTypeTestJoinAblation/Add-Ablation-Ws512-AG_ccTesting",
            "RUNlogs/CrossCellTypeTestJoinAblation/Add-Ablation-Ws512+AG_ccTesting",
        ],


    }
    negative_samples = {
        f"CrossCellTypeCrossNegative{i}": [
            f"RUNlogs/CrossCellTypeTestNegativeSampling/Arti-Ws512-AG_T_Cell_{i}_ccTesting/",
            f"RUNlogs/CrossCellTypeTestNegativeSampling/Arti-Ws512+AG_T_Cell_{i}_ccTesting/",
        ] for i in [10, 20, 30, 40, 50, "Hard"]
        
    }
    experiments = experiments | negative_samples

    per_guide_dfs = []
    pooled_dfs = []

    for dataset_name, test_dirs in experiments.items():
        per_guide_df, pooled_df = run_experiment(dataset_name, test_dirs)
        print(f"\n=== {dataset_name} : per-guide ===")
        print(per_guide_df)
        print(f"\n=== {dataset_name} : pooled (GuideID-independent) ===")
        print(pooled_df)

        per_guide_dfs.append(per_guide_df)
        pooled_dfs.append(pooled_df)

    all_per_guide = pd.concat(per_guide_dfs, ignore_index=True)
    all_pooled = pd.concat(pooled_dfs, ignore_index=True)

    per_guide_out_path = "../CRISPRPlots/AllExperiments_PerGuide.tsv"
    pooled_out_path = "../CRISPRPlots/AllExperiments_PooledByCondition.tsv"

    all_per_guide.to_csv(per_guide_out_path, sep="\t", index=False)
    all_pooled.to_csv(pooled_out_path, sep="\t", index=False)

    print(f"\nWrote per-guide table -> {per_guide_out_path}")
    print(f"Wrote pooled table -> {pooled_out_path}")