#!/usr/bin/env python3
"""
Collect AUCPR per GuideID from Arti-Ws512 ccTesting results.

Walks Arti-Ws512+AG_ccTesting and Arti-Ws512-AG_ccTesting,
finds every final_predictions.tsv, extracts GuideID/label/predictions,
and computes AUCPR (average precision) per GuideID per seed per AG-condition.

Output: TSV with columns  condition  seed  GuideID  aucpr
"""

import glob
import os
import sys

import numpy as np
import pandas as pd


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


def extract_seed(dirname: str) -> str:
    """Extract seed value from directory name like 'K562Test_seed126' -> '126'."""
    return dirname.split("_seed")[-1]


def extract_condition(top_dir: str) -> str:
    """Extract +AG / -AG from parent directory name."""
    if "+AG" in top_dir:
        return "+AG"
    elif "-AG" in top_dir:
        return "-AG"
    return "unknown"


def find_predictions(base: str) -> list[str]:
    """Recursively find all final_predictions.tsv under base."""
    pattern = os.path.join(base, "**", "final_predictions.tsv")
    return glob.glob(pattern, recursive=True)


def main(test_dirs):
    results = []

    # Only ccTesting directories


    for test_dir in test_dirs:
        condition = extract_condition(test_dir)
        files = find_predictions(test_dir)
        print(f"{condition}: found {len(files)} prediction files")

        for fpath in files:
            # Determine seed from the parent chain
            parts = fpath.split(os.sep)
            seed_dirname = None
            for p in parts:
                print(p)
                if "Test_seed" in p or p.startswith("T_cell_"):
                    seed_dirname = p
                    break
            if seed_dirname is None:
                print(f"WARNING: Could not determine seed from path: {fpath}", file=sys.stderr)
                continue
            seed = extract_seed(seed_dirname)

            df = pd.read_csv(fpath, sep="\t", index_col=0)

            for guide_id, group in df.groupby("GuideID"):
                labels = group["label"].values.astype(float)
                preds = group["predictions"].values.astype(float)

                n_pos = int(labels.sum())
                n_total = len(labels)

                # Need at least one positive and one negative for a meaningful PR curve
                if n_pos == 0 or n_pos == n_total:
                    aucpr = float("nan")
                else:
                    aucpr = average_precision_score(labels, preds)

                results.append({
                    "condition": condition,
                    "seed": int(seed),
                    "GuideID": int(guide_id),
                    "aucpr": aucpr,
                })

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values(["condition", "seed", "GuideID"]).reset_index(drop=True)


    return out_df


if __name__ == "__main__":
    
    test_dirs = [
        "RUNlogs/CrossCellTypeTest/Arti-Ws512+AG_ccTesting",
        "RUNlogs/CrossCellTypeTest/Arti-Ws512-AG_ccTesting",
    ]
    df = main(test_dirs)
    print(df)
    out_path = "../CRISPRPlots/CrossCellTypeTestK562PerGuide.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    
    test_dirs = [
        "RUNlogs/CrossCellTypeTestIPSC/Arti-Ws512+AG_ccTesting",
        "RUNlogs/CrossCellTypeTestIPSC/Arti-Ws512-AG_ccTesting",
    ]
    df = main(test_dirs)
    print(df)
    out_path = "../CRISPRPlots/CrossCellTypeTestIPSCPerGuide.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    
    test_dirs = [
        "RUNlogs/CrossCellTypeTestIVK562/Arti-Ws512+AG_ccTesting",
        "RUNlogs/CrossCellTypeTestIVK562/Arti-Ws512-AG_ccTesting",
    ]
    df = main(test_dirs)
    print(df)
    out_path = "../CRISPRPlots/CrossCellTypeTestIPSCPerGuide.tsv"
    df.to_csv(out_path, sep="\t", index=False)

