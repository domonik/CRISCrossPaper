import re
from typing import List, Dict, Any
import pandas as pd

MODEL_RE = re.compile(r"#+\s*([A-Za-z0-9_+-]+)\s*#+")
SIDE_RE = re.compile(r"Side Features\s*=\s*(True|False)")
SEED_RE = re.compile(r"Seed\s+(\d+)")
PAIR_RE = re.compile(r"\s+([\w_]+):\s+([0-9.]+)")
COMPARISON_RE = re.compile(r"#+\s*([A-Za-z0-9_ ]+ vs base)\s*#+")

def fix_modes(df: pd.DataFrame) -> pd.DataFrame:
    # start with a copy
    df = df.copy()
    
    # Initialize mode column
    df['mode'] = None

    # AG vs base
    mask_ag = df['comparison'] == 'AG vs base'
    df.loc[mask_ag & (df['side_features'] == True), 'mode'] = 'AG'
    df = df[~(mask_ag & (df['side_features'] == False))]


    # Ex vs base
    mask_ex = df['comparison'] == 'Ex vs base'
    df.loc[mask_ex & (df['side_features'] == True), 'mode'] = 'Ex'
    df.loc[mask_ex & (df['side_features'] == False), 'mode'] = 'Base'

    # drop side_features=False rows for Ex vs base

    return df.reset_index(drop=True)

def parse_results_with_comparison(file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    current_model = None
    current_comparison = None
    current_side = None
    current_seed = None
    with open(file) as handle:
        for line in handle:
            line = line.rstrip()

            # model name
            m = MODEL_RE.match(line)
            if m:
                current_model = m.group(1)
                continue

            # comparison header (AG vs base, Ex vs base, etc.)
            m = COMPARISON_RE.match(line)
            if m:
                current_comparison = m.group(1)
                continue

            # side features
            m = SIDE_RE.search(line)
            if m:
                current_side = (m.group(1) == "True")
                continue

            # seed
            m = SEED_RE.search(line)
            if m:
                current_seed = int(m.group(1))
                continue

            # per-pair AUPRC
            m = PAIR_RE.match(line)
            if m and current_model is not None and current_seed is not None:
                pair, value = m.groups()
                rows.append({
                    "model": current_model,
                    "comparison": current_comparison,
                    "side_features": current_side,
                    "seed": current_seed,
                    "pair": pair,
                    "auprc": float(value),
                })

    return rows


if __name__ == "__main__":

    rows = parse_results_with_comparison("overall_results_ex_leaveOneOut.txt")
    df = pd.DataFrame(rows)
    df = fix_modes(df)
    df["mode"] = df["side_features"].map(lambda y: "Base" if y is True else "Ex")
    print(df)
    df.to_csv("results_table.tsv", sep="\t", index=False)
