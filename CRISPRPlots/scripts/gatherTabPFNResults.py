import re
import pandas as pd

def parse_results(filepath):
    rows = []
    current_section = None
    current_seed = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Section header (lines starting with ##)
            if line.startswith("##"):
                current_section = line.strip("# ").strip()
                continue

            # Seed line
            seed_match = re.match(r"Seed\s+(\d+)", line)
            if seed_match:
                current_seed = int(seed_match.group(1))
                continue

            # Score lines (sgX: value)
            score_match = re.match(r"(sg\d+):\s+([0-9.]+)", line)
            if score_match and current_section is not None and current_seed is not None:
                sg = score_match.group(1)
                score = float(score_match.group(2))
                rows.append({
                    "section": current_section,
                    "seed": current_seed,
                    "split": sg,
                    "AUCPR": score
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    
# Example usage
    df = parse_results("Tabpfn_results.txt")
    mask = df.section.str.contains("(baseline)", regex=False)
    df.loc[mask, "Epi"] = "Baseline"
    mask = df.section.str.contains("ag_epi", regex=False)
    df.loc[mask, "Epi"] = "AG"
    mask = df.section.str.contains("ex_epi", regex=False)
    df.loc[mask, "Epi"] = "EX"
    breakpoint()
    df.to_csv("TabPFNResults.tsv", sep="\t", index=False)