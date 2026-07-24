import pandas as pd


def main():
    # --- CRISCross tensorboard results ---
    file1 = "../Results/CRISCrossTensorboard_summary.tsv"
    df1 = pd.read_csv(file1, sep="\t")

    df1.loc[df1["mode"].isna(), "mode"] = "Base"
    df1["model"] = "CRISCross"
    df1 = df1.rename({"test_auprc": "AUCPR"}, axis=1)
    df1["mode"] = df1["mode"].apply(
        lambda x: x if "abl" not in x else "-Histone" if "Histone" in x else "-ATAC"
    )
    df1 = df1[df1["run"].str.contains("ccFineTuning") != True]

    # --- Old model results (unified table) ---
    file_old = "../Results/OldModelsUnified.tsv"
    df_old = pd.read_csv(file_old, sep="\t")

    # --- Concatenate ---
    df = pd.concat([df1, df_old], ignore_index=True)
    df["pretrain"] = df["mode"].apply(
        lambda x: not ("NoPretrain" in str(x) or "no pretrain" in str(x))
    )
    df = df.sort_values(["model", "dataset", "mode", "seed", "split"]).reset_index(drop=True)

    df.to_csv("../Results/FinalSummary.tsv", sep="\t", index=False)
    print(f"Wrote {len(df)} rows to ../Results/FinalSummary.tsv")


if __name__ == "__main__":
    main()
