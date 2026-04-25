import pandas  as pd


grouped_unique_values = [['sg28', 'sg23', 'sg3'], ['sg2', 'sg24', 'sg8'], ['sg1', 'sg6', 'sg7'],
                        ['sg12', 'sg18', 'sg10'], ['sg13', 'sg26', 'sg5'], ['sg19', 'sg16']]
grouped_unique_values = [[g] for glist in grouped_unique_values for g in glist]


def main():
    file1 = "../Results/CRISCrossTensorboard_summary.tsv"
    file2 = "../Results/CRISPert_nopretraining.csv"
    file3 = "../Results/ex_vs_base.csv"
    
    df1 = pd.read_csv(file1, sep="\t")
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    df2 = df2.rename(
        {
            "group": "split",
            "AUPRC": "AUCPR"
            
        },
        axis=1
    )
    
    df2['split'] = df2['split'].apply(lambda x: next(i for i, sublist in enumerate(grouped_unique_values) if x in sublist))   
    df3 = df3.rename(
        {
            "group": "split",
            "AUPRC": "AUCPR",
            "NAME": "model"
            
        },
        axis=1
    )
    df3['split'] = df3['split'].apply(lambda x: next(i for i, sublist in enumerate(grouped_unique_values) if x in sublist))   

    df3  = df3[(df3["model"] == "CRISPPert") & ~df3["side_features"]]
    df3["model"] = "CRISPert"
    df3["mode"] = "CRISPert original"
    df3["window_size"] = 23
    df3["dataset"] = "T-cell"


    df1.loc[df1["mode"].isna(), "mode"] = "Base"

    df1["model"] = "CRISCross"
    df2["model"] = "CRISPert"
    df2["window_size"] = 23
    df2["dataset"] = "T-cell"
    df2["mode"] = "CRISPert no pretrain"
    df2 = df2.drop(
        [
            "running_overall_mean",
        "mean_AUPRC_seed"
        ],
        axis=1
    )
    df3 = df3.drop(
        [
        "mean_AUPRC_seed"
        ],
        axis=1
    )
    df1 = df1.rename(
        {
            "test_auprc": "AUCPR",
            
        },
        axis=1
    )
    df1["mode"] = df1["mode"].apply(lambda x: x if "abl" not in x else "-Histone" if "Histone" in x else "-ATAC")
    df = pd.concat((df1, df2, df3))
    df["pretrain"] = df["mode"].apply(lambda x: not("NoPretrain" in x or "no pretrain" in x))
    df = df.sort_values(["seed", "split"])

    df.to_csv("../Results/FinalSummary.tsv", sep="\t", index=False)
    
    
    
    
if __name__ == "__main__":
    main()