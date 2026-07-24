import pandas as pd
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from plotly_template import COLORS


def main():
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    df2 = df[(df["comparison"] == "IPSC") & (df["window_size"] == 512)]
    df4 = df[(df["comparison"] == "CrossCellTypeTestNormCorr2") & (df["window_size"] == 23)]

    df3 = df[
        (df["dataset"] == "K562") & (df["mode"] == "Arti-Ws512-AG_ccTesting") |
        (df["dataset"] == "K562") & (df["mode"] == "Arti-Ws512+AG_ccTesting") & (df["run"].str.contains("AG")) 
    ]
    fdf = pd.concat((df2, df3, df4))
    fdf = fdf[["AUCPR", "mode", "comparison", "seed", "split", "dataset"]]
    res = fdf.groupby(["mode", "dataset", "comparison"])["AUCPR"].mean().reset_index()
    res["Feature"] = res["mode"].apply(lambda x: "Sequence only" if "-AG" in x else "Sequence + ATAC-seq")
    res["test dataset"] = res["dataset"]
    res["train dataset"] = "T-Cell"
    res["pretrain epigenetic tracks"] = res["comparison"].apply(lambda x: "T-Cell" if "ThreeTrack" in x else "T-Cell, K562, IPSC")
    res = res[["train dataset", "test dataset", "Feature", "pretrain epigenetic tracks", "AUCPR"]]
    res = res.sort_values(["test dataset", "Feature"])
    res.to_csv("TableForDinh.tsv", sep="\t", index=False)
    breakpoint()


if __name__ == "__main__":
    main()