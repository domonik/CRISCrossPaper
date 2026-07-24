import pandas as pd


def main(df):
    df.loc[df["Strand"] == "+", "start"] = df.loc[df["Strand"] == "+", "start"] + 1
    df.loc[df["Strand"] == "+", "end"] = df.loc[df["Strand"] == "+", "end"] + 1
    df.to_csv("tmpdatasets/K562WithWrongCoordsForOldPipeline.tsv", sep="\t", index=False)


if __name__ == "__main__":
    df = pd.read_csv("datasets/K562WithIDs.tsv", sep="\t")
    main(df)