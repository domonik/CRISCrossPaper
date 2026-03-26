
import pandas as pd
import numpy as np

seqlen = 2**14

def main():
    file = "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv"
    df = pd.read_csv(file, sep="\t")
    
    df.loc[df["Strand"] == "+", "start"] = df.loc[df["Strand"] == "+", "start"] -1
    df.loc[df["Strand"] == "+", "end"] = df.loc[df["Strand"] == "+", "end"] -1
    df.to_csv("datasets/TCellDatasetWithCorrectCoords.tsv", sep="\t", index=False)






if __name__ == "__main__":
    main()