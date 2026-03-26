import pandas as pd
import os 

if __name__ == "__main__":
    grouped_unique_values = [['sg28', 'sg23', 'sg3'], ['sg2', 'sg24', 'sg8'], ['sg1', 'sg6', 'sg7'],
                         ['sg12', 'sg18', 'sg10'], ['sg13', 'sg26', 'sg5'], ['sg19', 'sg16']]
    grouped_unique_values = [[g] for glist in grouped_unique_values for g in glist]
    df = pd.read_csv("tmpdatasets/WillsDatasetWithextendedSequences.tsv", sep="\t")
    old_df = pd.read_csv("../datasets/Tcell_guideseq_28sgRNA_reducedIR40.csv")
    old_df = old_df.rename({"strand": "Strand", "target": "Guide_sequence"}, axis=1)
    cols = ["chr", "start", "end",  "Strand", "Guide_sequence", "ID"]
    old_df = old_df[cols]
    old_df.loc[df["Strand"] == "+", "start"] = old_df.loc[old_df["Strand"] == "+", "start"] + 1
    old_df.loc[df["Strand"] == "+", "end"] = old_df.loc[old_df["Strand"] == "+", "end"] + 1
    df_new = df.merge(old_df, on=cols[:-1])
    print(len(df_new))
    df_new["GuideID"] = df_new["ID"]
    df_new = df_new.rename({"offtarget": "Target_sequence"}, axis=1)
    os.makedirs("datasets", exist_ok=True)

    df_new.to_csv("datasets/TCellDatasetWithextendedSequencesAndIDs.tsv", sep="\t")
    split_df = {
        "val_set": [],
        "test_set": [],
        "exclude": []
    }
    unique_guides = df_new["GuideID"].unique().tolist()
    for group in grouped_unique_values:
        split_df["val_set"].append([])
        split_df["test_set"].append(group)
        split_df["exclude"].append([])
    split_df = pd.DataFrame(split_df)
    os.makedirs("runSettings", exist_ok=True)
    split_df.to_csv("runSettings/RunSettingsLeaveOneOut.tsv", sep="\t")

