import pandas as pd
import os

def main():
    pickle = "AGjoined.pckl"
    df = pd.read_pickle(pickle)
    dir = "negatives"
    df.output_type = df.output_type.apply(lambda y: ", ".join(y))
    df = df.drop("output_type", axis=1)
    mapping = {row["Guide_sequence"]: row["Target_sequence"] for _, row in df[(df["Guide_sequence"].str[-3:] == "NGG") & (df["Identity"] == 0)].iterrows()}
    df.loc[(df["Guide_sequence"].str[-3:] == "NGG"), "Guide_sequence"] = df.loc[(df["Guide_sequence"].str[-3:] == "NGG"), "Guide_sequence"].map(mapping)

    for file in os.listdir(dir):
        ff = os.path.join(dir, file)
        read_df = pd.read_csv(ff)
        guide = file.split("_")[-1].split(".")[0]
        assembly = file.split("_")[0]

        query_terms = df[df["Guide_sequence"].str[:-3] == guide[:-3]]["Query term"].unique().tolist()
        guides = df[df["Guide_sequence"].str[:-3] == guide[:-3]]["Guide_sequence"].unique().tolist()
        for guide in guides:
            for query_term in query_terms:
                ndf = read_df.copy()
                row = df[(df["Assembly"] == assembly) & (df["Query term"] == query_term) & (df["Guide_sequence"] == guide)].iloc[0]

                guide = guide
                ndf["start"] = ndf["EndPosition"] - 23
                ndf["Guide_sequence"] = guide
                ndf["Technology"] = dir
                ndf["Source"] = dir
                ndf["PMID"] = dir
                ndf["Assembly"] = assembly
                ndf["Query term"] = query_term
                ndf["Match in database"] = row["Match in database"]
                ndf["Reason for match"] = row["Reason for match"]
                ndf["On_target_site"] = row["On_target_site"]
                ndf["Score"] = 0
                ndf["Score_norm"] = 0
                ndf["on_target_reads"] = 0
                ndf["Score_norm"] = ndf["Score_norm"].apply(lambda x: [x])
                ndf["Score"] = ndf["Score"].apply(lambda x: [x])
                ndf["on_target_reads"] = ndf["on_target_reads"].apply(lambda x: [x])
                ndf["chr"] = ndf["Chromosome"].str.split("_").str[0]
                ndf = ndf.rename({
                    "EndPosition": "end",
                    "AlignedText": "Target_sequence",
                    "#Mismatches": "Identity"
                    
                }, axis=1)
                ndf = ndf[df.columns]
                df = pd.concat((df, ndf), ignore_index=True)
    df.drop_duplicates(["Query term", "Technology", "Assembly", "chr", "start", "end", "Strand", "Guide_sequence"])
    print(len(df))
    df.to_csv("NegativesJoined.tsv", sep="\t", index=False)



if __name__ == "__main__":
    main()