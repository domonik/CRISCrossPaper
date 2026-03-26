import pandas as pd
import os

def main():
    tsv = "../datasets/Tcell_guideseq_28sgRNA_reducedIR40.csv"
    agindex = "../datasets/AlphagenomeInfo.csv"
    ag_index = pd.read_csv(agindex, sep="\t")
    tcelldf = pd.read_csv(tsv, sep=",")
    tcelldf["Technology"] = "GUIDE-seq"
    tcelldf["Source"] = "T cells"
    tcelldf["PMID"] = 32541958
    tcelldf["On_target_site"] = pd.NA
    tcelldf["Assembly"] = "hg38"
    tcelldf = tcelldf.rename(
        {"offtarget_sequence": "Target_sequence", "target": "Guide_sequence", "chrom": "chr", "chromStart": "start", "chromEnd": "end", "strand": "Strand", "distance": "Identity", "GUIDEseq_reads": "Score"},
          axis=1)

    tcelldf.to_csv("Tcell_guideseq_28sgRNA_reducedIR40Fixed.tsv", sep="\t", index=False)
    tcelldf = tcelldf.rename(
        {"offtarget_sequence": "Target_sequence", "target": "Guide_sequence", "chrom": "chr", "chromStart": "start", "chromEnd": "end", "strand": "Strand", "distance": "Identity", "GUIDEseq_reads": "Score"},
          axis=1)
    
    df = tcelldf
    ag_mapping = pd.read_csv("../datasets/cell_type_mapping.tsv", sep="\t")
    ag_mapping = ag_mapping[~pd.isna(ag_mapping["Match in database"])]
    df = df.merge(ag_mapping, right_on="Query term", left_on="Source", how="left")
    summ = ag_mapping.merge(ag_index, left_on="Match in database", right_on="biosample_name", how="left")
    summ2 = summ.groupby("Query term")["output_type"].agg(list)
    df = df.merge(summ2, on="Query term", how="left")
    os.makedirs("tmpdatasets", exist_ok=True)
    df.to_csv("tmpdatasets/Tcell_guideseq_28sgRNA_reducedIR40Fixed.tsv", sep="\t", index=False)




if __name__ == "__main__":
    main()