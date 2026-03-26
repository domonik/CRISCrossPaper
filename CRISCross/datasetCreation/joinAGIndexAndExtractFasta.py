import pandas as pd
import numpy as np
import subprocess
from Bio import SeqIO
from io import StringIO
import os

seqlen = 2**14


def attach_sequences(bed_df, genome_fasta):
    # run bedtools getfasta
    assert os.path.exists(genome_fasta)
    result = subprocess.run(
        ["bedtools", "getfasta", "-fi", genome_fasta, "-bed", "/dev/stdin", "-s", "-name"],
        input=bed_df.to_csv(sep='\t', header=False, index=False),
        text=True,
        capture_output=True,
        check=True
    )

    # parse output fasta from stdout
    fasta_io = StringIO(result.stdout)
    seq_records = list(SeqIO.parse(fasta_io, "fasta"))

    # extract mapping from ID to sequence
    seq_map = {int(rec.id.split(":")[0]): str(rec.seq) for rec in seq_records}

    # map sequences back
    bed_df["extended_off_target"] = bed_df["ID"].map(seq_map)
    return bed_df

if __name__ == "__main__":
    ag_info = pd.read_csv("tmpdatasets/AlphagenomeIndex.tsv", sep="\t")
    cols= ["Assembly", "Guide_sequence", "Query term", "chr", "start", "end", "Strand", "AlphagenomeIndex"]
    ag_info = ag_info[cols]
    df = pd.read_csv("tmpdatasets/WillsDatasetFixed.tsv", sep="\t")
    df = df.merge(ag_info, on=cols[:-1], how="left")
    df["GuideID"] = df.groupby(["Assembly", "Query term", "Guide_sequence", "PMID"], sort=False).ngroup()

    df["center"] = (df["end"] + df["start"]) // 2
    df["extStart"] = df["center"] - seqlen // 2
    df["extEnd"] = df["center"] + seqlen // 2
    df["ID"] = np.arange(len(df))
    df["ID2"] = df["ID"]
    bed_dfs = []
    for assembly in ["hg38"]:
        bed_df = df[df["Assembly"] == assembly].copy()

        bed_df = bed_df[["chr", "extStart", "extEnd", "ID", "ID2", "Strand"]]
        fasta = f"GRC{assembly[0]}38.primary_assembly.genome.fa"
        bed_df = attach_sequences(bed_df, fasta)
        bed_dfs.append(bed_df[["ID", "extended_off_target"]])
    bed_df = pd.concat(bed_dfs, ignore_index=True)
    df = df.merge(bed_df, on="ID")
    df = df.drop("ID", axis=1)
    df.to_csv("tmpdatasets/WillsDatasetWithextendedSequences.tsv", sep="\t", index=False)

