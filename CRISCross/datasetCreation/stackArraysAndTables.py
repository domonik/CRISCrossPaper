import numpy as np
import pandas as pd
import os
from io import StringIO
import subprocess
from Bio import SeqIO

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


def main():
    
    tcdf = "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv"

    tcdf = pd.read_csv(tcdf, sep="\t")
    tcdf["epiDir"] = "AGTensorsCL:0000624" 

    
    kagindex = 'tmpdatasets/AlphagenomeIndexK562.tsv'
    kagindex =  pd.read_csv(kagindex, sep="\t")
    kagindex["cell_type"] = "K562"
    kagindex["epiDir"] = "AGTensorsEFO:0002067"

    tcdf["cell_type"] = "TCell"

    tcell_shape = (len(tcdf), 2**14, 1)
    k562_shape = (len(kagindex), 2**14, 1)
    
    kagindex["AlphagenomeIndex"] = kagindex["AlphagenomeIndex"] + 1 + tcdf["AlphagenomeIndex"].max()

    kagindex["center"] = (kagindex["end"] + kagindex["start"]) // 2
    kagindex["extStart"] = kagindex["center"] - seqlen // 2
    kagindex["extEnd"] = kagindex["center"] + seqlen // 2
    
    kagindex["ID"] = np.arange(len(kagindex))
    kagindex["ID2"] = kagindex["ID"]
    bed_dfs = []
    for assembly in ["hg38"]:
        bed_df = kagindex[kagindex["Assembly"] == assembly].copy()

        bed_df = bed_df[["chr", "extStart", "extEnd", "ID", "ID2", "Strand"]]
        fasta = f"GRC{assembly[0]}38.primary_assembly.genome.fa"
        bed_df = attach_sequences(bed_df, fasta)
        bed_dfs.append(bed_df[["ID", "extended_off_target"]])
    bed_df = pd.concat(bed_dfs, ignore_index=True)
    kagindex = kagindex.merge(bed_df, on="ID")
    fulldf = pd.concat((tcdf, kagindex))

    # load fully into RAM
    tcell_arr = np.memmap(
        "AGTensors3/ATAC.np", 
        dtype=np.float32,
        mode="r",
        shape=(tcell_shape)
    )   # or .np if that's your file

    k562_arr = np.memmap(
            "AGTensorsK562/ATAC.np", 
            dtype=np.float32,
            mode="r",
            shape=(k562_shape)
        )
    # Load fully into RAM
    tcell_loaded = np.array(tcell_arr)
    k562_loaded = np.array(k562_arr)
    full = np.concat((tcell_loaded, k562_loaded))
    # Shuffle all values
    shape = (len(tcdf) + len(kagindex), 2**14, 1)
    os.makedirs("StackedAGTensors", exist_ok=True)
    # Write to new memmap
    out = np.memmap(
        "StackedAGTensors/ATAC.np",
        dtype=np.float32,
        mode="w+",
        shape=full.shape,
    )

    out[:] = full
    out.flush()
    fulldf.to_csv("tmpdatasets/TcellAnK562Stacked.tsv", sep="\t", index=False)
    assert fulldf["AlphagenomeIndex"].max() == full.shape[0] -1
    
    
    
if __name__ == "__main__":
    main()