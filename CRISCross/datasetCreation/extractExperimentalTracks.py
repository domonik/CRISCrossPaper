import pandas as pd
import os
import pyBigWig
import numpy as np
from multiprocessing import Pool, set_start_method




WINDOW_SIZE = 2**14


def create_array(filename, shape):
    mmap_array = np.memmap(filename, dtype=np.float32, mode="w+", shape=shape)
    mmap_array.flush()



def fetch_data_and_write_to_array(bw_file, mmap_array, shape, chrom, start, end, ag_index, strand):
    vals = bw_file.values(chrom, int(start), int(end))
    vals = np.asarray(vals, dtype=np.float32)
    if strand == "-":
        vals = vals[::-1]
        
    mmap_array[ag_index] = vals[..., None]

    

def _mp_wrapper(file, shape, df, ag_dir, bwdir):
    name = file.split(".")[0]
    name = f"{name}"
    np_filename = os.path.join(ag_dir, f"{name}.np")
    create_array(np_filename, shape)
    file = os.path.join(bwdir, file)
    bw_file = pyBigWig.open(file)
    mmap_array = np.memmap(np_filename, dtype=np.float32, mode='r+', shape=shape)

    df.apply(lambda row: fetch_data_and_write_to_array(
        bw_file=bw_file, mmap_array=mmap_array, shape=shape, chrom=row.chr, strand=row.Strand, start=row.extStart, end=row.extEnd, ag_index=row.AlphagenomeIndex
        ), axis=1
    )
    mmap_array.flush()


def main():
    tsv = "tmpdatasets/WillsDatasetWithextendedSequences.tsv"
    df = pd.read_csv(tsv, sep="\t")
    ag_dir = "AGTensors3"
    bwdir = "BigWigTracks"
    shape = df["AlphagenomeIndex"].max() + 1, WINDOW_SIZE, 1 
    df = df[["chr", "extStart", "extEnd", "AlphagenomeIndex", "Strand"]].copy()
    calls = [(file, shape, df, ag_dir, bwdir) for file in os.listdir(bwdir) if file.endswith(".bw")]
    with Pool(16) as pool:
        res = pool.starmap(_mp_wrapper, calls)
    
        





    



if __name__ == "__main__":
    set_start_method("spawn")

    main()