import numpy as np
import pandas as pd
import os
from io import StringIO
import subprocess
from Bio import SeqIO

seqlen = 2**14




def main():
    
    tcdf = "datasets/TCellDatasetWithCorrectCoords.tsv"

    tcdf = pd.read_csv(tcdf, sep="\t")
    tcdf["epiDir"] = "AGTensorsCL:0000624" 

    
    kagindex = "datasets/K562WithEpidir.tsv"
    kagindex =  pd.read_csv(kagindex, sep="\t")
    kagindex["cell_type"] = "K562"
    kagindex["epiDir"] = "AGTensorsEFO:0002067"

    tcdf["cell_type"] = "TCell"
    fulldf = pd.concat((tcdf, kagindex))

    fulldf.to_csv("datasets/TcellAndK562StackedCorrCoords.tsv", sep="\t", index=False)
    
    
    
if __name__ == "__main__":
    main()