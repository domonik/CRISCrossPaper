
import numpy as np
import pandas as pd
import os


df = "datasets/TCellDatasetWithextendedSequencesAndIDs.tsv"

df = pd.read_csv(df, sep="\t")


np.random.seed(42)
shape = (len(df), 2**14, 1)

# load fully into RAM
arr = np.memmap(
    "AGTensors3/ATAC.np", 
    dtype=np.float32,
    mode="r",
    shape=(shape)
)   # or .np if that's your file

# Load fully into RAM
loaded = np.array(arr)

# Shuffle all values
flat = loaded.ravel()
np.random.shuffle(flat)

os.makedirs("ShuffledAGTensors", exist_ok=True)
# Write to new memmap
out = np.memmap(
    "ShuffledAGTensors/ATAC.np",
    dtype=np.float32,
    mode="w+",
    shape=shape,
)


out[:] = loaded
out.flush()
