import os
import numpy as np
import pyBigWig
import multiprocessing

# ----------------------------
# CONFIG
# ----------------------------
bigwig_dir = "EX_BigWigs"     # folder containing .bw files
out_dir = "BigWigMemmapsPerChr" # folder to store per-chromosome memmaps
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# FUNCTION TO CONVERT ONE FILE
# ----------------------------
def convert_bigwig_to_memmap_per_chr(bw_path, out_dir):
    fname = os.path.splitext(os.path.basename(bw_path))[0]
    print(f"Processing {bw_path}")

    with pyBigWig.open(bw_path) as bw:
        chroms = bw.chroms()  # dict: {chrom_name: length}

        for chrom, length in chroms.items():
            memmap_path = os.path.join(out_dir, f"{fname}_{chrom}.npy")
            print(f"  Writing {memmap_path} ({length} bases)")

            # create memmap array
            arr = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(length,))

            # load chromosome values in one call
            vals = bw.values(chrom, 0, length, numpy=True)
            vals = np.nan_to_num(vals, nan=0.0)  # replace NaN with 0
            arr[:] = vals

            arr.flush()
    print(f"Done: {bw_path}\n")

# ----------------------------
# MAIN LOOP
# ----------------------------
calls = []

bw_files = [
    os.path.join(bigwig_dir, f)
    for f in os.listdir(bigwig_dir)
    if f.endswith(".bw")
]
args = [(bw_path, out_dir) for bw_path in bw_files]

num_workers = min(len(bw_files), multiprocessing.cpu_count())
with multiprocessing.Pool(num_workers) as pool:
    pool.starmap(convert_bigwig_to_memmap_per_chr, args)

print("All files converted to per-chromosome memmaps.")
