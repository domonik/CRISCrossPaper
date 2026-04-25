import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_template import PT8


CHUNK_SIZE = 1000
WINDOW_SIZE = 2 ** 14
CENTER = WINDOW_SIZE // 2
AGDIR = "../CRISCross/AGTensors3/"
_default_size = (2**14, 1)

EPI_SHAPES = {
            "ATAC": _default_size,
            #"DNASE": _default_size, 
            #"RNA_SEQ": (2 ** 14, 4),
            "EX_ATAC": _default_size,
            "EX_H3K4me1": _default_size,
            "EX_H3K4me3": _default_size,
            "EX_H3K9ac": _default_size,
            "EX_H3K9me3": _default_size,
            "EX_H3K27ac": _default_size,
            "EX_H3K27me3": _default_size,
            "EX_H3K36me3": _default_size,
            "CHIP_HISTONE": (128, 1)

        }
_default_size = (2**14, 1)
histone_size = (128, 1)

EPI_FEATURES = {
            "ATAC": _default_size,
            "DNASE": _default_size, 
            "EX_ATAC": _default_size,
            "EX_H3K4me1": _default_size,
            "EX_H3K4me3": _default_size,
            "EX_H3K9ac": _default_size,
            "EX_H3K9me3": _default_size,
            "EX_H3K27ac": _default_size,
            "EX_H3K27me3": _default_size,
            "EX_H3K36me3": _default_size,
            #'+_polyA plus RNA-seq': _default_size,
            '+_total RNA-seq': _default_size,
            #'-_polyA plus RNA-seq': _default_size,
            '-_total RNA-seq': _default_size,
            'H3K27ac': histone_size,
            'H3K27me3': histone_size,
            'H3K36me3': histone_size,
            'H3K4me1': histone_size,
            'H3K4me3': histone_size,
            'H3K9me3': histone_size,


        }

FEATURES = ["EX_ATAC", "EX_H3K4me3", "ATAC", "H3K4me3"]



def generate_data(tsv, outfile_postfix):
    df = pd.read_csv(tsv, sep="\t")
    indices = df[df.label == 1]["AlphagenomeIndex"].values
    neg_indices = df[df.label == 0]["AlphagenomeIndex"].values
    mask = np.zeros(len(df))
    mask[indices] = 1
    full_values = np.zeros((10, WINDOW_SIZE))
    chip_vals = np.zeros((10, EPI_SHAPES["CHIP_HISTONE"][0]))
    for feature in FEATURES:
        if feature.startswith("H3"):
            file = os.path.join(AGDIR, f"{feature}.np")
            shape = EPI_SHAPES["CHIP_HISTONE"]
            mmap = np.memmap(file, dtype=np.float32, mode="r", shape=(len(df), *shape))
            window_size = shape[0]
            values = chip_vals
            feat_dim = 0
 
        else:
            file = os.path.join(AGDIR, f"{feature}.np")
            shape = EPI_SHAPES[feature]
            mmap = np.memmap(file, dtype=np.float32, mode="r", shape=(len(df), *shape))
            window_size = WINDOW_SIZE
            values = full_values
            feat_dim = 0

        outfile = f"GenomicSummary/{feature}_{outfile_postfix}"
        
        for i in range(0, window_size, CHUNK_SIZE):
            cur_val = mmap[:, i:i+CHUNK_SIZE, feat_dim]
               
            pos_val = cur_val[indices]
            neg_val = cur_val[neg_indices]
            
            
            p_mean = np.nanmean(pos_val, axis=0)
            p_median = np.nanmedian(pos_val, axis=0)
            p_q25 = np.nanquantile(pos_val,q=0.25, axis=0)
            p_q75 = np.nanquantile(pos_val,q=0.75, axis=0)
            p_std = np.nanstd(pos_val, axis=0)
            
            

            values[0, i:i+CHUNK_SIZE] = p_mean
            values[1, i:i+CHUNK_SIZE] = p_median
            values[2, i:i+CHUNK_SIZE] = p_std
            values[3, i:i+CHUNK_SIZE] = p_q25
            values[4, i:i+CHUNK_SIZE] = p_q75

            n_mean = np.nanmean(neg_val, axis=0)
            n_median = np.nanmedian(neg_val, axis=0)
            n_q25 = np.nanquantile(neg_val,q=0.25, axis=0)
            n_q75 = np.nanquantile(neg_val,q=0.75, axis=0)
            n_std = np.nanstd(neg_val, axis=0)

            values[5, i:i+CHUNK_SIZE] = n_mean
            values[6, i:i+CHUNK_SIZE] = n_median
            values[7, i:i+CHUNK_SIZE] = n_std
            values[8, i:i+CHUNK_SIZE] = n_q25
            values[9, i:i+CHUNK_SIZE] = n_q75

        np.save(outfile, values)


HISTONEWINDOWS = [23] + [2**x for x  in range(7, 14)]



def calc_stat_per_row_ag_histone(row, memmap, shape):
    feat = memmap[row["AlphagenomeIndex"]]
    center = shape[0] // 2

    out = []

    for window in HISTONEWINDOWS:
        half = (window // 2) // 128
        if half > 0:
            start = center - half
            end   = center + half
        else:
            start = center
            end = center + 1

        win = feat[start:end]

        stats = np.array([
            np.nanmean(win),
            np.nanmedian(win),
            np.nanmax(win),
            np.nanmin(win),
        ])

        out.append(stats)

    return np.concatenate(out)   


def calc_stat_per_row_atac(row, memmap, shape):
    feat = memmap[row["AlphagenomeIndex"]]
    center = shape[0] // 2

    out = []

    for window in HISTONEWINDOWS:
        start = center - window//2 - window % 2
        end   = center + window//2
        win = feat[start:end]

        stats = np.array([
            np.nanmean(win),
            np.nanmedian(win),
            np.nanmax(win),
            np.nanmin(win),
        ])
        out.append(stats)

    return np.concatenate(out)

def compute_feature_block(indices, memmap, shape, histone=False):
    center = shape[0] // 2
    feats = memmap[indices]  # loads only required rows

    outputs = []

    for window in HISTONEWINDOWS:
        if histone:
            half = (window // 2) // 128
            if half > 0:
                start = center - half
                end   = center + half
            else:
                start = center
                end = center + 1
        else:
            start = center - window//2 - window % 2
            end   = center + window//2

        win = feats[:, start:end]

        stats = np.stack([
            np.nanmean(win, axis=1),
            np.nanmedian(win, axis=1),
            np.nanmax(win, axis=1),
            np.nanmin(win, axis=1),
        ], axis=1)

        outputs.append(stats)

    return np.concatenate(outputs, axis=1)


def _compute_feature(args):
    feature, df = args
    file = os.path.join(AGDIR, f"{feature}.np")
    shape = EPI_FEATURES[feature]
    mmap = np.memmap(file, dtype=np.float32, mode="r", shape=(len(df), *shape))

    cols = [
        f"{feature}_{hfeat}_{window_size}"
        for window_size in HISTONEWINDOWS
        for hfeat in ["mean", "median", "max", "min"]
    ]

    if feature.startswith("H3"):
        func = calc_stat_per_row_ag_histone
    else:
        func = calc_stat_per_row_atac

    data = df.apply(
        lambda row: pd.Series(func(row, mmap, shape), index=cols),
        axis=1
    )
    print(f"finished: {feature}")
    return data

from multiprocessing import Pool, cpu_count

def generate_window_stats(tsv):
    df = pd.read_csv(tsv, sep="\t")
    df = df[["chr", "start", "end", "Strand", "Guide_sequence",
             "ID", "label", "AlphagenomeIndex"]]
    n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
    with Pool(n_cpus) as pool:
        results = pool.map(_compute_feature, [(f, df) for f in EPI_FEATURES.keys()])

    final = pd.concat([df] + results, axis=1)
    final.to_csv("SummaryStatsPerWinsize.tsv", sep="\t", index=False)



def process_batch(args):


    start, end, indices, file, df_len, shape, histone, feature = args
    print(f"start: {feature}_{start}")

    mmap = np.memmap(
            file,
            dtype=np.float32,
            mode="r",
            shape=(df_len, *shape)
        )
    stats = compute_feature_block(indices, mmap, shape, histone)
    stats = np.squeeze(stats)  # ensure 2D
    return stats

from multiprocessing import set_start_method

set_start_method("spawn", force=True)

def generate_window_stats(tsv, batch_size=5000, n_cpus=4):
    df = pd.read_csv(tsv, sep="\t")
    df = df[["chr","start","end","Strand","Guide_sequence",
             "ID","label","AlphagenomeIndex"]]

    all_feature_dfs = []

    for feature in EPI_FEATURES.keys():
        file = os.path.join(AGDIR, f"{feature}.np")
        shape = EPI_FEATURES[feature]



        histone = feature.startswith("H3")

        cols = [
            f"{feature}_{hfeat}_{window}"
            for window in HISTONEWINDOWS
            for hfeat in ["mean", "median", "max", "min"]
        ]

        # Prepare arguments for Pool
        batch_args = []
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            indices = df["AlphagenomeIndex"].values[start:end]
            batch_args.append((start, end, indices, file, len(df), shape, histone, feature))

        # Multiprocessing
        n_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
        print(f"using {n_cpus}")
        with Pool(processes=n_cpus) as pool:
            feature_blocks = pool.map(process_batch, batch_args)

        feature_matrix = np.vstack(feature_blocks)
        feature_df = pd.DataFrame(feature_matrix, columns=cols)
        all_feature_dfs.append(feature_df)

    final = pd.concat([df.reset_index(drop=True)] + all_feature_dfs, axis=1)
    final.to_csv("SummaryStatsPerWinsize.tsv", sep="\t", index=False)




if __name__ == "__main__":
    tsv = "../CRISCross/datasets/TCellDatasetWithextendedSequencesAndIDs.tsv"
    outfile = "SummaryNumpyArray.npy"
    generate_data(tsv, outfile)
    generate_window_stats(tsv, n_cpus=128)
