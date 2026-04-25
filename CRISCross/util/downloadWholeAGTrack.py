
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
from alphagenome.data.genome import Interval
from alphagenome.models.dna_client import OutputType

import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import numpy as np



CHROMOSOME_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415
}





#WINDOW_SIZE = 2048
#WINDOW_SIZE = 2**17
WINDOW = 2**20

EXTRACT = 2**16         # 16,384
FLANK = (WINDOW - EXTRACT) // 2



import grpc

def run_batch_prediction(dna_model, intervals, organism, ontology_terms, OutputType,
                         max_retries=5, base_delay=1, max_delay=60):
    attempt = 0

    while attempt < max_retries:
        try:
            outputs = dna_model.predict_intervals(
                intervals=intervals,
                requested_outputs=set(OutputType),
                ontology_terms=ontology_terms,
                organism=organism,
                progress_bar=True,
                max_workers=5
            )
            return outputs

        except grpc._channel._MultiThreadedRendezvous as e:
            # Check if the error is RESOURCE_EXHAUSTED (quota exceeded)
            attempt += 1
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"Quota exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
            print(e)
            time.sleep(wait_time)
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            attempt += 1
            wait_time = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"Quota exceeded. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
            print(e)

            time.sleep(wait_time)
            return []

    print("Max retries exceeded. Returning empty batch.")
    return []

#### change ontology depending on cell type, Tcell = 'CL:0000084' ######
def extract_batch_features(rows_df, ontology_terms=['CL:0000624'], skipped_rows_path="./alphagenome_testcode/skipped_intervals.csv"):
    intervals = []
    valid_rows = []
    skipped_rows = []

    for _, row in rows_df.iterrows():
        chrom = row['chr']
        start = row['start']

        # Normalize chromosome name
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom

        # Update row if we modified chrom
        row = row.copy()
        row['chr'] = chrom

        intervals.append(interval)
        valid_rows.append(row)
    

    if not intervals:
        return []

    batch_features = run_batch_prediction(dna_model=dna_model, intervals=intervals, valid_rows=valid_rows,
                                          ontology_terms=ontology_terms, OutputType=OutputType, max_retries=25, base_delay=5, max_delay=60)
    

    

    return batch_features



def create_interval_from_row(row):       
    return Interval(
        chromosome=row['chrom'],
        start=row["start"],
        end=row["end"],
    )


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ontology",
        required=True,
        help="Ontology CURIE (e.g. CL:0000624)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    BATCH_SIZE = 100
    FEATURE_LIST = ["ATAC", "DNASE", "CHIP_HISTONE", "RNA_SEQ"]

    print("Extracting features for all intervals...")
    feature_list = []
    organism = dna_client.Organism.HOMO_SAPIENS 
    args = parse_args()
    ONT = args.ontology
    
    OUTDIR = f"AGTensors{ONT}"


    apikey = os.environ['AGAPIKEY']

    
    dna_model = dna_client.create(apikey)


    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)
        
    rows = []
    for chrom, chrom_size in CHROMOSOME_SIZES.items():
        extract_start = 0

        while extract_start < chrom_size:
            extract_end = min(extract_start + EXTRACT, chrom_size)

            # Proposed centered window
            start = extract_start - FLANK
            end = start + WINDOW

            # Adjust for chromosome boundaries
            if start < 0:
                start = 0
                end = min(WINDOW, chrom_size)

            if end > chrom_size:
                end = chrom_size
                start = max(0, chrom_size - WINDOW)

            rows.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "extract_start": extract_start,
                "extract_end": extract_end
            })

            extract_start += EXTRACT

    df = pd.DataFrame(rows)
    df = (df.groupby(["chrom", "start", "end"], as_index=False).agg({"extract_start": min, "extract_end": max}))

    #df = df.sample(1000).reset_index()
    ag_info = pd.read_csv("media-1.csv", sep="\t")
    ag_info = ag_info[ag_info["ontology_curie"] == ONT]
    ag_info = ag_info[ag_info.output_type.isin(FEATURE_LIST)]

    ag_info.loc[ag_info["output_type"] == "CHIP_HISTONE", "newTrackIndex"] = ag_info.loc[ag_info["output_type"] == "CHIP_HISTONE", "Target label"]
    ag_info.loc[ag_info["output_type"] == "RNA_SEQ", "newTrackIndex"] = ag_info.loc[ag_info["output_type"] == "RNA_SEQ", "strand"] + "_" + ag_info.loc[ag_info["output_type"] == "RNA_SEQ", "Assay title"]


    TRACK_NAMES = ag_info.groupby("output_type")["newTrackIndex"].unique().reset_index()
    NAMETRACKNAMMPING = {row["name"]: row["newTrackIndex"] for _, row in ag_info.iterrows()}
    TRACK_NAMES = TRACK_NAMES.explode("newTrackIndex", ignore_index=True)
    TRACK_NAMES["list_index"] = (
    TRACK_NAMES.groupby(["output_type"])
      .cumcount()
    )
    TRACK_NAMES["newTrackIndex"] = TRACK_NAMES["newTrackIndex"].fillna(TRACK_NAMES["output_type"])


    ag_info = ag_info[["biosample_name", "ontology_curie"]]
    ag_info = ag_info.drop_duplicates(["biosample_name", "ontology_curie"])



    def create_arrays(outputs):
        output = outputs[0]
        sizes = {}
        for idx, row in TRACK_NAMES.iterrows():
            for chrom, chrom_size in CHROMOSOME_SIZES.items():
                filename = os.path.join(OUTDIR, f"{row['newTrackIndex']}_{chrom}.npy")
                data = output.get(OutputType[row["output_type"]])
                assert data is not None
                shape = chrom_size, 1 
                mmap_array = np.memmap(filename, dtype=np.float32, mode="w+", shape=shape)
                mmap_array.flush()


    def fill_array(outputs, rows):
        assert len(outputs) == len(rows)


        for idx, track_row in TRACK_NAMES.iterrows():
            fdata = outputs[0].get(OutputType[track_row["output_type"]])
            chrom = rows.iloc[0]["chrom"]
            filename = os.path.join(OUTDIR, f"{track_row['newTrackIndex']}_{chrom}.npy")
            shape = CHROMOSOME_SIZES[chrom], 1 

            mmap_array = np.memmap(filename, dtype=np.float32, mode='r+', shape=shape)
            for i, output in enumerate(outputs):
                row = rows.iloc[i]
                data = output.get(OutputType[track_row["output_type"]])
                row_idx = fdata.metadata["name"].map(NAMETRACKNAMMPING)
                if data.resolution != 1:
                    data = data.change_resolution(resolution=1)
                if not pd.isna(row_idx).any():
                    if track_row["list_index"] >= data.values.shape[1]:
                        breakpoint()
                    vals = data.values[:, track_row["list_index"]][..., None]
                else:
                    vals = data.values
                vals_offset_start = row["extract_start"] - row["start"]
                vals_offset_end   = row["extract_end"] - row["start"]
                mmap_array[row["extract_start"]:row["extract_end"]] = vals[vals_offset_start:vals_offset_end]
            mmap_array.flush()





    idx = 0
    df["Intervals"] = df.apply(create_interval_from_row, axis=1)
    for (chrom), group in df.groupby(["chrom"]):
        print("Extracting features in batches...")

        for batch_idx, start_idx in enumerate(range(0, len(group), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(group))
            batch_df = group.iloc[start_idx:end_idx]

            print(f"Processing batch {start_idx} to {end_idx}...")
            batch_output = run_batch_prediction(dna_model=dna_model, organism=organism, intervals=batch_df["Intervals"], 
                                            ontology_terms=[ONT], OutputType=OutputType, max_retries=25, base_delay=5, max_delay=60)
        
            if idx == 0:
                sizes = create_arrays(batch_output)
                idx += 1
            fill_array(outputs=batch_output, rows=batch_df)










