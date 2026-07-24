
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
from alphagenome.data.genome import Interval
from alphagenome.models.dna_client import OutputType

import pandas as pd
import time
import os
import numpy as np


apikey = os.environ['AGAPIKEY']

dna_model = dna_client.create(apikey)

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
WINDOW_SIZE = 2**14

HALF_WINDOW = WINDOW_SIZE // 2

def create_interval_from_row(row):        # need to change column names depending on the input file
    center = row['start'] + 11
    return Interval(
        chromosome=row['chr'],
        start=center - HALF_WINDOW,
        end=center + HALF_WINDOW,
        strand=row['Strand']
    )

def is_valid_interval(chrom, start):
    # Ensure 'chr' prefix
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    chrom_length = CHROMOSOME_SIZES.get(chrom)
    if chrom_length is None:
        return False
    return 0 <= start < chrom_length and (start + WINDOW_SIZE) <= chrom_length



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

        interval = create_interval_from_row(row)
        intervals.append(interval)
        valid_rows.append(row)
    

    if not intervals:
        return []

    batch_features = run_batch_prediction(dna_model=dna_model, intervals=intervals, valid_rows=valid_rows,
                                          ontology_terms=ontology_terms, OutputType=OutputType, max_retries=25, base_delay=5, max_delay=60)
    

    

    return batch_features


if __name__ == "__main__":
    BATCH_SIZE = 100
    FEATURE_LIST = ["ATAC"]

    print("Extracting features for all intervals...")
    feature_list = []
    OUTDIR = "AGTensorsK562"

    df = pd.read_csv("tmpdatasets/K562WithWrongCoordsForOldPipeline.tsv", sep="\t")
    if not os.path.isdir(OUTDIR):
        os.mkdir(OUTDIR)


    #df = df.sample(1000).reset_index()
    ag_info = pd.read_csv("../datasets/AlphagenomeInfo.csv", sep="\t")
    ag_info = ag_info[ag_info.output_type.isin(FEATURE_LIST)]
    #ag_info = ag_info[ag_info["biosample_name"].isin(df["Match in database"])]
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
    #df = df.merge(ag_info, left_on="Match in database", right_on="biosample_name", how="left")

    df["Assembly"] = "hg38"
    df["Query term"] = "K562"
    df = df.drop_duplicates(["chr", "start", "end", "Strand", "Assembly", "Query term"])
    df["AlphagenomeIndex"] = np.arange(len(df))



    df["Intervals"] = df.apply(create_interval_from_row, axis=1)
    df["ontology_curie"] = "EFO:0002067"





    def create_arrays(outputs):
        output = outputs[0]
        sizes = {}
        for idx, row in TRACK_NAMES.iterrows():
            filename = os.path.join(OUTDIR, f"{row['newTrackIndex']}.np")
            data = output.get(OutputType[row["output_type"]])
            assert data is not None
            shape = len(df), WINDOW_SIZE // data.resolution, 1 
            mmap_array = np.memmap(filename, dtype=np.float32, mode="w+", shape=shape)
            mmap_array.flush()


    def fill_array(outputs, rows):
        assert len(outputs) == len(rows)


        for idx, track_row in TRACK_NAMES.iterrows():
            fdata = outputs[0].get(OutputType[track_row["output_type"]])

            filename = os.path.join(OUTDIR, f"{track_row['newTrackIndex']}.np")
            shape = len(df), WINDOW_SIZE // fdata.resolution, 1 

            mmap_array = np.memmap(filename, dtype=np.float32, mode='r+', shape=shape)
            for i, output in enumerate(outputs):
                row = rows.iloc[i]
                ag_index = row["AlphagenomeIndex"]
                data = output.get(OutputType[track_row["output_type"]])
                row_idx = fdata.metadata["name"].map(NAMETRACKNAMMPING)
                if not pd.isna(row_idx).any():
                    mmap_array[ag_index] = data.values[:, track_row["list_index"]][..., None]
                else:
                    mmap_array[ag_index] = data.values
            mmap_array.flush()





    idx = 0
    for (assembly, term), group in df.groupby(["Assembly", "ontology_curie"]):
        print("Extracting features in batches...")
        for batch_idx, start_idx in enumerate(range(0, len(group), BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, len(group))
            batch_df = group.iloc[start_idx:end_idx]

            print(f"Processing batch {start_idx} to {end_idx}...")
            organism = dna_client.Organism.HOMO_SAPIENS if assembly == "hg38" else dna_client.Organism.MUS_MUSCULUS
            batch_output = run_batch_prediction(dna_model=dna_model, organism=organism, intervals=batch_df["Intervals"], 
                                            ontology_terms=[batch_df["ontology_curie"].iloc[0]], OutputType=OutputType, max_retries=25, base_delay=5, max_delay=60)
        
            if idx == 0:
                sizes = create_arrays(batch_output)
                idx += 1
            fill_array(outputs=batch_output, rows=batch_df)

    print(feature_list)
    # # Convert to DataFrame
    features_df = pd.DataFrame(feature_list)

    print(features_df.columns)
    print(features_df.shape)
    # # Save the features
    df.to_csv('tmpdatasets/AlphagenomeIndexK562.tsv', sep="\t")
    TRACK_NAMES.to_csv("tmpdatasets/TrackNamesK562.tsv", sep="\t")








