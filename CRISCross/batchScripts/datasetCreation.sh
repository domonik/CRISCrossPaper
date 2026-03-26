#!/bin/bash


if [ -z "$AGAPIKEY" ]; then
    echo "Error: AGAPIKEY environment variable must be set."
    exit 1
fi

python datasetCreation/renameAndFixTCellDataset.py && \
python datasetCreation/changeTCellCoordinates.py && \
python datasetCreation/get_AGfeatures.py && \
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz && \
python datasetCreation/joinAGIndexAndExtractFasta.py && \
python datasetCreation/run_combinations_new.py

wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE149nnn/GSE149363/suppl/GSE149363_RAW.tar
mkdir -p BigWigTracks
tar -xf GSE149363_RAW.tar -C BigWigTracks
for f in BigWigTracks/GSM*; do
    new=$(echo "$(basename "$f")" | sed -E 's/^GSM[0-9]+_/EX_/; s/_FE\.bdg//')
    mv "$f" "BigWigTracks/$new"
done

python datasetCreation/extractExperimentalTracks.py
python datasetCreation/refixWillDataset.py
