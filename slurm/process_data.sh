#!/bin/bash
#
#SBATCH --mail-user=nguyenhieu@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=process_data 
#SBATCH --chdir=/home/nguyenhieu/multimodal-ccRCC
#SBATCH --output=/home/nguyenhieu/multimodal-ccRCC/slurm/out/%j.%N.stdout
#SBATCH --error=/home/nguyenhieu/multimodal-ccRCC/slurm/out/%j.%N.stderr
#SBATCH --partition=debug 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=1-00:00:00

conda activate multimodal-ccRCC

python scripts/download_rna_seq.py
python scripts/process_rna_seq.py
python scripts/process_clinical.py
python scripts/process_wsi.py