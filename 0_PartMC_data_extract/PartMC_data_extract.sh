#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
#$ -pe smp.pe 8

# module load apps/binapps/anaconda3/2022.10
source activate partmc



python PartMC_data_extract.py



conda deactivate

echo "Job Finish"
