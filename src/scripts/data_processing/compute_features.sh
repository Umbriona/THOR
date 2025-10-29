#!/bin/bash
#SBATCH --job-name=Generate_MLM_features
#SBATCH --account=NAISS2025-22-904
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-10:00:00
#SBATCH --output=%x.%j.out


FILE_FASTA=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/IMG_Thermophiles_EggNOG_90_filtered/split_files/file_${SLURM_ARRAY_TASK_ID}.fasta
IMAGE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/env/singularity/feature.sif
SCRIPT=generate_MLM_probability_features.py
DIR_FEATURE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/features_mlm/IMG_thermophiles_filtered_90 #non_redundant_v2


singularity exec --nv  ${IMAGE} /opt/conda/bin/python ${SCRIPT} -i ${FILE_FASTA} -o ${DIR_FEATURE} --max_entries 1000 --name IMG_thermophiles_filtered_90_rateMask_5_File_${SLURM_ARRAY_TASK_ID}_
