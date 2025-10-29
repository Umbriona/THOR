#!/bin/bash
#SBATCH --job-name=Generate_MLM_features
#SBATCH --account=NAISS2025-22-904
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --output=%x.%j.out


FILE_FASTA=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/IMG_train_09_5/test_IMG_09_05.fasta #${SLURM_ARRAY_TASK_ID}.fasta

IMAGE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/env/singularity/esm-finetune.sif

SCRIPT=compute_MLM_features_quantized.py
DIR_FEATURE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/features_mlm/IMG_test #non_redundant_v2

CHECKPOINT=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/weights/esm1v_mlm_ur90_finetune/checkpoint-119432


singularity exec --nv  ${IMAGE} /opt/conda/bin/python \
 ${SCRIPT} \
 -i ${FILE_FASTA} \
 -o ${DIR_FEATURE} \
 --model ${CHECKPOINT} \
 --mode single \
 --quantize uint8 \
 --max_entries 10000 \
 --batch_tokens 40000\
 --name Target_IMG_test_single_ckp119432 \
 --bf16

# --mask_prob 0.15 \
