#!/bin/bash
#SBATCH --job-name=Generate_MLM_features
#SBATCH --account=NAISS2025-22-904
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --output=%x.%j.out


FILE_FASTA=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/fasta/THOR_BIG/test_THOR_Meso.fasta #_${SLURM_ARRAY_TASK_ID}.fasta #OGT_train_09_16/test_OGT_09_05.fasta #${SLURM_ARRAY_TASK_ID}.fasta

IMAGE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/env/singularity/esm-finetune_tf.sif

SCRIPT=compute_MLM_features_quantized_tfrecord.py
DIR_FEATURE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/THOR_BIG/records/ESM/test #non_redundant_v2

CHECKPOINT=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/weights/esm1v_mlm_ur90_finetune/checkpoint-119432 # prot_bert_bfd #esm1v_mlm_ur90_finetune/checkpoint-119432


singularity exec --nv  ${IMAGE} /opt/conda/bin/python \
 ${SCRIPT} \
 -i ${FILE_FASTA} \
 -o ${DIR_FEATURE} \
 --model ${CHECKPOINT} \
 --mode random \
 --mask_prob 0.15 \
 --quantize uint8 \
 --max_entries 50000 \
 --batch_tokens 40000\
 --random_batch 64 \
 --format tfrecord\
 --name Meso_file_0${SLURM_ARRAY_TASK_ID} \
 --min_quality 0.1 \
 --quality_bins 0.2,0.5,0.7,1.0\
 --drop_nan_quality \
 --bf16

#  \
