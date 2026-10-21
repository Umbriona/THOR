#!/bin/bash
#SBATCH --job-name=Generate_MLM_features
#SBATCH --account=NAISS2025-5-369
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --array=50-99


FILE_FASTA=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/THOR_HUGE/fasta/IMG_Thermophiles/split_files/IMG_Thermophiles_NoCOG_${SLURM_ARRAY_TASK_ID}.fasta #PETases.fasta #/THOR_BIG/test_THOR_Meso.fasta #_${SLURM_ARRAY_TASK_ID}.fasta #OGT_train_09_16/test_OGT_09_05.fasta #${SLURM_ARRAY_TASK_ID}.fasta

IMAGE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/env/singularity/esm-finetune_tf.sif

SCRIPT=compute_MLM_features_quantized_tfrecord.py
DIR_FEATURE=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/data/THOR_HUGE/records/IMG_Thermophiles #non_redundant_v2

CHECKPOINT=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN/weights/esm1v_mlm_ur90_finetune/checkpoint-119432 #esm1v_mlm_ur90_finetune/checkpoint-119432 #esm1v_mlm_ur90_finetune/checkpoint-119432 # prot_bert_bfd #esm1v_mlm_ur90_finetune/checkpoint-119432


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
 --name IMG_Thermophiles_NOCOG_${SLURM_ARRAY_TASK_ID} \
 --min_quality 0.0 \
 --quality_bins 0.0,0.2,0.5,0.7,1.0\
 --drop_nan_quality \
 --bf16

#  \
