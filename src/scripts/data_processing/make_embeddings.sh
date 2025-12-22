#!/bin/bash
#SBATCH --job-name=ThermalGAN
#SBATCH --account=NAISS2025-22-904
#SBATCH --gres=gpu:A40:1
#SBATCH --time=0-06:30:00

# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

DATA_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/Design_ProteinMPNN #Experiment_11137_thermo

RESULT=${DATA_DIR}/best_variants_70_TM.pkl

IMAGE=${SRC_DIR}/env/singularity/pytorch_transformer.sif # thermalgan.sif

#INPUT=/data/fasta/train_IMG20_2025_05_31/train_IMG20_2025_05_31_${SLURM_ARRAY_TASK_ID}.fasta
INPUT=${DATA_DIR}/best_variants_70_TM.fasta

 

singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${SRC_DIR}:/ThermalGAN \
--nv ${IMAGE}  /bin/python3 make_embeddings.py \
-i ${INPUT} -o ${RESULT} --max_entries=10000 --ID 0 #${SLURM_ARRAY_TASK_ID}
