#! /bin/bash
#SBATCH --job-name=Classifyer_OGT
#SBATCH --account=NAISS2024-5-302
#SBATCH --gres=gpu:A40:1
#SBATCH --time=1-20:30:00

## BASE_DIR
BASE_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

## Weights
WEIGHTS_DIR=${BASE_DIR}/weights

## Training script 
TRAINING_SCRIPT=/ThermalGAN/src/scripts/evaluate_classifier_embeddings.py

## Image
IMAGE=${BASE_DIR}/env/singularity/pytorch_transformer.sif


## RUN

singularity  exec --bind  $(pwd),${BASE_DIR}:/ThermalGAN  --nv ${IMAGE} /bin/python3 ${TRAINING_SCRIPT} -c ${CONFIG_FILE} --gpu 0 -v