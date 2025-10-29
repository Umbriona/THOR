#! /bin/bash
#SBATCH --job-name=Classifyer_OGT
#SBATCH --account=NAISS2025-22-904
#SBATCH --gres=gpu:A100:1
#SBATCH --time=1-20:30:00

## BASE_DIR
BASE_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

## Weights
WEIGHTS_DIR=${BASE_DIR}/weights

## Training script 
TRAINING_SCRIPT=/ThermalGAN/src/scripts/train_classifier_embeddings.py

## Image
IMAGE=${BASE_DIR}/env/singularity/pytorch_transformer2.sif

## Output dir
OUTPUT=/ThermalGAN/weights/Classifier/OGT_IMG_DATA_EMBEDDINGS_4

## RUN
singularity  exec --bind  $(pwd),${BASE_DIR}:/ThermalGAN  --nv ${IMAGE} /bin/python3 ${TRAINING_SCRIPT} -o ${OUTPUT} --max_epochs 1000
