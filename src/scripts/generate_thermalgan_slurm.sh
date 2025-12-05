#!/bin/bash
#SBATCH -t 0-03:00:00
#SBATCH --job-name=Generate_THOR_Variants
#SBATCH --gpus-per-node=T4:1
#SBATCH --account=NAISS2025-5-369
#SBATCH -p alvis

# BASE PARAMETERS

# Directory definitions

module purge

# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

DATA_DIR=${SRC_DIR}/data/THOR_BIG #Different_target_identity #Different_target_identity #Experiment_11137_thermo#../../data/Non_ortologus #ALL_FAMILIES #Experiment_11137_thermo
RESULTS_DIR=${DATA_DIR}/results/experiments

WEIGHTS_CLASSIFIER_DIR=${SRC_DIR}/weights/OGT_ensemble
#TMP_WEIGHTS_CLASSIFIER_DIR=$TMPDIR/weights/OGT_ensemble
#mkdir -p $TMP_WEIGHTS_CLASSIFIER_DIR
#rsync -aP $WEIGHTS_CLASSIFIER_DIR/TMP_WEIGHTS_CLASSIFIER_DIR
LOG_DIR=${SRC_DIR}/log


## Singularity
##IMAGE=../../env/singularity/thermalgan.sif
IMAGE=${SRC_DIR}/env/singularity/thermalgan-image1.sif #test_6.sif # thermalgan.sif

INPUT=${DATA_DIR}/results/20251125-162825config_46.yaml/ #../../results/Models_used_in_experiments/Batch_4_models2/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml #../..//results/20240216-124423config_0.yaml/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml


#for idx in 0 
#do
# run training
singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN,/apps/Common/software/CUDA/12.9.1/extras/CUPTI:/usr/local/cuda/extras/CUPTI \
--nv ${IMAGE}  python test_generate_ThermalGAN.py \
--input ${INPUT} -v -g 0 \
--epoch 33
#done
