#!/bin/bash
#SBATCH -t 0-00:40:00
#SBATCH -n 16
#SBATCH --gpus-per-node=A100:1
#SBATCH -A C3SE2023-1-18
#SBATCH -p vera

# BASE PARAMETERS

# Directory definitions

module purge

DATA_DIR=../../data/Experiment_MDH_CuSOD #ALL_FAMILIES #Experiment_11137_thermo
RESULTS_DIR=${DATA_DIR}/results/experiments

WEIGHTS_CLASSIFIER_DIR=../../weights/OGT_ensemble
LOG_DIR=${DATA_DIR}/log

SRC_DIR=../../../ThermalGAN

## Singularity
IMAGE=../../env/singularity/thermalgan.sif


CONFIG=../../results/20240216-124423config_0.yaml/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml #../..//results/20240216-124423config_0.yaml/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml


module load Anaconda3/2022.05
pip install simple_pid

#for idx in 0 
#do
# run training
singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN,/cephyr/users/sandravi/Alvis/.local/lib/python3.9/site-packages/simple_pid:/opt/conda/lib/python3.9/site-packages/simple_pid \
--nv ${IMAGE}  /opt/conda/bin/python test_generate_ThermalGAN.py \
-c ${CONFIG} -v -g 0
#done
