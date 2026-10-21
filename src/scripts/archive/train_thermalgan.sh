#!/bin/bash

IMAGE=../../env/singularity/thermalgan.sif


# Directory definitions
DATA_DIR=../../data/processed/EE_22_09_27/11137
RESULTS_DIR=../../results/experiments
WEIGHTS_CLASSIFIER_DIR=../../weights/OGT_ensemble
LOG_DIR=../../log

SRC_DIR=../../../ThermalGAN


# config file

CONFIG="../../config/Cycle_gan/EE_22_09_27/Experiment_22_10_07.yaml"

singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${WEIGHTS_CLASSIFIER_DIR}:/weights,${SRC_DIR}:/ThermalGAN,/cephyr/users/sandravi/Alvis/.local/lib/python3.9/site-packages/simple_pid:/opt/conda/lib/python3.9/site-packages/simple_pid  \
--nv ${IMAGE}  /opt/conda/bin/python train_cyclegan.py \
-c ${CONFIG} -v
