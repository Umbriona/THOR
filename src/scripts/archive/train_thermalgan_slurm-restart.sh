#!/bin/bash
#SBATCH -t 5-00:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH -A NAISS2023-22-617
#SBATCH -p alvis

# BASE PARAMETERS
##SLURM_ARRAY_TASK_ID=5 # Only used for debugging
# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/ThermalGAN

DATA_DIR=${SRC_DIR}/data/Different_target_identity #Experiment_11137_thermo
#TMP_DATA_DIR=$TMPDIR/data/Different_target_identity
#mkdir -p $TMP_DATA_DIR
#rsync -aP $DATA_DIR/$TMP_DATA_DIR
RESULTS_DIR=${DATA_DIR}/results/experiments
#TMP_RESULTS_DIR=$TMP_DATA_DIR/results/experiments
#mkdir -p $TMP_RESULTS_DIR

WEIGHTS_CLASSIFIER_DIR=${SRC_DIR}/weights/OGT_ensemble
#TMP_WEIGHTS_CLASSIFIER_DIR=$TMPDIR/weights/OGT_ensemble
#mkdir -p $TMP_WEIGHTS_CLASSIFIER_DIR
#rsync -aP $WEIGHTS_CLASSIFIER_DIR/TMP_WEIGHTS_CLASSIFIER_DIR
LOG_DIR=${SRC_DIR}/log
#mkdir -p LOG_DIR

#SRC_DIR=../../../ThermalGAN

## Singularity
#module load TensorFlow #TensorFlow-Graphics/2021.12.3-foss-2021b-CUDA-11.4.1

#export APPTAINERENV_PATH=$PATH
#export APPTAINERENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
IMAGE=${SRC_DIR}/env/singularity/test_6.sif # thermalgan.sif


CONFIG=${DATA_DIR}/config_ortologues_experiment/config_${SLURM_ARRAY_TASK_ID}.yaml


#module load Anaconda3/2022.05
#pip install simple_pid

#for idx in 0 
#do
# run training
singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN \
--nv ${IMAGE}  python -u train_cyclegan-diff_training.py \
-c ${CONFIG} -v -g 0
#done
