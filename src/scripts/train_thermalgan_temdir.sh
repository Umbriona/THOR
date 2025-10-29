#!/bin/bash
#SBATCH -t 0-00:20:00
#SBATCH --gpus-per-node=A40:1
#SBATCH -A NAISS2023-22-617
#SBATCH -p alvis

# BASE PARAMETERS

# Directory definitions

SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/ThermalGAN

EXPERIMENT=Different_target_identity

DATA_DIR=${SRC_DIR}/data/$EXPERIMENT #Experiment_11137_thermo

# Creating and transfering TEMP data
TMP_DATA_DIR=$TMPDIR/data/$EXPERIMENT
mkdir -p $TMP_DATA_DIR
rsync -aP $DATA_DIR/ $TMP_DATA_DIR

# Creating and TEMP results
export HDF5_USE_FILE_LOCKING=FALSE
TMP_RESULTS_DIR=$TMP_DATA_DIR/results/experiments
mkdir -p $TMP_RESULTS_DIR
rsync -aP /mimer/NOBACKUP/groups/snic2022-6-127/ThermalGAN/results/20240328-143412config_5.yaml $TMP_RESULTS_DIR

WEIGHTS_CLASSIFIER_DIR=${SRC_DIR}/weights/OGT_ensemble
TMP_WEIGHTS_CLASSIFIER_DIR=$TMPDIR/weights/OGT_ensemble
mkdir -p $TMP_WEIGHTS_CLASSIFIER_DIR
rsync -aP $WEIGHTS_CLASSIFIER_DIR/ $TMP_WEIGHTS_CLASSIFIER_DIR
TMP_LOG_DIR=${TMP_DATA_DIR}/log
mkdir -p $TMP_LOG_DIR

#SRC_DIR=../../../ThermalGAN

## Singularity
IMAGE=${SRC_DIR}/env/singularity/test_6.sif # thermalgan.sif


CONFIG=${DATA_DIR}/config_nonortologues_experiment/config_${SLURM_ARRAY_TASK_ID}.yaml


#module load Anaconda3/2022.05
#pip install simple_pid

#for idx in 0 
#do
# run training
singularity exec \
-H $(pwd) \
--bind ${TMP_DATA_DIR}:/data,${TMP_RESULTS_DIR}:/results,${TMP_LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN \
--nv ${IMAGE}  python train_cyclegan-diff_training.py \
-c ${CONFIG} -v -g 0

#rsync -aP $RESULTS_DIR $DATA_DIR
#rsync -aP $TMP_LOG_DIR $DATA_DIR
#done
