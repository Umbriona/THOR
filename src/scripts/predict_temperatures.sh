#!/bin/bash
#SBATCH --job-name=Predict_Temperature
#SBATCH --account=NAISS2025-5-369
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --output=%x.%j.out

# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

DATA_DIR=${SRC_DIR}/data/OGT_IMG #Different_target_identity #Different_target_identity #Experiment_11137_thermo
RESULTS_DIR=${SRC_DIR}/results/
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





#module load Anaconda3/2022.05
#pip install simple_pid

#for idx in 0 
#do
# run training
time singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN, \
--nv ${IMAGE}  python get_temperature_prediction.py 
#done
