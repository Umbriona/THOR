#!/bin/bash
#SBATCH --job-name=THOR
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x.%j.out

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch train_thor.sh
# ---------------------------------------------------------------------------
if [ -z "${THOR_ROOT:-}" ]; then
    _here=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
    THOR_ROOT=$(cd "$_here" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null) || THOR_ROOT=""
    [ -n "$THOR_ROOT" ] || THOR_ROOT=$(cd "$_here/../.." && pwd)
fi
THOR_IMAGE=${THOR_IMAGE:-$THOR_ROOT/thor.sif}
THOR_DATA=${THOR_DATA:-$THOR_ROOT/data}
THOR_RESULTS=${THOR_RESULTS:-$THOR_DATA/results}
THOR_LOG=${THOR_LOG:-$THOR_ROOT/log}
mkdir -p "$THOR_LOG"

# Directory definitions
SRC_DIR=${THOR_ROOT}

DATA_DIR=${SRC_DIR}/data/THOR_HUGE #Different_target_identity #Different_target_identity #Experiment_11137_thermo
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
IMAGE=${THOR_IMAGE}


#CONFIG=${DATA_DIR}/config_nonortologues_experiment/config_${SLURM_ARRAY_TASK_ID}.yaml
CONFIG=${DATA_DIR}/config/config_${SLURM_ARRAY_TASK_ID}.yaml


#module load Anaconda3/2022.05
#pip install simple_pid

#for idx in 0 
#do
# run training
singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN \
--nv ${IMAGE}  python3 train_thor.py \
-c ${CONFIG} -v
#done
