#! /bin/bash
#SBATCH --job-name=ET_regressor
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --gres=gpu:A40:1
#SBATCH --time=1-20:30:00

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch train_et_regressor.sh
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

## BASE_DIR
BASE_DIR=${THOR_ROOT}

## Weights
WEIGHTS_DIR=${BASE_DIR}/weights

## Config file
CONFIG_FILE=${BASE_DIR}/config/Classifier/config_OGT_IMG.yaml

## Training script 
TRAINING_SCRIPT=/ThermalGAN/src/scripts/train_et_regressor.py

## Image
IMAGE=${THOR_IMAGE}


## RUN

singularity  exec --bind  $(pwd),${BASE_DIR}:/ThermalGAN  --nv ${IMAGE} python3 ${TRAINING_SCRIPT} -c ${CONFIG_FILE} --gpu 0 -v