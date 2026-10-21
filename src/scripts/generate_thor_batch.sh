#!/bin/bash
#SBATCH -t 0-03:00:00
#SBATCH --job-name=Generate_THOR_Variants
#SBATCH --gpus-per-node=T4:1
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
##SBATCH -p <partition>        # site-specific; uncomment and set

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch generate_thor_batch.sh
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

# BASE PARAMETERS

# Directory definitions

module purge

# Directory definitions
SRC_DIR=${THOR_ROOT}

DATA_DIR=${SRC_DIR}/data/THOR_BIG #Different_target_identity #Different_target_identity #Experiment_11137_thermo#../../data/Non_ortologus #ALL_FAMILIES #Experiment_11137_thermo
RESULTS_DIR=${DATA_DIR}/results/experiments

WEIGHTS_CLASSIFIER_DIR=${SRC_DIR}/weights/OGT_ensemble
#TMP_WEIGHTS_CLASSIFIER_DIR=$TMPDIR/weights/OGT_ensemble
#mkdir -p $TMP_WEIGHTS_CLASSIFIER_DIR
#rsync -aP $WEIGHTS_CLASSIFIER_DIR/TMP_WEIGHTS_CLASSIFIER_DIR
LOG_DIR=${SRC_DIR}/log


## Singularity
##IMAGE=../../env/singularity/thermalgan.sif
IMAGE=${THOR_IMAGE}

INPUT=${DATA_DIR}/results/20251125-162825config_46.yaml/ #../../results/Models_used_in_experiments/Batch_4_models2/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml #../..//results/20240216-124423config_0.yaml/config.yaml #${DATA_DIR}/config/config_all_${SLURM_ARRAY_TASK_ID}.yaml


#for idx in 0 
#do
# run training
for EPOCH in 3 6 9 12 15 18 21 24 27 30 33 36 39
do
singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN \
--nv ${IMAGE}  python generate_thor_batch.py \
--input ${INPUT} \
'--store_softmax' \
-v -g 0 \
--epoch $EPOCH
done
