#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH --job-name=Gen_THOR
#SBATCH --gpus-per-node=A40:1
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
##SBATCH -p <partition>        # site-specific; uncomment and set
#SBATCH -o %x_%j.out

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch generate_thor.sh
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

set -euo pipefail

module purge

########################
# Paths to edit
########################
SRC_DIR=${THOR_ROOT}
IMAGE=${THOR_IMAGE}

# Trained run directory containing config.yaml and weights/epoch_<n>/
RUN_DIR=${SRC_DIR}/data/THOR_HUGE/results/20251218-225521config_67.yaml #20251218-225521config_67.yaml #20251218-225520config_64.yaml
# Input FASTA to score (temperature can be last token in header, optional)
FASTA=${SRC_DIR}/data/fasta/PETase/PETases.fasta #PETase/PETases.fasta
# Where to drop generated FASTA/JSONL (defaults to RUN_DIR if left empty)
OUTPUT_DIR=${RUN_DIR}

# Generation knobs
EPOCH=39                        # loads RUN_DIR/weights/epoch_${EPOCH}/generator_G.h5
REPLICATES=10                    # variants per input sequence
TEMPERATURE=1                 # sampling temperature
STORE_SOFTMAX=--store_softmax   # set to empty string "" to skip JSONL
GPU_ID=0                        # passed to --gpu inside the container
ESM_DEVICE=cuda                 # set to cuda to score ESM on GPU (uses more memory)
BATCH_TOKENS=20000               # reduce if you hit OOM
ESM_BF16=                      # set to "--esm_bf16" to enable bfloat16 autocast
NLL_THREASHOLD=3
OPTIMISATION_CYCLES=1
NAME=PET_10
########################
# Bind mounts
########################
DATA_DIR=${SRC_DIR}/data
RESULTS_DIR=${SRC_DIR}/results
LOG_DIR=${SRC_DIR}/log

echo "Running generate_thor.py"
singularity exec \
  -H "$(pwd)" \
  --bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN \
  --nv ${IMAGE} \
  python3 /ThermalGAN/src/scripts/generate_thor.py \
    --run_dir ${RUN_DIR} \
    --fasta ${FASTA} \
    --weights weights/epoch_${EPOCH} \
    --replicates ${REPLICATES} \
    --temperature ${TEMPERATURE} \
    --gpu ${GPU_ID} \
    --output_dir ${OUTPUT_DIR} \
    --device ${ESM_DEVICE} \
    --batch_tokens ${BATCH_TOKENS} \
    --esm_filter_threshold ${NLL_THREASHOLD} \
    --filter_opt_cycles ${OPTIMISATION_CYCLES} \
    --sampling_cycles 0 \
    ${ESM_BF16} \
    --name ${NAME} \
    ${STORE_SOFTMAX} \
    --esm_init_mode random \
    --esm_random_mask_prob 0.15 \