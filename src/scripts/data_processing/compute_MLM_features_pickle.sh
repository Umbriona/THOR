#!/bin/bash
#SBATCH --job-name=MLM_features_pickle
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --output=%x.%j.out

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch compute_MLM_features_pickle.sh
# ---------------------------------------------------------------------------
if [ -z "${THOR_ROOT:-}" ]; then
    _here=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
    THOR_ROOT=$(cd "$_here" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null) || THOR_ROOT=""
    [ -n "$THOR_ROOT" ] || THOR_ROOT=$(cd "$_here/../../.." && pwd)
fi
THOR_IMAGE=${THOR_IMAGE:-$THOR_ROOT/thor.sif}
THOR_DATA=${THOR_DATA:-$THOR_ROOT/data}
THOR_RESULTS=${THOR_RESULTS:-$THOR_DATA/results}
THOR_LOG=${THOR_LOG:-$THOR_ROOT/log}
mkdir -p "$THOR_LOG"


FILE_FASTA=${THOR_ROOT}/data/fasta/IMG_train_09_5/test_IMG_09_05.fasta #${SLURM_ARRAY_TASK_ID}.fasta

IMAGE=${THOR_IMAGE}

SCRIPT=compute_MLM_features.py
DIR_FEATURE=${THOR_ROOT}/data/features_mlm/IMG_test #non_redundant_v2

CHECKPOINT=${THOR_ROOT}/weights/esm1v_mlm_ur90_finetune/checkpoint-119432


singularity exec --nv  ${IMAGE} python3 \
 ${SCRIPT} \
 -i ${FILE_FASTA} \
 -o ${DIR_FEATURE} \
 --model ${CHECKPOINT} \
 --mode single \
 --quantize uint8 \
 --max_entries 10000 \
 --batch_tokens 40000\
 --name Target_IMG_test_single_ckp119432 \
 --format pickle \
 --bf16

# --mask_prob 0.15 \
