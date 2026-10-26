#!/bin/bash
#SBATCH --job-name=MLM_features_tfrecord
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --array=0-99

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch compute_MLM_features_tfrecord.sh
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


FILE_FASTA=${THOR_ROOT}/data/THOR_HUGE/fasta/Uniprot_enzymes/split_files/file_${SLURM_ARRAY_TASK_ID}.fasta #PETases.fasta #/THOR_BIG/test_THOR_Meso.fasta #_${SLURM_ARRAY_TASK_ID}.fasta #OGT_train_09_16/test_OGT_09_05.fasta #${SLURM_ARRAY_TASK_ID}.fasta

IMAGE=${THOR_IMAGE}

SCRIPT=compute_MLM_features.py
DIR_FEATURE=${THOR_ROOT}/data/THOR_HUGE/records/train/Uniprot_enzymes #non_redundant_v2

# ESM model used to compute per-position likelihoods. Stock ESM-1v is what the
# published results use: fine-tuning ESM on the training corpus was tried (see
# finetune_ESM.py) and made no appreciable difference. Override to point at a
# local directory if you want to avoid re-downloading the weights.
ESM_MODEL=${ESM_MODEL:-facebook/esm1v_t33_650M_UR90S_1}


singularity exec --nv  ${IMAGE} python3 \
 ${SCRIPT} \
 -i ${FILE_FASTA} \
 -o ${DIR_FEATURE} \
 --model ${ESM_MODEL} \
 --mode random \
 --mask_prob 0.15 \
 --quantize uint8 \
 --max_entries 50000 \
 --batch_tokens 40000\
 --random_batch 64 \
 --format tfrecord\
 --name file_${SLURM_ARRAY_TASK_ID} \
 --min_quality 0.0 \
 --quality_bins 0.0,0.2,0.5,0.7,1.0\
 --drop_nan_quality \
 --bf16

#  \
