#!/bin/bash
#SBATCH -t 0-08:00:00
#SBATCH --job-name=THOR_structures
#SBATCH --gpus-per-node=A40:1
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
##SBATCH -p <partition>        # site-specific; uncomment and set
#SBATCH -o %x_%j.out

# ---------------------------------------------------------------------------
# What the Colab notebook does, as a batch job: for every record in a FASTA,
# pair it with the structure of the same name, blend ESM and ProteinMPNN
# features, generate variants, score them, then write the tables and figures.
#
# Each record is a separate generate_thor.py call, so ESM is loaded once per
# sequence. That costs about a minute each - fine for tens of proteins, worth
# rethinking for hundreds.
#
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch generate_thor_structures.sh
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
RUN_DIR=${RUN_DIR:-${THOR_RESULTS}/CHANGE_ME}
# Input sequences. A record id must match its structure's filename stem.
FASTA=${FASTA:-${THOR_DATA}/fasta/CHANGE_ME.fasta}
# Structures named <record_id>.cif or <record_id>.pdb (AlphaFold/Boltz .cif is fine).
# Leave empty to run ESM-only, exactly as before the ProteinMPNN work.
STRUCTURE_DIR=${STRUCTURE_DIR:-${THOR_DATA}/structures/CHANGE_ME}
# Results root; each record gets its own subdirectory under it.
OUTPUT_DIR=${OUTPUT_DIR:-${THOR_RESULTS}/thor_structures}

# ProteinMPNN is not in the image - it is a git checkout, bind-mounted below.
#     git clone https://github.com/dauparas/ProteinMPNN ${THOR_ROOT}/ProteinMPNN
MPNN_DIR=${MPNN_DIR:-${THOR_ROOT}/ProteinMPNN}

########################
# Generation knobs
########################
EPOCH=${EPOCH:-39}                       # loads RUN_DIR/weights/epoch_${EPOCH}/generator_G.h5
REPLICATES=${REPLICATES:-10}             # variants per input sequence
TEMPERATURE=${TEMPERATURE:-1.0}          # sampling temperature
MPNN_WEIGHT=${MPNN_WEIGHT:-0.3}          # 0 = pure ESM features, 1 = pure structure
MPNN_CHAIN=${MPNN_CHAIN:-}               # empty = first chain with CA atoms
NLL_THRESHOLD=${NLL_THRESHOLD:-3}
OPTIMISATION_CYCLES=${OPTIMISATION_CYCLES:-1}
SAMPLING_CYCLES=${SAMPLING_CYCLES:-0}
GPU_ID=${GPU_ID:-0}                      # passed to --gpu inside the container
ESM_DEVICE=${ESM_DEVICE:-cuda}           # cpu to keep ESM off the GPU
BATCH_TOKENS=${BATCH_TOKENS:-20000}      # reduce if you hit OOM
ESM_BF16=${ESM_BF16:---esm_bf16}         # set to "" to disable bfloat16 autocast
STORE_SOFTMAX=${STORE_SOFTMAX:---store_softmax}        # "" to skip the JSONL
# On by default, as in the Colab notebook: the ProteinMPNN line is the only
# record of what the structure said. Set to "" for smaller outputs.
STORE_SCORING_PROBS=${STORE_SCORING_PROBS- --store_scoring_probs}
SEQ_SOURCE=${SEQ_SOURCE:-fasta}          # "structure" to generate from the resolved chain instead

########################
# Bind mounts
########################
DATA_DIR=${THOR_DATA}
RESULTS_DIR=${THOR_RESULTS}
LOG_DIR=${THOR_LOG}

mkdir -p "${OUTPUT_DIR}"

# Only pass the structure flags when a directory was actually given.
STRUCTURE_ARGS=()
if [ -n "${STRUCTURE_DIR}" ] && [ -d "${STRUCTURE_DIR}" ]; then
    STRUCTURE_ARGS+=(--structure_dir "${STRUCTURE_DIR}" --mpnn_dir "${MPNN_DIR}" --mpnn_weight "${MPNN_WEIGHT}")
    [ -n "${MPNN_CHAIN}" ] && STRUCTURE_ARGS+=(--mpnn_chain "${MPNN_CHAIN}")
    MPNN_BIND=",${MPNN_DIR}:${MPNN_DIR}"
    echo "Structures from ${STRUCTURE_DIR} (ProteinMPNN weight ${MPNN_WEIGHT})"
else
    MPNN_BIND=""
    echo "No structure directory; running ESM-only"
fi

echo "Running generate_thor_structures.py"
singularity exec \
  -H "$(pwd)" \
  --bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN${MPNN_BIND} \
  --nv ${IMAGE} \
  python3 /ThermalGAN/src/scripts/generate_thor_structures.py \
    --run_dir "${RUN_DIR}" \
    --fasta "${FASTA}" \
    --output_dir "${OUTPUT_DIR}" \
    --seq_source "${SEQ_SOURCE}" \
    ${STRUCTURE_ARGS[@]+"${STRUCTURE_ARGS[@]}"} \
    -- \
    --weights weights/epoch_${EPOCH} \
    --replicates ${REPLICATES} \
    --temperature ${TEMPERATURE} \
    --gpu ${GPU_ID} \
    --device ${ESM_DEVICE} \
    --batch_tokens ${BATCH_TOKENS} \
    --esm_filter_threshold ${NLL_THRESHOLD} \
    --filter_opt_cycles ${OPTIMISATION_CYCLES} \
    --sampling_cycles ${SAMPLING_CYCLES} \
    --esm_init_mode random \
    --esm_random_mask_prob 0.15 \
    ${ESM_BF16} \
    ${STORE_SOFTMAX} \
    ${STORE_SCORING_PROBS}

echo "Results under ${OUTPUT_DIR} (combined table: ${OUTPUT_DIR}/all_variants.csv)"
