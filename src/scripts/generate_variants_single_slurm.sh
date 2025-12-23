#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH --job-name=Gen_ThermalGAN
#SBATCH --gpus-per-node=A40:1
#SBATCH --account=NAISS2025-5-369
#SBATCH -p alvis
#SBATCH -o %x_%j.out

set -euo pipefail

module purge

########################
# Paths to edit
########################
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN
IMAGE=${SRC_DIR}/env/singularity/thermalgan-image1.2.sif #thermalgan-image1.sif

# Trained run directory containing config.yaml and weights/epoch_<n>/
RUN_DIR=${SRC_DIR}/data/THOR_HUGE/results/20251219-123030config_71.yaml #20251218-225521config_67.yaml #20251218-225520config_64.yaml
# Input FASTA to score (temperature can be last token in header, optional)
FASTA=${SRC_DIR}/data/fasta/MDH_test/MDH.fasta #PETase/PETases.fasta
# Where to drop generated FASTA/JSONL (defaults to RUN_DIR if left empty)
OUTPUT_DIR=${RUN_DIR}

# Generation knobs
EPOCH=39                        # which epoch under RUN_DIR/weights/ to load
REPLICATES=5                    # variants per input sequence
TEMPERATURE=1                 # sampling temperature
STORE_SOFTMAX=--store_softmax   # set to empty string "" to skip JSONL
GPU_ID=0                        # passed to --gpu inside the container
ESM_DEVICE=cuda                 # set to cuda to score ESM on GPU (uses more memory)
BATCH_TOKENS=6000               # reduce if you hit OOM
ESM_BF16=                      # set to "--esm_bf16" to enable bfloat16 autocast
NLL_THREASHOLD=3
OPTIMISATION_CYCLES=1
NAME=MDH
########################
# Bind mounts
########################
DATA_DIR=${SRC_DIR}/data
RESULTS_DIR=${SRC_DIR}/results
LOG_DIR=${SRC_DIR}/log

echo "Running generate_variants_single.py"
singularity exec \
  -H "$(pwd)" \
  --bind ${DATA_DIR}:/data,${RESULTS_DIR}:/results,${LOG_DIR}:/log,${SRC_DIR}:/ThermalGAN,/apps/Common/software/CUDA/12.9.1/extras/CUPTI:/usr/local/cuda/extras/CUPTI \
  --nv ${IMAGE} \
  python /ThermalGAN/src/scripts/generate_variants_single.py \
    --run_dir ${RUN_DIR} \
    --fasta ${FASTA} \
    --epoch ${EPOCH} \
    --replicates ${REPLICATES} \
    --temperature ${TEMPERATURE} \
    --gpu ${GPU_ID} \
    --output_dir ${OUTPUT_DIR} \
    --device ${ESM_DEVICE} \
    --batch_tokens ${BATCH_TOKENS} \
    --esm_filter_threshold ${NLL_THREASHOLD} \
    --filter_opt_cycles ${OPTIMISATION_CYCLES} \
    ${ESM_BF16} \
    --name ${NAME} \
    ${STORE_SOFTMAX}
