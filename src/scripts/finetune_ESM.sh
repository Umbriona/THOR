#!/bin/bash
#SBATCH -J esm1v-mlm
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=3-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone: THOR_ROOT is resolved
# from the directory you submitted from (sbatch) or the script's own location.
# Override any of these by exporting them, e.g.
#     THOR_DATA=/scratch/$USER/data sbatch finetune_ESM.sh
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


# --- Env setup (edit to your cluster) ---
module purge
# Example: use a conda env with pytorch>=2.1, transformers>=4.39, datasets>=2.14
# module load cuda/12.1  # if your site needs it
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate esm-ft

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_DATASETS_CACHE=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
mkdir -p logs "$HF_DATASETS_CACHE"

# Recommended on A100:
BF16_FLAG="--bf16"

# Paths
DATA_PATH=${THOR_ROOT}/data/fasta/IMG_train_09_5/   # or directory, or .txt
DATA=train_IMG_09_05.fasta
DATA_VAL=test_IMG_09_05.fasta
OUT_DIR=${THOR_ROOT}/weights/esm1v_mlm_ur90_finetune
MODEL_ID=facebook/esm1v_t33_650M_UR90S_1

# Training hyperparams (tweak to your VRAM):
EPOCHS=4
PER_DEVICE_BATCH=4         # ~4 x 1022 tokens fits on 40–80GB A100; adjust if OOM
ACCUM=4                    # effective batch = 4 GPUs * 4 * 4 = 64 sequences/step
LR=1e-4
WARMUP=4000

# --- NCCL / PyTorch sanity for single node multi-GPU ---
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1     # single-node NVLink usually faster; avoid IB
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# training image
IMAGE=${THOR_IMAGE}

# Binds:
#  - current working dir (code, outputs, caches)
#  - data directory (if not inside $PWD), e.g., /scratch or /proj
# Add more -B mounts if your data lives elsewhere.
BIND_OPTS="-B $PWD:$PWD -B $THOR_DATA:/data"

# IMPORTANT: use absolute path inside container for torchrun
TCR=/opt/conda/bin/torchrun
SCRIPT=finetune_ESM.py

echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

export OMP_NUM_THREADS=8

# Run: one process per GPU, all within the container


 # choose a per-job port
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + ($SLURM_JOB_ID % 50000)))

srun --ntasks=1 singularity exec --nv \
  $BIND_OPTS \
  --pwd $PWD \
  --env OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} \
  --env HF_HOME=$HF_HOME \
  --env HF_DATASETS_CACHE=$HF_DATASETS_CACHE \
  --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
  --env TORCH_HOME=$TORCH_HOME \
  --env NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
  --env NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} \
  --env NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} \
  --env CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} \
  --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
  "$IMAGE" \
  $TCR \
    --nproc_per_node=${SLURM_GPUS_ON_NODE:-4} \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    $SCRIPT \
      --model_name "$MODEL_ID" \
      --data_path "$DATA_PATH/$DATA" \
      --val_path  "$DATA_PATH/$DATA_VAL" \
      --eval_steps 2000 \
      --output_dir "$OUT_DIR" \
      --max_length 1022 \
      --epochs "$EPOCHS" \
      --per_device_batch "$PER_DEVICE_BATCH" \
      --accum "$ACCUM" \
      --lr "$LR" \
      --warmup_steps "$WARMUP" \
      --resume "auto" \
      $BF16_FLAG
