#!/bin/bash
#SBATCH --job-name=THOR_example_features
#SBATCH --account=CHANGE_ME     # site-specific; or delete this line
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --output=%x.%j.out

# Compute ESM-1v features for the four example splits and write them as
# TFRecords, ready for train_thor.py.
#
#     sbatch data/example_MDH/prepare_example.sh
#     # or directly on a machine with a GPU:
#     bash   data/example_MDH/prepare_example.sh
#
# Produces, under data/example_MDH/records/:
#     meso_train/   meso_val/   thermo_train/   thermo_val/
#
# Needs a GPU. On an A100 this takes a few minutes; on a smaller card, longer.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths. Nothing here needs editing for a normal clone.
# ---------------------------------------------------------------------------
if [ -z "${THOR_ROOT:-}" ]; then
    _here=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}
    THOR_ROOT=$(cd "$_here" 2>/dev/null && git rev-parse --show-toplevel 2>/dev/null) || THOR_ROOT=""
    [ -n "$THOR_ROOT" ] || THOR_ROOT=$(cd "$_here/../.." && pwd)
fi
THOR_IMAGE=${THOR_IMAGE:-$THOR_ROOT/thor.sif}
THOR_DATA=${THOR_DATA:-$THOR_ROOT/data}

# ESM model used to compute per-position likelihoods. Stock ESM-1v is what the
# published results use. Override to point at a local directory if you want to
# avoid re-downloading the weights on a cluster without outbound network.
ESM_MODEL=${ESM_MODEL:-facebook/esm1v_t33_650M_UR90S_1}

echo "repo   : $THOR_ROOT"
echo "image  : $THOR_IMAGE"
echo "esm    : $ESM_MODEL"

if [ ! -f "$THOR_IMAGE" ]; then
    echo "No container at $THOR_IMAGE"
    echo "  singularity pull thor.sif docker://ghcr.io/umbriona/thor:latest"
    exit 1
fi

mkdir -p "$THOR_DATA/example_MDH/records"

for SPLIT in meso_train meso_val thermo_train thermo_val; do
    echo
    echo "=== ${SPLIT} ==========================================="
    singularity exec --nv \
        --bind "${THOR_ROOT}":/ThermalGAN,"${THOR_DATA}":/data \
        "${THOR_IMAGE}" \
        python3 /ThermalGAN/src/scripts/data_processing/compute_MLM_features.py \
            -i "/data/example_MDH/${SPLIT}.fasta" \
            -o "/data/example_MDH/records/${SPLIT}" \
            --name "${SPLIT}" \
            --model "${ESM_MODEL}" \
            --mode random \
            --mask_prob 0.15 \
            --random_batch 16 \
            --quantize uint8 \
            --format tfrecord \
            --tfrecord_compress gzip \
            --max_entries 2000 \
            --batch_tokens 20000 \
            --bf16
done

echo
echo "Done. Records written to ${THOR_DATA}/example_MDH/records/"
find "${THOR_DATA}/example_MDH/records" -name '*.tfrecord*' | sort | sed 's|^|  |'
echo
echo "Next: fine-tune with"
echo "  singularity exec --nv ${THOR_IMAGE} python3 \\"
echo "      /ThermalGAN/src/scripts/train_thor.py \\"
echo "      -c /ThermalGAN/config/Cycle_gan/MDH_finetune.yaml"
