#!/bin/bash
#SBATCH --job-name=Predict_Temperature
#SBATCH --account=NAISS2025-5-369
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --output=%x.%j.out

# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

DATA_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/Design_ProteinMPNN #Different_target_identity #Different_target_identity #Experiment_11137_thermo


WEIGHTS_CLASSIFIER_DIR=${SRC_DIR}/weights/OGT_ensemble

IMAGE=${SRC_DIR}/env/singularity/test_6.sif # thermalgan.sif


time singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${SRC_DIR}:/ThermalGAN, \
--nv ${IMAGE}  python predict_OGT_from_fasta.py /data/best_variants_70.fasta /data/best_variants_70_TM.fasta
#done
