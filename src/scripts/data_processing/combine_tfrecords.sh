#!/bin/bash


# Directory definitions
SRC_DIR=/mimer/NOBACKUP/groups/snic2022-6-127/sandra/ThermalGAN

DATA_DIR=${SRC_DIR}/data/THOR_BIG 

IMAGE=${SRC_DIR}/env/singularity/test_6.sif # thermalgan.sif



singularity exec \
-H $(pwd) \
--bind ${DATA_DIR}:/data,${SRC_DIR}:/ThermalGAN,/cephyr/users/sandravi/Alvis/.local/lib/python3.9/site-packages/simple_pid:/opt/conda/lib/python3.9/site-packages/simple_pid \
--nv ${IMAGE}  python combine_tfrecords.py \

#done
