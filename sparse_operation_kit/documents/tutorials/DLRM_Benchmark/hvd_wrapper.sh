#!/bin/bash

# Get local process ID from OpenMPI or alternatively from SLURM
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then
        LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"
    elif [ -n "${SLURM_LOCALID:-}" ]; then
        LOCAL_RANK="${SLURM_LOCALID}"
    fi
    export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
fi
echo "[hvd_wrapper] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

exec "$@"

