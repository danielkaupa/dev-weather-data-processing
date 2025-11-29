#!/usr/bin/env bash
#PBS -l select=1:ncpus=30:mpiprocs=30:mem=100gb
#PBS -l walltime=01:00:00
#PBS -N step2a_process_grib_mpi

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step2a_process_grib_mpi}
JOBID=${PBS_JOBID:-$$}

mkdir -p logs
exec >logs/${JOBNAME}.o${JOBID} 2>logs/${JOBNAME}.e${JOBID}

set -euo pipefail

# ------------------------------------------------------------------------------
# Load modules
# ------------------------------------------------------------------------------
module purge
module load tools/prod || true
module load miniforge/3 || true

# ------------------------------------------------------------------------------
# Important: DO NOT activate conda here for main shell.
#            Instead we activate inside the mpiexec-spawned shell.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Thread limits (avoid BLAS oversubscription)
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== MPI Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="
echo "Node allocation summary:"
uniq -c "$PBS_NODEFILE"

# ------------------------------------------------------------------------------
# Variables (can be overridden via qsub -v)
# ------------------------------------------------------------------------------
GRIB_DIR=${GRIB_DIR:-"../data/raw"}
MASK_PARQUET=${MASK_PARQUET:-"masks/era5-world_INDIA_mask_centroid_264612.parquet"}
MASK_META=${MASK_META:-"masks/mask_metadata/era5-world_INDIA_mask_centroid_264612.json"}
OUTPUT_DIR=${OUTPUT_DIR:-"../data/interim"}

LOGLEVEL=${LOGLEVEL:-"INFO"}

# ------------------------------------------------------------------------------
# Build command
# ------------------------------------------------------------------------------
CMD="step2a_mask_and_process_grib.py \
    --grib-dir $GRIB_DIR \
    --mask-parquet $MASK_PARQUET \
    --mask-meta $MASK_META \
    --output-dir $OUTPUT_DIR \
    --backend mpi \
    --log $LOGLEVEL"

NP=$(wc -l < "$PBS_NODEFILE")
echo "MPI ranks allocated: $NP"

echo "mpiexec -np $NP bash -lc \"source ~/miniforge3/bin/activate osme && python $CMD\""
echo

# ------------------------------------------------------------------------------
# ACTUAL MPI RUN
# Every rank starts a login shell (bash -lc), activates conda,
# then launches python with mpi4py correctly available.
# ------------------------------------------------------------------------------
mpiexec -np "$NP" bash -lc "source ~/miniforge3/bin/activate osme && python $CMD"

echo "=== Job finished at $(date) ==="
