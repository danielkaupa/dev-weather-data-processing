#!/usr/bin/env bash
#PBS -l select=3:ncpus=10:mem=36gb:mpiprocs=10
#PBS -l walltime=01:00:00
#PBS -N step1c_extract_and_validate_variable_metadata

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step1c_extract_and_validate_variable_metadata}
JOBID=${PBS_JOBID:-$$}
mkdir -p logs
exec >logs/${JOBNAME}.o${JOBID} 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod
module load miniforge/3

# Load conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate osme

# Prevent over-threading in libraries
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1


echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="
echo "Node allocation summary:"
cat $PBS_NODEFILE | uniq -c

# ---------------------------------------------------------------------------
# CONFIGURATION (override via `qsub -v`)
# ---------------------------------------------------------------------------

GRIB_DIR=${GRIB_DIR:-"../data/raw"}
OUTPUT_DIR=${OUTPUT_DIR:-"../data/interim"}
DELAY=${DELAY:-1.0}
LOGLEVEL=${LOGLEVEL:-"INFO"}
# PROCESS_ALL=${PROCESS_ALL:-""}

which mpiexec

# Infer MPI worker count
MPI_NP=$(wc -l < "$PBS_NODEFILE")

echo "MPI ranks allocated: $MPI_NP"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

CMD=( python step1c_extract_and_validate_variable_metadata.py
      --grib-dir "$GRIB_DIR"
      --output-dir "$OUTPUT_DIR"
      --delay "$DELAY"
      --log "$LOGLEVEL"
      --processing-mode mpi
)

echo "Launching with MPI:"
printf 'mpiexec -n %d ' "$MPI_NP"
printf '%q ' "${CMD[@]}"
echo

mpiexec -n "$MPI_NP" "${CMD[@]}"

echo "=== Job $JOBNAME (ID $JOBID) finished on $(hostname) at $(date) ==="
