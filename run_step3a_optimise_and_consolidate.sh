#!/usr/bin/env bash
#PBS -l select=2:ncpus=6:mpiprocs=6:mem=400gb
#PBS -l place=scatter:excl
#PBS -l walltime=1:00:00
#PBS -N step3a_optimise_and_consolidate_mpi

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step3a_optimise_and_consolidate_mpi}
JOBID=${PBS_JOBID:-$$}

mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

###############################################################################
# MODULES
###############################################################################
module purge
module load tools/prod || true
module load miniforge/3 || true

###############################################################################
# ACTIVATE ENV
###############################################################################
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate osme

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

###############################################################################
# FIX 1 — Clean hostnames for MPICH
###############################################################################
NODEFILE_CLEAN=clean_nodefile_${JOBID}.txt
sed 's/\..*//' "$PBS_NODEFILE" > "$NODEFILE_CLEAN"

###############################################################################
# FIX 2 — MPICH Launcher stabilisation
###############################################################################
export HYDRA_LAUNCHER=fork
export HYDRA_DEBUG=0

###############################################################################
# INFO
###############################################################################
NP=$(wc -l < "$NODEFILE_CLEAN")
echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="
echo "=== MPI ranks: $NP ==="
echo "=== Clean nodefile ==="
awk '{print "  "$0}' "$NODEFILE_CLEAN"

###############################################################################
# RUN (MPICH requires -f and -n explicitly)
###############################################################################
mpiexec -f "$NODEFILE_CLEAN" -n "$NP" \
    python step3a_optimise_and_consolidate.py \
        --sample-file "../data/interim/era5-world_INDIA_d514a3a3c256_2025_06.parquet" \
        --input-dir "../data/interim" \
        --clean-dir "../data/temp/temp_clean" \
        --agg-dir "../data/temp/temp_agg" \
        --output-dir "../data/processed" \
        --metadata-json "../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json" \
        --modes annual biannual quarterly \
        --overwrite \
        --cleanup-temp \
        --log-level INFO

echo "=== Job $JOBNAME (ID $JOBID) finished at $(date) ==="
