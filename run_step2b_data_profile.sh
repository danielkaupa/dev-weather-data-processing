#!/usr/bin/env bash
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:10:00
#PBS -N data_profile

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-data_profile}
JOBID=${PBS_JOBID:-$$}

mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod || true
module load miniforge/3 || true

# Activate your conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate osme

# Reduce threaded libraries (Polars benefits from parallelism,
# but on HPC it's safer to restrict)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== Profiling job started on $(hostname) at $(date) ==="

# Change this to your target Parquet file or provide it via CLI
INPUT_FILE="../data/interim/era5-world_INDIA_d514a3a3c256_2025_06.parquet"

# Example run – produces both TXT and CSV reports
python step2b_data_profile.py "$INPUT_FILE" \
    --report-format both \
    --log INFO

echo "=== Profiling job finished on $(hostname) at $(date) ==="
