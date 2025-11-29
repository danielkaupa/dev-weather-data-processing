
#!/usr/bin/env bash
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -l walltime=00:10:00
#PBS -N step2c_update_metadata_with_processing_names

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step2c_update_metadata_with_processing_names}
JOBID=${PBS_JOBID:-$$}

mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod || true
module load miniforge/3 || true

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate osme

echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="

# ------------------------------------------------------------
# Run script
# Override input/output if desired (default paths are in script)
# ------------------------------------------------------------

python step2c_update_metadata_with_processing_names.py \
    --input  "../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json" \
    --output "../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json"

echo "=== Job $JOBNAME (ID $JOBID) finished at $(date) ==="
