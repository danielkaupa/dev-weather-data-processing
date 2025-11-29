#!/usr/bin/env bash
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=01:00:00
#PBS -N step1b_plot_country_boundary

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step1b_plot_country_boundary}
JOBID=${PBS_JOBID:-$$}
mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod || true
module load miniforge/3 || true

# Load conda
eval "$(~/miniforge3/bin/conda shell.bash hook)"   # adjust if needed
conda activate osme

# Avoid over-threading (matplotlib, GDAL)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="

# Optional: verify GeoPandas and Matplotlib import
python - << 'EOF' || exit 1
import geopandas as gpd, matplotlib
print("GeoPandas:", gpd.__version__)
print("Matplotlib:", matplotlib.__version__)
EOF

# ---------------------------------------------------------------------------
# CONFIGURATION (override via qsub -v)
# ---------------------------------------------------------------------------

GEOJSON=${GEOJSON:-"geoBoundariesCGAZ_ADM0/India.geojson"}
TITLE=${TITLE:-"India Boundary Verification"}
SAVE=${SAVE:-"India_boundary_verification.png"}
LOGLEVEL=${LOGLEVEL:-"INFO"}

# ---------------------------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------------------------

# If SAVE is empty, we don't pass --save
if [[ -z "$SAVE" ]]; then
    python step1b_plot_country_boundary.py \
        --geojson "$GEOJSON" \
        --title "$TITLE" \
        --log "$LOGLEVEL"
else
    python step1b_plot_country_boundary.py \
        --geojson "$GEOJSON" \
        --title "$TITLE" \
        --save "$SAVE" \
        --log "$LOGLEVEL"
fi

echo "=== Job $JOBNAME (ID $JOBID) finished on $(hostname) at $(date) ==="
