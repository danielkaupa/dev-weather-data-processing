#!/usr/bin/env bash
#PBS -l select=1:ncpus=1:mem=32gb
#PBS -l walltime=01:30:00
#PBS -N step1a_extract_country_boundary

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step1a_extract_country_boundary}
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

# Avoid over-threading (GeoPandas + GDAL behave better this way)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="

# Optional: verify module import
python -c "import geopandas as gpd; print('GeoPandas OK, version:', getattr(gpd,'__version__','?'))" || exit 1

# ---------------------------------------------------------------------------
# CONFIGURATION (edit these or pass via qsub environment variables)
# ---------------------------------------------------------------------------

SHAPEFILE=${SHAPEFILE:-"geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp"}
COUNTRY=${COUNTRY:-"India"}
FIELD=${FIELD:-"shapeName"}
OUTDIR=${OUTDIR:-"geoBoundariesCGAZ_ADM0/"}
LOGLEVEL=${LOGLEVEL:-"INFO"}

# ---------------------------------------------------------------------------
# RUN SCRIPT
# ---------------------------------------------------------------------------

python step1a_extract_country_boundary.py \
    --shapefile "$SHAPEFILE" \
    --country "$COUNTRY" \
    --field "$FIELD" \
    --outdir "$OUTDIR" \
    --log "$LOGLEVEL"

echo "=== Job $JOBNAME (ID $JOBID) finished on $(hostname) at $(date) ==="
