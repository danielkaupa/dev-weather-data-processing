#!/usr/bin/env bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -N vis_masks

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-vis_masks}
JOBID=${PBS_JOBID:-$$}
mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod || true
module load miniforge/3 || true

# Load conda
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate osme

# Keep libraries from over-threading
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="

# Quick imports test
python - << 'EOF' || exit 1
import geopandas, shapely, polars, xarray, eccodes, matplotlib
print("GeoPandas:", geopandas.__version__)
print("Shapely:", shapely.__version__)
print("Polars:", polars.__version__)
print("Xarray:", xarray.__version__)
print("eccodes OK")
print("matplotlib:", matplotlib.__version__)
EOF

# ---------------------------------------------------------------------------
# CONFIGURATION — override these via: qsub -v GRIB_FILE=...,MASK_FILES=...
# ---------------------------------------------------------------------------

GRIB_FILE=${GRIB_FILE:-"../data/raw/era5-world_N37W68S6E98_d514a3a3c256_2018_01.grib"}
BOUNDARY_FILE=${BOUNDARY_FILE:-"geoBoundariesCGAZ_ADM0/India.geojson"}

MASK_FILES=${MASK_FILES:-"masks/era5-world_INDIA_mask_centroid_264612.parquet"}
OUTPUT_PNG=${OUTPUT_PNG:-"images/multi_mask_comparison.png"}
MAX_COLUMNS=${MAX_COLUMNS:-4}

# Convert comma-separated list → space-separated list
IFS=',' read -r -a MASK_ARRAY <<< "$MASK_FILES"

# ---------------------------------------------------------------------------
# Run the visualisation
# ---------------------------------------------------------------------------

CMD=( python step1e_visualise_masks.py )

# Since the script uses __main__ defaults, we override via env vars
# (the script uses fixed globals, so we rely on env overrides + symlinks if needed)

# Best option: call the compare function directly via Python
python - << EOF
from pathlib import Path
from step1e_visualise_masks import compare_multiple_masks

grib_file = Path("$GRIB_FILE")
boundary_file = Path("$BOUNDARY_FILE")
mask_files = [Path(p) for p in ${MASK_ARRAY[@]@Q}]
out_png = Path("$OUTPUT_PNG")

compare_multiple_masks(
    grib_file=grib_file,
    mask_files=mask_files,
    boundary_file=boundary_file,
    out_png=out_png,
    max_cols=int("$MAX_COLUMNS"),
)

print(f"[OK] Figure saved → {out_png}")
EOF

echo "=== Job $JOBNAME (ID $JOBID) finished on $(hostname) at $(date) ==="
