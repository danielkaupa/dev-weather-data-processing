#!/usr/bin/env bash
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -N step1d_generate_country_mask

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step1d_generate_country_mask}
JOBID=${PBS_JOBID:-$$}
mkdir -p logs
exec 1>logs/${JOBNAME}.o${JOBID}
exec 2>logs/${JOBNAME}.e${JOBID}
set -euo pipefail

module purge
module load tools/prod || true
module load miniforge/3 || true

# Load conda env
eval "$(~/miniforge3/bin/conda shell.bash hook)"   # adjust if needed
conda activate osme

# Thread control for xarray/shapely/eccodes
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== Job $JOBNAME (ID $JOBID) started on $(hostname) at $(date) ==="

# Sanity check for required deps
python - << 'EOF' || exit 1
import geopandas, shapely, eccodes, polars, xarray
print("GeoPandas OK:", geopandas.__version__)
print("Shapely OK:", shapely.__version__)
print("eccodes OK")
print("Polars OK:", polars.__version__)
print("Xarray OK:", xarray.__version__)
EOF

# ---------------------------------------------------------------------------
# CONFIGURATION (override these via: qsub -v GRIB_DIR=...,BOUNDARY_FILE=...)
# ---------------------------------------------------------------------------

GRIB_DIR=${GRIB_DIR:-"../data/raw"}
BOUNDARY_FILE=${BOUNDARY_FILE:-"geoBoundariesCGAZ_ADM0/India.geojson"}

MASK_DIR=${MASK_DIR:-"masks"}
METADATA_DIR=${METADATA_DIR:-"masks/mask_metadata"}
IMAGE_DIR=${IMAGE_DIR:-"images"}

DATASET_PREFIX=${DATASET_PREFIX:-""}

INCLUSION_MODE=${INCLUSION_MODE:-"centroid"}
FRACTION_THRESHOLD=${FRACTION_THRESHOLD:-0.5}

EXCLUSION_BBOX_JSON=${EXCLUSION_BBOX_JSON:-""}

REUSE_EXISTING=${REUSE_EXISTING:-""}
OVERWRITE_EXISTING=${OVERWRITE_EXISTING:-"1"}

GENERATE_IMAGE=${GENERATE_IMAGE:-""}

PARALLEL_BACKEND=${PARALLEL_BACKEND:-"processpool"}
PARALLEL_MIN_CELLS=${PARALLEL_MIN_CELLS:-10000}
PARALLEL_CHUNK_SIZE=${PARALLEL_CHUNK_SIZE:-5000}

LOGLEVEL=${LOGLEVEL:-"INFO"}

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

CMD=( python step1d_generate_country_mask.py
      --grib-dir "$GRIB_DIR"
      --boundary-file "$BOUNDARY_FILE"
      --mask-dir "$MASK_DIR"
      --metadata-dir "$METADATA_DIR"
      --image-dir "$IMAGE_DIR"
      --inclusion-mode "$INCLUSION_MODE"
      --fraction-threshold "$FRACTION_THRESHOLD"
      --parallel-backend "$PARALLEL_BACKEND"
      --parallel-min-cells "$PARALLEL_MIN_CELLS"
      --parallel-chunk-size "$PARALLEL_CHUNK_SIZE"
      --log "$LOGLEVEL"
)

[[ -n "$DATASET_PREFIX" ]] && CMD+=( --dataset-prefix "$DATASET_PREFIX" )
[[ -n "$EXCLUSION_BBOX_JSON" ]] && CMD+=( --exclusion-bbox-json "$EXCLUSION_BBOX_JSON" )

# boolean flags
[[ "$REUSE_EXISTING" == "1" ]] && CMD+=( --reuse-existing )
[[ "$OVERWRITE_EXISTING" == "1" ]] && CMD+=( --overwrite-existing )
[[ "$GENERATE_IMAGE" == "1" ]] && CMD+=( --generate-image )

echo "Running:"
printf '%q ' "${CMD[@]}"
echo -e "\n"

"${CMD[@]}"

echo "=== Job $JOBNAME (ID $JOBID) finished on $(hostname) at $(date) ==="
