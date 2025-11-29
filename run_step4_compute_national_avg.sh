#!/usr/bin/env bash
#PBS -l select=2:ncpus=3:mpiprocs=3:mem=200gb
#PBS -l place=scatter:excl
#PBS -l walltime=1:00:00
#PBS -N step4_compute_national_average_mpi

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME:-step4_compute_national_average_mpi}
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
# MPICH FIXES
###############################################################################
NODEFILE_CLEAN=clean_nodefile_${JOBID}.txt
sed 's/\..*//' "$PBS_NODEFILE" > "$NODEFILE_CLEAN"
export HYDRA_LAUNCHER=fork
export HYDRA_DEBUG=0

###############################################################################
# USER INPUTS
###############################################################################
EXAMPLE_FILE="../data/processed/era5-world_INDIA_d514a3a3c256_2018.parquet"
OUTDIR="../data/processed/national/"
mkdir -p "$OUTDIR"

###############################################################################
# DISCOVER MATCHING FILES
###############################################################################
MATCH_FILES=($(python - <<EOF
from pathlib import Path
from step4_compute_national_average import discover_matching_files
example = Path("$EXAMPLE_FILE")
for f in discover_matching_files(example):
    print(f)
EOF
))

NUM_FILES=${#MATCH_FILES[@]}
NP=$(wc -l < "$NODEFILE_CLEAN")

echo "=== MPI ranks: $NP ==="
echo "=== Matching files: $NUM_FILES ==="
for f in "${MATCH_FILES[@]}"; do echo " - $f"; done
echo "==================================="

if [[ $NUM_FILES -eq 0 ]]; then
    echo "ERROR: No files found."
    exit 1
fi

###############################################################################
# BUILD FILE LIST
###############################################################################
FILES_LIST=files_${JOBID}.txt
printf "%s\n" "${MATCH_FILES[@]}" > "$FILES_LIST"

###############################################################################
# MPI EXECUTION
###############################################################################
mpiexec -f "$NODEFILE_CLEAN" -n "$NP" \
    bash -c '
        RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
        FILE=$(sed -n "$((RANK + 1))p" "'"$FILES_LIST"'")

        if [[ -z "$FILE" ]]; then
            echo "[Rank $RANK] No file assigned. Exiting."
            exit 0
        fi

        echo "[Rank $RANK] Processing: $FILE"

        python step4_compute_national_average.py \
            --input "$FILE" \
            --outdir "'"$OUTDIR"'" \
            --overwrite \
            --jobs 1 \
            --single

        echo "[Rank $RANK] Done."
    '

###############################################################################
# CLEANUP
###############################################################################
rm -f "$NODEFILE_CLEAN" "$FILES_LIST"

echo "=== Job finished ==="
