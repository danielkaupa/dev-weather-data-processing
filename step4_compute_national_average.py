#!/usr/bin/env python3
"""
step4_compute_national_average.py
==================================

Compute national-average ERA5 time series from gridded ERA5 data.

Supports two modes:

1. Normal mode (default):
   Given --input <yearfile>, discover ALL matching yearly ERA5 files in the
   same directory and process them sequentially (or using multiprocessing).

2. MPI single-file mode (--single):
   Process ONLY the provided input file. Used when MPI distributes one file
   per rank.

Outputs use atomic writes (tmp → rename) and are validated after writing
to prevent corrupted Parquet files on shared HPC filesystems.

Files eligible for processing must end with "_YYYY.parquet" and NOT contain:
- "national"
- spaces
- month indicators like _01
- H1/H2
- any non-year suffix

This ensures that input+output can safely coexist in the same directory.

"""

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List
import polars as pl
import re


# ======================================================================
# Atomic Parquet Write
# ======================================================================

def atomic_sink_parquet(lazyframe: pl.LazyFrame, output_path: Path) -> None:
    """Write parquet atomically and validate."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    lazyframe.sink_parquet(str(tmp_path))
    tmp_path.replace(output_path)

    try:
        _ = pl.scan_parquet(str(output_path)).fetch(1)
    except Exception as e:
        raise RuntimeError(f"Output parquet corrupted: {output_path}") from e


# ======================================================================
# Compute National Average for One File
# ======================================================================

def compute_single_file(input_path: Path, output_dir: Path, overwrite: bool) -> None:
    """Compute national-average ERA5 time series for one yearly file."""
    output_path = output_dir / f"{input_path.stem}_national.parquet"

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Exists: {output_path}")
        return

    print(f"[RUN ] Input : {input_path}")
    print(f"[OUT ] Output: {output_path}")

    static_cols = {
        "latitude",
        "longitude",
        "frac_in_region",
        "centroid_in_region",
        "cell_area_m2",
    }

    lf = pl.scan_parquet(str(input_path))
    schema = lf.collect_schema()

    numeric_cols = [
        col
        for col, dtype in schema.items()
        if (
            col not in static_cols
            and dtype.is_numeric()
            and dtype != pl.Boolean
        )
    ]

    result = (
        lf.group_by("time")
          .agg(pl.col(numeric_cols).mean())
          .sort("time")
    )

    atomic_sink_parquet(result, output_path)
    print(f"[DONE] {input_path}")


# ======================================================================
# Discover Matching Yearly Files
# ======================================================================

def discover_matching_files(example_file: Path) -> List[Path]:
    """
    Return all files in the same directory ending with _YYYY.parquet.

    Excludes:
    - 'national'
    - spaces
    - monthly (_01), seasonal (H1/H2), or other partial files
    """
    directory = example_file.parent
    matched = []

    for f in directory.glob("*.parquet"):
        s = f.stem

        # must end with _YYYY
        if not re.search(r"_(19|20)\d{2}$", s):
            continue

        # exclude national output
        if "national" in s:
            continue

        # exclude bad/partial files with spaces
        if " " in s:
            continue

        matched.append(f)

    return sorted(matched)


# ======================================================================
# Main CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute national-average ERA5 data.")
    parser.add_argument("--input", required=True, help="Input file (must end with _YYYY.parquet).")
    parser.add_argument("--outdir", required=True, help="Directory to write national outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (non-MPI only).")
    parser.add_argument("--single", action="store_true", help="Process ONLY this input file (MPI mode).")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MPI SINGLE MODE: process only the given file and exit
    # ------------------------------------------------------------------
    if args.single:
        print(f"[MPI] Processing single file: {input_path}")
        compute_single_file(input_path, output_dir, args.overwrite)
        print("[MPI] Single-file complete.")
        return

    # ------------------------------------------------------------------
    # NORMAL MODE: discover all matching files
    # ------------------------------------------------------------------
    files = discover_matching_files(input_path)

    print(f"Found {len(files)} matching yearly files:")
    for f in files:
        print(" -", f)

    if args.jobs > 1:
        print(f"[MP] Running multiprocessing with {args.jobs} workers...")
        with mp.Pool(args.jobs) as pool:
            pool.starmap(
                compute_single_file,
                [(f, output_dir, args.overwrite) for f in files]
            )
    else:
        print("[SEQ] Running sequentially...")
        for f in files:
            compute_single_file(f, output_dir, args.overwrite)

    print("All files completed successfully.")


if __name__ == "__main__":
    main()
