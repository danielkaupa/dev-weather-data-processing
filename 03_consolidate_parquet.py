"""
03_consolidate_parquet.py
Parallel-by-year, multi-mode consolidation of processed ERA5 Parquet files.

Features:
- Consolidates monthly Parquet into annual / biannual / quarterly
- Supports running multiple modes in a single execution
- Parallel backends: none | processpool | mpi
- Configurable MAX_WORKERS (default = 2/3 CPUs)
- Schema alignment across all files
- Pure concatenation (no math)
- Full statistics: per-file, per-year, per-mode, total CPU and wall time
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import logging
import re
from typing import List, Dict, Set
from dataclasses import dataclass

import time
import polars as pl
import multiprocessing

# Optional MPI
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# ---------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------

INTERIM_DIR = Path("../data/interim")
OUTPUT_DIR = INTERIM_DIR

# Example:
# CONSOLIDATION_MODES = ["annual"]
# CONSOLIDATION_MODES = ["annual", "quarterly"]
CONSOLIDATION_MODES = ["annual", "biannual", "quarterly"]

# Parallel backend
# Options: "none", "processpool", "mpi"
PARALLEL_BACKEND = "processpool"

# Max workers for processpool
# Default: 2/3 of CPU cores, min 2
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

OVERWRITE = False
LOG_LEVEL = logging.INFO

# COUNTRY TOKEN (match Script 02)
COUNTRY_TOKEN = "INDIA"   # e.g. "INDIA", "FRANCE", "USA", "BRAZIL"



# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# FILENAME PARSING
# ---------------------------------------------------------------------

FILE_RE = re.compile(
    fr".*_{COUNTRY_TOKEN}_(?P<uid>[A-Za-z0-9]+)_(?P<year>\d{{4}})_(?P<month>\d{{2}})\.parquet$"
)


def parse_year_month(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename pattern not recognized: {path.name}")
    return int(m.group("year")), int(m.group("month")), m.group("uid")


def parse_prefix_uid_year_month(path: Path):
    """
    Extract dataset prefix, UID, year, month from filenames like:
      era5-world_N37W68S6E98_d514a3a3c256_2024_02.parquet
    """
    m = re.match(
        r"^(?P<prefix>[^_]+_[^_]+)_"          # era5-world_N37W68S6E98
        r"(?P<uid>[A-Za-z0-9]+)_"             # d514a3a3c256
        r"(?P<year>\d{4})_"
        r"(?P<month>\d{2})\.parquet$",
        path.name
    )
    if not m:
        raise ValueError(f"Filename not recognized: {path.name}")

    return (
        m.group("prefix"),    # "era5-world_N37W68S6E98"
        m.group("uid"),       # "d514a3a3c256"
        int(m.group("year")),
        int(m.group("month")),
    )



# ---------------------------------------------------------------------
# STATISTICS STRUCTURES
# ---------------------------------------------------------------------

@dataclass
class FileStats:
    output_name: str
    n_input_files: int
    cpu_time_sec: float
    wall_time_sec: float
    row_count: int


@dataclass
class YearStats:
    year: int
    mode: str
    outputs: List[FileStats]


# ---------------------------------------------------------------------
# DISCOVER FILES
# ---------------------------------------------------------------------

def load_monthly_files() -> Dict[int, Dict]:
    """
    Returns:
    {
        2018: {
            "prefix": "...",
            "uid": "...",
            "files": [Path, Path, Path]
        },
        ...
    }
    """
    files = sorted(INTERIM_DIR.glob("*INDIA*.parquet"))

    by_year: Dict[int, Dict] = {}

    for f in files:
        try:
            prefix, uid, year, month = parse_prefix_uid_year_month(f)
        except Exception:
            logging.warning("Skipping unrecognized file: %s", f.name)
            continue

        if year not in by_year:
            by_year[year] = {
                "prefix": prefix,
                "uid": uid,
                "files": []
            }

        by_year[year]["files"].append((month, f))

    # sort by month
    for year in by_year:
        by_year[year]["files"] = [
            f for (_, f) in sorted(by_year[year]["files"], key=lambda x: x[0])
        ]

    logging.info("Found %d years of data.", len(by_year))
    return by_year


# ---------------------------------------------------------------------
# COLUMN NORMALISATION
# ---------------------------------------------------------------------

def get_all_columns(files: List[Path]) -> Dict[str, pl.DataType]:
    all_cols: Dict[str, pl.DataType] = {}
    for f in files:
        schema = pl.scan_parquet(f).collect_schema()
        for name, dtype in schema.items():
            # promote Float32 → Float64 globally
            if dtype == pl.Float32:
                dtype = pl.Float64
            all_cols[name] = dtype
    return all_cols



def normalize(df: pl.DataFrame, all_cols: Dict[str, pl.DataType]) -> pl.DataFrame:
    # Add missing columns
    missing = set(all_cols) - set(df.columns)
    for col in missing:
        df = df.with_columns(pl.lit(None).cast(all_cols[col]).alias(col))

    # Cast existing columns to expected dtype
    casts = []
    for col, dtype in all_cols.items():
        if col in df.columns and df.schema[col] != dtype:
            casts.append(pl.col(col).cast(dtype).alias(col))

    if casts:
        df = df.with_columns(casts)

    return df.select(sorted(all_cols.keys()))



# ---------------------------------------------------------------------
# GROUPING LOGIC
# ---------------------------------------------------------------------

def group_files(files: List[Path], mode: str):
    if mode == "annual":
        return [files]

    elif mode == "biannual":
        h1 = [f for f in files if parse_year_month(f)[1] <= 6]
        h2 = [f for f in files if parse_year_month(f)[1] > 6]
        return [h1, h2]

    elif mode == "quarterly":
        groups = []
        for i in range(4):
            low = 1 + 3 * i
            high = low + 2
            groups.append([f for f in files if low <= parse_year_month(f)[1] <= high])
        return groups

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------
# WORKER — PROCESS A SINGLE YEAR
# ---------------------------------------------------------------------

def process_year_worker(year: int, info: Dict, modes: List[str], overwrite: bool):
    prefix = info["prefix"]
    uid = info["uid"]
    files = info["files"]    # must be List[Path]

    logging.basicConfig(
        level=LOG_LEVEL,
        format=f"%(asctime)s [%(levelname)s] [Year={year}] %(message)s",
    )
    logging.info(f"Starting consolidation for year {year} ({len(files)} files)")


    stats_for_year: List[YearStats] = []

    # Determine union schema across all monthly files
    all_cols = get_all_columns(files)

    for mode in modes:
        mode_stats = YearStats(year=year, mode=mode, outputs=[])

        # group into annual / biannual / quarterly
        for idx, group in enumerate(group_files(files, mode), start=1):
            if not group:
                continue

            # build filename
            if mode == "annual":
                out_name = f"{prefix}_{uid}_{year}.parquet"

            elif mode == "biannual":
                half = "H1" if idx == 1 else "H2"
                out_name = f"{prefix}_{uid}_{year}_{half}.parquet"

            elif mode == "quarterly":
                out_name = f"{prefix}_{uid}_{year}_Q{idx}.parquet"

            out_path = OUTPUT_DIR / out_name

            if out_path.exists() and not overwrite:
                continue

            start_wall = time.time()
            start_cpu = time.process_time()

            # Load + normalise all monthly files
            dfs = []
            for f in group:
                df = pl.read_parquet(f)
                df = normalize(df, all_cols)
                dfs.append(df)

            combined = pl.concat(dfs, how="vertical")
            combined.write_parquet(out_path, compression="snappy")
            logging.info(f"Wrote {out_path} ({combined.height} rows)")

            cpu_elapsed = time.process_time() - start_cpu
            wall_elapsed = time.time() - start_wall

            mode_stats.outputs.append(
                FileStats(
                    output_name=out_name,
                    n_input_files=len(group),
                    cpu_time_sec=cpu_elapsed,
                    wall_time_sec=wall_elapsed,
                    row_count=combined.height,
                )
            )

        stats_for_year.append(mode_stats)

    return stats_for_year



# ---------------------------------------------------------------------
# MAIN — PROCESSPOOL BACKEND
# ---------------------------------------------------------------------

def run_processpool(year_files, modes, overwrite):
    from concurrent.futures import ProcessPoolExecutor  # ← ADD THIS
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(process_year_worker, y, files, modes, overwrite)
            for y, files in year_files.items()
        ]
        for f in futures:
            results.extend(f.result())
    return results



# ---------------------------------------------------------------------
# MAIN — MPI BACKEND
# ---------------------------------------------------------------------

def run_mpi(year_files, modes, overwrite):
    if MPI is None:
        raise RuntimeError("mpi4py not installed.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    years = sorted(year_files.keys())
    results_local = []

    for i, year in enumerate(years):
        if i % size == rank:
            res = process_year_worker(year, year_files[year], modes, overwrite)
            results_local.extend(res)

    gathered = comm.gather(results_local, root=0)

    if rank == 0:
        merged = []
        for chunk in gathered:
            merged.extend(chunk)
        return merged
    else:
        return []


# ---------------------------------------------------------------------
# MAIN CONTROLLER
# ---------------------------------------------------------------------

def consolidate():
    setup_logging()

    overall_wall_start = time.time()
    overall_cpu_start = time.process_time()

    year_files = load_monthly_files()

    if PARALLEL_BACKEND == "none":
        all_stats = []
        for y, files in year_files.items():
            all_stats.extend(process_year_worker(y, files, CONSOLIDATION_MODES, OVERWRITE))

    elif PARALLEL_BACKEND == "processpool":
        logging.info("Running in PROCESSPOOL mode (max workers = %d)", MAX_WORKERS)
        all_stats = run_processpool(year_files, CONSOLIDATION_MODES, OVERWRITE)

    elif PARALLEL_BACKEND == "mpi":
        logging.info("Running in MPI mode")
        all_stats = run_mpi(year_files, CONSOLIDATION_MODES, OVERWRITE)
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
    else:
        raise ValueError(f"Unknown backend {PARALLEL_BACKEND}")

    # Final stats
    wall_total = time.time() - overall_wall_start
    cpu_total = time.process_time() - overall_cpu_start

    logging.info("====================================================")
    logging.info("CONSOLIDATION COMPLETE")
    logging.info("Total WALL time: %.2f sec (%.2f min)", wall_total, wall_total / 60)
    logging.info("Total CPU time:  %.2f sec (%.2f min)", cpu_total, cpu_total / 60)
    logging.info("====================================================")

    # Per-year detail
    for ys in all_stats:
        logging.info("-----------------------------------------------")
        logging.info("YEAR %d — MODE: %s", ys.year, ys.mode)

        if not ys.outputs:
            logging.info("  (no output files for this year/mode)")
            continue

        for out in ys.outputs:
            logging.info(
                "  WROTE %-40s | files=%2d | rows=%8d | CPU=%.2fs | WALL=%.2fs",
                out.output_name,
                out.n_input_files,
                out.row_count,
                out.cpu_time_sec,
                out.wall_time_sec,
            )


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    consolidate()
