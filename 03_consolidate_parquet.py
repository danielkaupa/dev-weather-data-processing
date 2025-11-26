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

from pathlib import Path
import logging
import re
from typing import List, Dict, Set
from dataclasses import dataclass

import time
import polars as pl
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

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
    r".*_INDIA_(?P<uid>[A-Za-z0-9]+)_(?P<year>\d{4})_(?P<month>\d{2})\.parquet$"
)

def parse_year_month(path: Path):
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename pattern not recognized: {path.name}")
    return int(m.group("year")), int(m.group("month")), m.group("uid")


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

def load_monthly_files() -> Dict[int, List[Path]]:
    files = sorted(INTERIM_DIR.glob("*INDIA*.parquet"))

    by_year: Dict[int, List[Path]] = {}
    for f in files:
        try:
            year, month, uid = parse_year_month(f)
        except Exception:
            logging.warning("Skipping unrecognized file: %s", f.name)
            continue

        by_year.setdefault(year, []).append((month, f))

    # sort by month
    for year in by_year:
        by_year[year] = [f for (_, f) in sorted(by_year[year], key=lambda x: x[0])]

    logging.info("Found %d years of data.", len(by_year))
    return by_year


# ---------------------------------------------------------------------
# COLUMN NORMALISATION
# ---------------------------------------------------------------------

def get_all_columns(files: List[Path]) -> Set[str]:
    all_cols: Set[str] = set()
    for f in files:
        schema = pl.scan_parquet(f).collect_schema()
        for c in schema.names():
            all_cols.add(c)
    return all_cols


def normalize(df: pl.DataFrame, all_cols: Set[str]) -> pl.DataFrame:
    missing = all_cols - set(df.columns)
    for col in missing:
        df = df.with_columns(pl.lit(None).alias(col))
    return df.select(sorted(all_cols))


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

def process_year_worker(year: int, files: List[Path],
                        modes: List[str], overwrite: bool):
    logging.basicConfig(
        level=LOG_LEVEL,
        format=f"%(asctime)s [%(levelname)s] [Year={year}] %(message)s",
    )

    stats_for_year: List[YearStats] = []
    all_cols = get_all_columns(files)
    uid = parse_year_month(files[0])[2]

    for mode in modes:
        mode_stats = YearStats(year=year, mode=mode, outputs=[])

        for idx, group in enumerate(group_files(files, mode), start=1):
            if not group:
                continue

            # Output name
            if mode == "annual":
                out_name = f"era5_INDIA_{uid}_{year}.parquet"
            elif mode == "biannual":
                half = "H1" if idx == 1 else "H2"
                out_name = f"era5_INDIA_{uid}_{year}_{half}.parquet"
            else:
                out_name = f"era5_INDIA_{uid}_{year}_Q{idx}.parquet"

            out_path = OUTPUT_DIR / out_name

            if out_path.exists() and not overwrite:
                continue

            start_wall = time.time()
            start_cpu = time.process_time()

            dfs = []
            for f in group:
                df = pl.read_parquet(f)
                df = normalize(df, all_cols)
                dfs.append(df)

            combined = pl.concat(dfs, how="vertical")
            combined.write_parquet(out_path, compression="snappy")

            cpu_elapsed = time.process_time() - start_cpu
            wall_elapsed = time.time() - start_wall

            mode_stats.outputs.append(
                FileStats(
                    output_name=out_name,
                    n_input_files=len(group),
                    cpu_time_sec=cpu_elapsed,
                    wall_time_sec=wall_elapsed,
                    row_count=combined.height
                )
            )

        stats_for_year.append(mode_stats)

    return stats_for_year


# ---------------------------------------------------------------------
# MAIN — PROCESSPOOL BACKEND
# ---------------------------------------------------------------------

def run_processpool(year_files, modes, overwrite):
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
        logging.info("YEAR %d — MODE: %s", ys.year, ys.mode)
        for out in ys.outputs:
            logging.info(
                "  %-36s  files=%d  rows=%d  CPU=%.2f  WALL=%.2f",
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
