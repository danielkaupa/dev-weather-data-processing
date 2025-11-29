"""
step2a_mask_and_process_grib.py
=================================

Convert monthly ERA5-style GRIB files into *country-only* Parquet files
using a *pre-computed mask* and *mask metadata* created in Step 1.

This script DOES NOT:
- build masks
- validate mask/GRIB compatibility
- perform area weighting
- perform plotting

This script DOES:
- parse and group GRIB files by (dataset, coordinates, uid)
- allow filtering by dataset / coordinates / uid
- optionally process all groups or only the most recent one
- load each GRIB variable separately via cfgrib
- join xarray→pandas→polars output to a mask parquet (inner join)
- write trimmed Parquet files (one per month)
- parallelise both:
    * variable consistency checks, and
    * monthly processing

File format assumptions
-----------------------
GRIB filename pattern:

    <dataset>_<coordinates>_<uid>_<year>_<month>.grib

Example:
    era5-world_N37W68S6E98_d514a3a3c256_2018_01.grib

Output:
    <dataset>_<COUNTRYTOKEN>_<uid>_<year>_<month>.parquet

Example:
    era5-world_INDIA_d514a3a3c256_2018_01.parquet
"""

from __future__ import annotations

# ======================================================================
# Imports
# ======================================================================

import argparse
import logging
import multiprocessing
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

# Optional MPI backend (global rank/size)
try:  # noqa: SIM105
    from mpi4py import MPI  # type: ignore[attr-defined]

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    MPI_ENABLED = True
except Exception:  # noqa: BLE001
    MPI = None
    COMM = None
    RANK = 0
    SIZE = 1
    MPI_ENABLED = False


# ======================================================================
# Global configuration defaults
# ======================================================================

GRIB_DIR = Path("../data/raw/")
MASK_META = Path("masks/mask_metadata/era5-world_INDIA_mask_centroid_264612.json")
MASK_PARQUET = Path("masks/era5-world_INDIA_mask_centroid_264612.parquet")
OUTPUT_DIR = Path("../data/interim/")

# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND_DEFAULT = "mpi"

# Worker count for process pool
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Logging level
DEFAULT_LOG_LEVEL = "INFO"

# Overwrite behaviour
OVERWRITE_EXISTING = True

# Process all groups or only most recent when unfiltered?
PROCESS_ALL_GROUPS_DEFAULT = False


# ======================================================================
# Logging
# ======================================================================

def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """
    Configure logging.

    Parameters
    ----------
    level : {"DEBUG","INFO","WARNING","ERROR","CRITICAL"}
        Console log level.
    """
    fmt = (
        f"%(asctime)s [RANK {RANK}] [%(levelname)s] %(message)s"
        if MPI_ENABLED
        else "%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=fmt,
    )


# ======================================================================
# Time formatting
# ======================================================================

def format_hms(seconds: float) -> str:
    """
    Convert seconds → H:M:S (zero-padded).
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


# ======================================================================
# GRIB filename parsing + grouping
# ======================================================================

FNAME_PATTERN = re.compile(
    r"""
    ^(?P<dataset>[^_]+)_          # dataset
    (?P<coords>[^_]+)_            # coordinates (N37W68S6E98, etc.)
    (?P<uid>[^_]+)_               # uid/hash
    (?P<year>\d{4})_              # YYYY
    (?P<month>\d{2})\.grib$       # MM
    """,
    re.VERBOSE,
)


def parse_grib_name(path: Path) -> Dict[str, str]:
    """
    Parse an ERA5-style GRIB filename.

    Returns {"dataset","coords","uid","year","month"}.
    """
    m = FNAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match ERA5 format: {path.name}")
    return m.groupdict()


def group_grib_files_by_triplet(
    files: List[Path],
) -> Dict[Tuple[str, str, str], List[Path]]:
    """
    Group GRIB files by the triplet (dataset, coords, uid).
    """
    groups: Dict[Tuple[str, str, str], List[Path]] = {}
    for f in files:
        t = parse_grib_name(f)
        key = (t["dataset"], t["coords"], t["uid"])
        groups.setdefault(key, []).append(f)
    return groups


def build_output_filename(path: Path, region_token: str) -> str:
    """
    Build the output parquet filename for a GRIB file.

    Input:  dataset_coords_uid_year_month.grib
    Output: dataset_REGIONTOKEN_uid_year_month.parquet
    """
    t = parse_grib_name(path)
    return f"{t['dataset']}_{region_token}_{t['uid']}_{t['year']}_{t['month']}.parquet"


def parse_year_month(path: Path) -> Tuple[int, int]:
    """
    Extract year and month integers from a GRIB filename.
    """
    t = parse_grib_name(path)
    return int(t["year"]), int(t["month"])


# ======================================================================
# Variable scanning + consistency
# ======================================================================

def scan_parameters(path: Path) -> List[Dict]:
    """
    Scan a GRIB file and return a list of variable descriptors.

    Each descriptor is {"paramId": int, "shortName": str}.
    """
    params: Dict[int, Dict] = {}

    try:
        with open(path, "rb") as f:
            while True:
                try:
                    gid = codes_grib_new_from_file(f)
                except CodesInternalError:
                    break
                if gid is None:
                    break

                try:
                    sn = codes_get(gid, "shortName")
                    pid = codes_get(gid, "paramId")
                    if isinstance(sn, bytes):
                        sn = sn.decode("utf-8")
                    params[int(pid)] = {"paramId": int(pid), "shortName": str(sn)}
                finally:
                    codes_release(gid)
    except Exception:  # noqa: BLE001
        logging.exception("Error scanning parameters: %s", path)

    return list(params.values())


def _consistency_check_one(
    path: Path,
    reference: Set[str],
) -> Tuple[str, List[str], List[str]]:
    """
    Compare one file's variable set to the reference shortName set.
    """
    vars_this = {p["shortName"] for p in scan_parameters(path)}
    missing = sorted(reference - vars_this)
    extra = sorted(vars_this - reference)
    return path.name, missing, extra


def _log_consistency_summary(
    mismatches: List[Tuple[str, List[str], List[str]]],
    reference_file: Path,
    total_files: int,
) -> None:
    """
    Log summary of consistency check results (called on rank 0).
    """
    if not mismatches:
        logging.info(
            "Variable consistency check complete — all %d files share the same "
            "shortName set as %s",
            total_files,
            reference_file.name,
        )
        return

    logging.warning(
        "Variable consistency check complete — %d files differ from %s",
        len(mismatches),
        reference_file.name,
    )
    for fname, missing, extra in mismatches:
        logging.warning("  %s:", fname)
        if missing:
            logging.warning("    Missing: %s", ", ".join(missing))
        if extra:
            logging.warning("    Extra:   %s", ", ".join(extra))


def verify_variable_consistency_parallel(
    files: List[Path],
    reference_shortnames: Set[str],
    backend: str,
) -> None:
    """
    Parallelised consistency check.

    Parameters
    ----------
    files : list[Path]
        Files to check.
    reference_shortnames : set[str]
        Expected shortName set (from reference file).
    backend : {"none","processpool","mpi"}
        How to parallelise:
        - "none": only rank 0, sequential.
        - "processpool": only rank 0, ProcessPoolExecutor.
        - "mpi": split across ranks by index (i % SIZE).

    Notes
    -----
    All logging summaries happen on rank 0. Other ranks do only their
    local work and then participate in collective operations.
    """
    if not files:
        if RANK == 0:
            logging.warning("No files provided for variable consistency check.")
        return

    reference_file = files[0]

    # ----------------- BACKEND: MPI -----------------
    if backend == "mpi":
        if not MPI_ENABLED or SIZE == 1:
            # Fall back to sequential on a single process
            if RANK == 0:
                logging.info(
                    "Backend 'mpi' requested but MPI world size=%d; "
                    "falling back to sequential consistency check.",
                    SIZE,
                )
                mismatches: List[Tuple[str, List[str], List[str]]] = []
                for f in files:
                    fname, missing, extra = _consistency_check_one(f, reference_shortnames)
                    if missing or extra:
                        mismatches.append((fname, missing, extra))
                _log_consistency_summary(mismatches, reference_file, len(files))
            return

        # MPI-enabled path: each rank checks a subset
        local_mismatches: List[Tuple[str, List[str], List[str]]] = []

        if RANK == 0:
            logging.info(
                "Checking variable consistency across %d files using MPI (%d ranks)...",
                len(files),
                SIZE,
            )

        for idx, f in enumerate(files):
            if idx % SIZE != RANK:
                continue
            fname, missing, extra = _consistency_check_one(f, reference_shortnames)
            if missing or extra:
                local_mismatches.append((fname, missing, extra))

        # Gather mismatches to rank 0
        all_mismatches = COMM.gather(local_mismatches, root=0)  # type: ignore[arg-type]
        COMM.Barrier()  # type: ignore[call-arg]

        if RANK == 0:
            flat: List[Tuple[str, List[str], List[str]]] = [
                x for sub in all_mismatches for x in sub
            ]
            _log_consistency_summary(flat, reference_file, len(files))
        return

    # ----------------- BACKEND: PROCESSPOOL -----------------
    if backend == "processpool":
        if RANK != 0:
            # Other ranks stay idle in this phase
            return

        logging.info(
            "Checking variable consistency across %d files using a local process pool (%d workers)...",
            len(files),
            MAX_WORKERS,
        )
        from concurrent.futures import ProcessPoolExecutor

        args = [(f, reference_shortnames) for f in files]
        mismatches: List[Tuple[str, List[str], List[str]]] = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for fname, missing, extra in ex.map(
                lambda p: _consistency_check_one(p[0], p[1]), args
            ):
                if missing or extra:
                    mismatches.append((fname, missing, extra))

        _log_consistency_summary(mismatches, reference_file, len(files))
        return

    # ----------------- BACKEND: NONE (SEQUENTIAL) -----------------
    if backend == "none":
        if RANK != 0:
            return

        logging.info(
            "Checking variable consistency across %d files sequentially on rank 0...",
            len(files),
        )
        mismatches = []
        for f in files:
            fname, missing, extra = _consistency_check_one(f, reference_shortnames)
            if missing or extra:
                mismatches.append((fname, missing, extra))

        _log_consistency_summary(mismatches, reference_file, len(files))
        return

    raise ValueError(f"Unknown backend for consistency check: {backend}")


# ======================================================================
# GRIB → xarray → polars
# ======================================================================

def open_single_variable_dataset(path: Path, shortname: str) -> xr.Dataset:
    """
    Open a single ERA5 variable via cfgrib, flattening forecast-style
    (time, step, valid_time) to a single time dimension.

    Returns Dataset with dims (time, latitude, longitude).
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": shortname}},
    )

    # Forecast-style flattening
    if "step" in ds.dims and "valid_time" in ds.coords:
        other_dims = [d for d in ds.dims if d not in ("time", "step")]
        stacked = ds.stack(time_step=("time", "step"))

        flat_time = pd.to_datetime(stacked["valid_time"].values.ravel())

        vars_out = {}
        for var_name, da in stacked.data_vars.items():
            data = da.transpose("time_step", *other_dims).values
            vars_out[var_name] = (["time", *other_dims], data)

        coords = {"time": flat_time}
        for dim in other_dims:
            coords[dim] = ds[dim]

        out = xr.Dataset(vars_out, coords=coords)
        ds.close()
        return out

    # Analysis-style
    if "time" in ds.coords:
        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))

    return ds


def load_grib_to_xarray(path: Path, shortnames: List[str]) -> xr.Dataset:
    """
    Load a GRIB file by opening each variable separately then merging.
    """
    datasets = [open_single_variable_dataset(path, sn) for sn in shortnames]
    merged = xr.merge(
        datasets,
        combine_attrs="override",
        compat="override",
        join="outer",
    )
    for ds in datasets:
        ds.close()
    return merged


# ======================================================================
# Single-file processing
# ======================================================================

def process_one_grib(
    path: Path,
    mask_parquet: Path,
    var_shortnames: List[str],
    region_token: str,
    output_dir: Path,
) -> float:
    """
    Process a single GRIB file into a country-only Parquet file.

    Returns wall time in seconds for this file.
    """
    start = time.time()

    year, month = parse_year_month(path)
    out_name = build_output_filename(path, region_token)
    out_path = output_dir / out_name

    if out_path.exists() and not OVERWRITE_EXISTING:
        logging.info("Skipping existing: %s", out_path.name)
        return 0.0

    logging.info("Processing %s → %s", path.name, out_name)

    # Load GRIB
    ds = load_grib_to_xarray(path, var_shortnames)
    df = ds.to_dataframe().reset_index()
    ds.close()

    pl_df = pl.from_pandas(df)

    # Join with mask
    lf = pl_df.lazy()
    mask_lf = pl.scan_parquet(mask_parquet)
    lf = lf.join(mask_lf, on=["latitude", "longitude"], how="inner")

    # Final write
    output_dir.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(out_path, compression="snappy")

    elapsed = time.time() - start
    logging.info("Finished %s in %.2f sec", out_name, elapsed)
    return elapsed


# ======================================================================
# Backend dispatch
# ======================================================================

def run_sequential(
    files: List[Path],
    mask_parquet: Path,
    var_shortnames: List[str],
    region_token: str,
    output_dir: Path,
) -> List[float]:
    """
    Run processing sequentially.
    """
    return [
        process_one_grib(f, mask_parquet, var_shortnames, region_token, output_dir)
        for f in files
    ]


def _worker(args) -> float:
    """
    Worker function for process pool.
    """
    (f, mask_parquet, var_shortnames, region_token, output_dir) = args
    setup_logging("INFO")
    return process_one_grib(f, mask_parquet, var_shortnames, region_token, output_dir)


def run_processpool(
    files: List[Path],
    mask_parquet: Path,
    var_shortnames: List[str],
    region_token: str,
    output_dir: Path,
) -> List[float]:
    """
    Run processing using a process pool.
    """
    from concurrent.futures import ProcessPoolExecutor

    args = [
        (f, mask_parquet, var_shortnames, region_token, output_dir)
        for f in files
    ]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(_worker, args))


def run_mpi(
    files: List[Path],
    mask_parquet: Path,
    var_shortnames: List[str],
    region_token: str,
    output_dir: Path,
) -> List[float]:
    """
    Run processing using MPI (index-based splitting across ranks).

    Returns flat list of durations on rank 0; empty list on other ranks.
    """
    if not MPI_ENABLED:
        raise RuntimeError("mpi4py / MPI not available but backend='mpi' was requested.")

    local_times: List[float] = []

    for i, f in enumerate(files):
        if i % SIZE == RANK:
            t = process_one_grib(f, mask_parquet, var_shortnames, region_token, output_dir)
            local_times.append(t)

    COMM.Barrier()  # type: ignore[call-arg]
    all_times = COMM.gather(local_times, root=0)  # type: ignore[arg-type]
    COMM.Barrier()  # type: ignore[call-arg]

    if RANK == 0:
        return [t for sub in all_times for t in sub]
    return []


# ======================================================================
# Group selection
# ======================================================================

def select_groups(
    groups: Dict[Tuple[str, str, str], List[Path]],
    dataset: str | None,
    coords: str | None,
    uid: str | None,
    process_all: bool,
) -> Dict[Tuple[str, str, str], List[Path]]:
    """
    Select appropriate GRIB groups based on filters and recency.
    """
    # Apply filters
    if dataset or coords or uid:
        groups = {
            k: v
            for k, v in groups.items()
            if (dataset is None or k[0] == dataset)
            and (coords is None or k[1] == coords)
            and (uid is None or k[2] == uid)
        }

    if not groups:
        raise RuntimeError("No matching GRIB groups found.")

    if len(groups) == 1:
        return groups

    # Multiple groups and no filters
    if process_all:
        logging.warning("Multiple groups found — processing all.")
        return groups

    logging.warning("Multiple groups found — selecting the most recent.")

    def newest_mtime(item: Tuple[Tuple[str, str, str], List[Path]]) -> float:
        """
        Return the newest modification time among a group's files.
        """
        _, files = item
        return max(f.stat().st_mtime for f in files)

    selected = max(groups.items(), key=newest_mtime)
    return {selected[0]: selected[1]}


# ======================================================================
# Main logic (two paths: MPI backend vs others)
# ======================================================================

def _single_process_main(args) -> None:
    """
    Main pipeline when backend is 'none' or 'processpool'.

    This path uses only a single process (rank 0). If the script is
    launched under mpiexec by accident, non-zero ranks exit early.
    """
    overall_start = time.time()
    setup_logging(args.log)

    if MPI_ENABLED and SIZE > 1 and RANK != 0:
        logging.info(
            "Backend '%s' only uses rank 0; rank %d exiting early.",
            args.backend,
            RANK,
        )
        return

    if RANK != 0:
        return  # extra guard

    logging.info("==================================================")
    logging.info("Running step2a_mask_and_process_grib.py (single-process path)")
    logging.info("==================================================")
    logging.info("Configuration used:")
    logging.info("\t\tGRIB source directory : %s", args.grib_dir)
    logging.info("\t\tMask parquet          : %s", args.mask_parquet)
    logging.info("\t\tMask metadata         : %s", args.mask_meta)
    logging.info("\t\tOutput directory      : %s", args.output_dir)
    logging.info("\t\tParallel backend      : %s", args.backend)
    logging.info("\t\tOverwrite existing    : %s", OVERWRITE_EXISTING)
    logging.info("\t\tProcess all groups    : %s", args.process_all_groups)
    logging.info("==================================================")

    # ---------------- GRIB discovery & grouping ----------------
    if not args.grib_dir.exists():
        raise FileNotFoundError(f"GRIB directory does not exist: {args.grib_dir}")

    grib_files = sorted(args.grib_dir.glob("*.grib"))
    if not grib_files:
        raise RuntimeError(f"No .grib files found in {args.grib_dir}")
    logging.info("Found [%d] GRIB files in [%s].", len(grib_files), args.grib_dir)

    logging.info("Grouping GRIB files by (dataset, coords, uid)...")
    all_groups = group_grib_files_by_triplet(grib_files)

    logging.info("Detected %d GRIB groups in %s:", len(all_groups), args.grib_dir)
    for (dataset, coords, uid), filelist in all_groups.items():
        logging.info("Grouping characteristics:")
        logging.info("\t\tDataset: [%s]", dataset)
        logging.info("\t\tCoordinates: [%s]", coords)
        logging.info("\t\tUID: [%s]", uid)
        logging.info("\t\tNumber of files: %d", len(filelist))

    groups = select_groups(
        all_groups,
        dataset=args.dataset,
        coords=args.coords,
        uid=args.uid,
        process_all=args.process_all_groups,
    )

    logging.info("After filtering, [%d] group(s) will now be processed:", len(groups))
    for (dataset, coords, uid), filelist in groups.items():
        logging.info(
            "\t\t[%s_%s_%s] with [%d] files)",
            dataset,
            coords,
            uid,
            len(filelist),
        )

    # ---------------- Load mask metadata ----------------
    if not args.mask_meta.exists():
        raise FileNotFoundError(f"Mask metadata not found: {args.mask_meta}")
    if not args.mask_parquet.exists():
        raise FileNotFoundError(f"Mask parquet not found: {args.mask_parquet}")

    with open(args.mask_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    region_token = str(meta.get("region_token", "REGION")).upper()
    logging.info("\t\tUsing region token: %s", region_token)

    # ---------------- Reference variables ----------------
    files_to_process: List[Path] = []
    for _, flist in groups.items():
        files_to_process.extend(sorted(flist))
    first_file = files_to_process[0]

    params = scan_parameters(first_file)
    shortnames = sorted({p["shortName"] for p in params})
    logging.info("\t\tWith variables : %s", shortnames)

    # ---------------- Consistency check ----------------
    verify_variable_consistency_parallel(
        files_to_process,
        reference_shortnames=set(shortnames),
        backend=args.backend if args.backend in {"none", "processpool"} else "none",
    )

    logging.info("Processing %d GRIB files.", len(files_to_process))

    # ---------------- Dispatch to backend ----------------
    if args.backend == "none":
        durations = run_sequential(
            files_to_process, args.mask_parquet, shortnames,
            region_token, args.output_dir
        )
    elif args.backend == "processpool":
        durations = run_processpool(
            files_to_process, args.mask_parquet, shortnames,
            region_token, args.output_dir
        )
    else:
        raise ValueError(f"Unexpected backend in single-process path: {args.backend}")

    # ---------------- Summary ----------------
    total_files = sum(1 for d in durations if d > 0)
    cpu_time_total = sum(durations)
    cpu_time_avg = cpu_time_total / total_files if total_files else 0.0
    wall_time_total = time.time() - overall_start

    cpu_total_fmt = format_hms(cpu_time_total)
    cpu_avg_fmt = format_hms(cpu_time_avg)
    wall_total_fmt = format_hms(wall_time_total)

    group_keys = list(groups.keys())
    group_str = ", ".join([f"{g[0]}_{g[1]}_{g[2]}" for g in group_keys])

    worker_count = (
        MAX_WORKERS if args.backend == "processpool" else 1
    )

    logging.info("==================================================")
    logging.info("PROCESS OVERVIEW")
    logging.info("==================================================")
    logging.info("INPUTS")
    logging.info("--------------------------------------------------")
    logging.info("\t\tSource directory : %s", args.grib_dir)
    logging.info("\t\tOutput directory : %s", args.output_dir)
    logging.info("\t\tMask parquet     : %s", args.mask_parquet)
    logging.info("\t\tMask metadata    : %s", args.mask_meta)
    logging.info("PROCESSING")
    logging.info("--------------------------------------------------")
    logging.info("\t\tGrouping key(s)  : %s", group_str)
    logging.info("\t\tFiles processed  : [%d]", total_files)
    logging.info("\t\tParallel backend : [%s]", args.backend)
    logging.info("\t\tParallel workers : [%d]", worker_count)
    logging.info("TIMINGS")
    logging.info("--------------------------------------------------")
    logging.info("\t\tAverage per file : %s", cpu_avg_fmt)
    logging.info("\t\tCPU total time   : %s", cpu_total_fmt)
    logging.info("\t\tTOTAL WALL TIME  : %s", wall_total_fmt)
    logging.info("--------------------------------------------------")
    logging.info("Processing complete.")


def _mpi_main(args) -> None:
    """
    Main pipeline when backend is 'mpi'.

    Rank 0 does all global pre-work (file listing, grouping, mask metadata,
    variable list, etc.) and then broadcasts to other ranks. Consistency
    checks and monthly processing are then split across ranks.
    """
    overall_start = time.time()
    setup_logging(args.log)

    if not MPI_ENABLED:
        raise RuntimeError("Backend 'mpi' requested but MPI is not available.")

    if RANK == 0:
        logging.info("==================================================")
        logging.info("Running step2a_mask_and_process_grib.py (MPI path)")
        logging.info("==================================================")
        logging.info("Configuration used:")
        logging.info("\t\tGRIB source directory : %s", args.grib_dir)
        logging.info("\t\tMask parquet          : %s", args.mask_parquet)
        logging.info("\t\tMask metadata         : %s", args.mask_meta)
        logging.info("\t\tOutput directory      : %s", args.output_dir)
        logging.info("\t\tParallel backend      : %s", args.backend)
        logging.info("\t\tOverwrite existing    : %s", OVERWRITE_EXISTING)
        logging.info("\t\tProcess all groups    : %s", args.process_all_groups)
        logging.info("\t\tMPI world size        : %d", SIZE)
        logging.info("==================================================")

    # ---------------- Rank 0 pre-work ----------------
    if RANK == 0:
        if not args.grib_dir.exists():
            raise FileNotFoundError(f"GRIB directory does not exist: {args.grib_dir}")

        grib_files = sorted(args.grib_dir.glob("*.grib"))
        if not grib_files:
            raise RuntimeError(f"No .grib files found in {args.grib_dir}")
        logging.info("Found [%d] GRIB files in [%s].", len(grib_files), args.grib_dir)

        logging.info("Grouping GRIB files by (dataset, coords, uid)...")
        all_groups = group_grib_files_by_triplet(grib_files)

        logging.info("Detected %d GRIB groups in %s:", len(all_groups), args.grib_dir)
        for (dataset, coords, uid), filelist in all_groups.items():
            logging.info("Grouping characteristics:")
            logging.info("\t\tDataset: [%s]", dataset)
            logging.info("\t\tCoordinates: [%s]", coords)
            logging.info("\t\tUID: [%s]", uid)
            logging.info("\t\tNumber of files: %d", len(filelist))

        groups = select_groups(
            all_groups,
            dataset=args.dataset,
            coords=args.coords,
            uid=args.uid,
            process_all=args.process_all_groups,
        )

        logging.info(
            "After filtering, [%d] group(s) will now be processed:", len(groups)
        )
        for (dataset, coords, uid), filelist in groups.items():
            logging.info(
                "\t\t[%s_%s_%s] with [%d] files)",
                dataset,
                coords,
                uid,
                len(filelist),
            )

        # Load mask metadata
        if not args.mask_meta.exists():
            raise FileNotFoundError(f"Mask metadata not found: {args.mask_meta}")
        if not args.mask_parquet.exists():
            raise FileNotFoundError(f"Mask parquet not found: {args.mask_parquet}")

        with open(args.mask_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        region_token = str(meta.get("region_token", "REGION")).upper()
        logging.info("\t\tUsing region token: %s", region_token)

        # Flatten files to process
        files_to_process: List[Path] = []
        for _, flist in groups.items():
            files_to_process.extend(sorted(flist))

        # Reference variables from first file
        first_file = files_to_process[0]
        params = scan_parameters(first_file)
        shortnames = sorted({p["shortName"] for p in params})
        logging.info("\t\tWith variables : %s", shortnames)

        group_keys = list(groups.keys())
        group_str = ", ".join([f"{g[0]}_{g[1]}_{g[2]}" for g in group_keys])
    else:
        files_to_process = None  # type: ignore[assignment]
        shortnames = None  # type: ignore[assignment]
        region_token = None  # type: ignore[assignment]
        group_str = ""  # type: ignore[assignment]

    # ---------------- Broadcast global state ----------------
    files_to_process = COMM.bcast(files_to_process, root=0)  # type: ignore[assignment, arg-type]
    shortnames = COMM.bcast(shortnames, root=0)  # type: ignore[assignment, arg-type]
    region_token = COMM.bcast(region_token, root=0)  # type: ignore[assignment, arg-type]
    group_str = COMM.bcast(group_str, root=0)  # type: ignore[assignment, arg-type]

    # ---------------- Consistency check (MPI) ----------------
    verify_variable_consistency_parallel(
        files_to_process,
        reference_shortnames=set(shortnames),
        backend="mpi",
    )

    if RANK == 0:
        logging.info("Processing %d GRIB files with MPI.", len(files_to_process))

    # ---------------- MPI processing ----------------
    durations = run_mpi(
        files_to_process,
        args.mask_parquet,
        shortnames,
        region_token,
        args.output_dir,
    )

    # ---------------- Summary (rank 0 only) ----------------
    if RANK != 0:
        return

    total_files = sum(1 for d in durations if d > 0)
    cpu_time_total = sum(durations)
    cpu_time_avg = cpu_time_total / total_files if total_files else 0.0
    wall_time_total = time.time() - overall_start

    cpu_total_fmt = format_hms(cpu_time_total)
    cpu_avg_fmt = format_hms(cpu_time_avg)
    wall_total_fmt = format_hms(wall_time_total)

    logging.info("==================================================")
    logging.info("PROCESS OVERVIEW")
    logging.info("==================================================")
    logging.info("INPUTS")
    logging.info("--------------------------------------------------")
    logging.info("\t\tSource directory : %s", args.grib_dir)
    logging.info("\t\tOutput directory : %s", args.output_dir)
    logging.info("\t\tMask parquet     : %s", args.mask_parquet)
    logging.info("\t\tMask metadata    : %s", args.mask_meta)
    logging.info("PROCESSING")
    logging.info("--------------------------------------------------")
    logging.info("\t\tGrouping key(s)  : %s", group_str)
    logging.info("\t\tFiles processed  : [%d]", total_files)
    logging.info("\t\tParallel backend : [mpi]")
    logging.info("\t\tParallel workers : [%d]", SIZE)
    logging.info("TIMINGS")
    logging.info("--------------------------------------------------")
    logging.info("\t\tAverage per file : %s", cpu_avg_fmt)
    logging.info("\t\tCPU total time   : %s", cpu_total_fmt)
    logging.info("\t\tTOTAL WALL TIME  : %s", wall_total_fmt)
    logging.info("--------------------------------------------------")
    logging.info("Processing complete.")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ERA5 GRIB files into country-only Parquet files."
    )

    parser.add_argument(
        "--grib-dir", type=Path, default=GRIB_DIR,
        help="Directory containing monthly ERA5 .grib files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory to write output monthly Parquet files.",
    )
    parser.add_argument(
        "--mask-parquet", type=Path, default=MASK_PARQUET,
        help="Path to mask parquet file.",
    )
    parser.add_argument(
        "--mask-meta", type=Path, default=MASK_META,
        help="Path to mask metadata JSON file.",
    )

    # Optional filters
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--coords", type=str, default=None)
    parser.add_argument("--uid", type=str, default=None)

    # Group selection behavior
    parser.add_argument(
        "--process-all-groups",
        action="store_true",
        default=PROCESS_ALL_GROUPS_DEFAULT,
        help=(
            "If multiple GRIB groups exist and no filters provided, process ALL. "
            "Default: only process the most recent group."
        ),
    )

    # Backend & logging
    parser.add_argument(
        "--backend",
        type=str,
        choices=["none", "processpool", "mpi"],
        default=PARALLEL_BACKEND_DEFAULT,
        help="Parallel backend to use.",
    )
    parser.add_argument(
        "--log",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Logging level.",
    )

    cli_args = parser.parse_args()

    if cli_args.backend == "mpi":
        _mpi_main(cli_args)
    else:
        _single_process_main(cli_args)
