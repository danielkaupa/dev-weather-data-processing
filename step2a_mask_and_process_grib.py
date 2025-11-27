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

If the mask region does **not** overlap the GRIB file domain, the inner
join produces zero rows. This is acceptable and expected.

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

# Optional MPI backend
try:
    from mpi4py import MPI
except Exception:
    MPI = None

# ======================================================================
# Global configuration options (hardcode - no argparse)
# ======================================================================

GRIB_DIR = Path("../data/raw/")
MASK_META = Path("masks/mask_metadata/era5-world_INDIA_mask_centroid_264612.json")
MASK_PARQUET = Path("masks/era5-world_INDIA_mask_centroid_264612.parquet")
OUTPUT_DIR = Path("../data/interim/")

# ======================================================================
# Global configuration defaults
# ======================================================================


# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND_DEFAULT = "processpool"

# Worker count for process pool
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Logging level
DEFAULT_LOG_LEVEL = "INFO"

# Overwrite behaviour
OVERWRITE_EXISTING = True

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
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

# ======================================================================
# Time formatting
# ======================================================================

def format_hms(seconds: float) -> str:
    """
    Convert seconds → H:M:S (zero-padded).

    Parameters
    ----------
    seconds : float

    Returns
    -------
    str
        "HH:MM:SS"
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

    Parameters
    ----------
    path : Path
        GRIB file path.

    Returns
    -------
    dict
        {"dataset","coords","uid","year","month"}

    Raises
    ------
    ValueError
        If filename does not match expected pattern.
    """
    m = FNAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match ERA5 format: {path.name}")
    return m.groupdict()


def group_grib_files_by_triplet(files: List[Path]) -> Dict[Tuple[str, str, str], List[Path]]:
    """
    Group GRIB files by the triplet (dataset, coords, uid).

    Parameters
    ----------
    files : list of Path

    Returns
    -------
    dict
        (dataset, coords, uid) → list[Path]
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

    Parameters
    ----------
    path : Path
        GRIB filename.
    region_token : str
        Uppercase country/region token from metadata.

    Returns
    -------
    str
        Filename of the output parquet.

    Notes
    -----
    Input:  dataset_region_uid_year_month.grib
    Output: dataset_region_uid_year_month.parquet
    """
    t = parse_grib_name(path)
    return f"{t['dataset']}_{region_token}_{t['uid']}_{t['year']}_{t['month']}.parquet"


def parse_year_month(path: Path) -> Tuple[int, int]:
    """
    Extract year and month integers from a GRIB filename.

    Parameters
    ----------
    path : Path

    Returns
    -------
    (year, month) : tuple[int,int]
    """
    t = parse_grib_name(path)
    return int(t["year"]), int(t["month"])


# ======================================================================
# Variable scanning + consistency
# ======================================================================

def scan_parameters(path: Path) -> List[Dict]:
    """
    Scan a GRIB file and return a list of variable descriptors.

    Parameters
    ----------
    path : Path

    Returns
    -------
    list of dict
        Each entry is {"paramId": int, "shortName": str}
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
                    params[int(pid)] = {"paramId": int(pid), "shortName": sn}
                finally:
                    codes_release(gid)

    except Exception:
        logging.exception("Error scanning parameters: %s", path)

    return list(params.values())


def verify_variable_consistency(files: List[Path], reference: Set[str]) -> None:
    """
    Log any GRIB files whose variable set differs from a reference set.

    Parameters
    ----------
    files : list of Path
    reference : set of str
        Set of shortNames expected.

    Notes
    -----
    This logs inconsistencies but does NOT stop execution.
    """
    logging.info(f"Checking variable consistency across {len(files)} files...")

    # --- Build reference set from first file ---
    # Each parameter dict must have "shortName"
    reference_file = files[0]
    reference = {p["shortName"] for p in scan_parameters(reference_file)}

    mismatches: Dict[str, Dict[str, List[str]]] = {}

    # --- Compare each file to reference ---
    for f in files:
        vars_this = {p["shortName"] for p in scan_parameters(f)}

        if vars_this != reference:
            missing = sorted(reference - vars_this)
            extra = sorted(vars_this - reference)

            mismatches[f.name] = {
                "missing": missing,
                "extra": extra,
            }

    # --- Summary ---
    if mismatches:
        logging.warning(
            "Variable consistency check complete — %d files differ from %s",
            len(mismatches),
            reference_file.name,
        )
        for fname, diff in mismatches.items():
            logging.warning("  %s:", fname)
            if diff["missing"]:
                logging.warning("    Missing: %s", ", ".join(diff["missing"]))
            if diff["extra"]:
                logging.warning("    Extra:   %s", ", ".join(diff["extra"]))
    else:
        logging.info(
            f"Variable consistency check complete — all {len(files)-1} other files match {reference_file.name}",
        )



# ======================================================================
# GRIB → xarray → polars
# ======================================================================

def open_single_variable_dataset(path: Path, shortname: str) -> xr.Dataset:
    """
    Open a single ERA5 variable via cfgrib, flattening forecast-style
    (time, step, valid_time) to a single time dimension.

    Parameters
    ----------
    path : Path
    shortname : str

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, latitude, longitude).
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

    Parameters
    ----------
    path : Path
    shortnames : list of str
        Variables to extract.

    Returns
    -------
    xr.Dataset
    """
    datasets = [open_single_variable_dataset(path, sn) for sn in shortnames]
    merged = xr.merge(datasets, combine_attrs="override", compat="override", join="outer")

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

    Parameters
    ----------
    path : Path
        GRIB file path.
    mask_parquet : Path
        Path to mask parquet file.
    var_shortnames : list of str
        Variables to extract.
    region_token : str
        Uppercase token used in output filename.
    output_dir : Path
        Output directory for monthly parquet files.

    Returns
    -------
    float
        Wall time in seconds for this file.
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

    Parameters
    ----------
    files : List[Path]
        List of Paths to GRIB files.
    mask_parquet : Path
        Path to mask parquet file.
    var_shortnames : List[str]
        Variables to extract.
    region_token : str
        Uppercase region token.
    output_dir : Path
        Output directory for monthly parquet files.

    Returns
    -------
    List[float]
        Processing times for each file.
    """
    return [
        process_one_grib(f, mask_parquet, var_shortnames, region_token, output_dir)
        for f in files
    ]


def _worker(args) -> float:
    """
    Worker function for process pool.

    Parameters
    ----------
    args : tuple
        (Path, Path, List[str], str, Path)

    Returns
    -------
    float
        Processing time for the file.
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

    Parameters
    ----------
    files : List[Path]
        List of Paths to GRIB files.
    mask_parquet : Path
        Path to mask parquet file.
    var_shortnames : List[str]
        Variables to extract.
    region_token : str
        Uppercase region token.
    output_dir : Path
        Output directory for monthly parquet files.

    Returns
    -------
    List[float]
        Processing times for each file.
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
    Run processing using MPI.

    Parameters
    ----------
    files : List[Path]
        List of Paths to GRIB files.
    mask_parquet : Path
        Path to mask parquet file.
    var_shortnames : List[str]
        Variables to extract.
    region_token : str
        Uppercase region token.
    output_dir : Path
        Output directory for monthly parquet files.

    Returns
    -------
    List[float]
        Processing times for each file (only on rank 0).
    """
    if MPI is None:
        raise RuntimeError("mpi4py not available.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_times = []
    for i, f in enumerate(files):
        if i % size == rank:
            t = process_one_grib(f, mask_parquet, var_shortnames, region_token, output_dir)
            local_times.append(t)

    comm.Barrier()
    all_times = comm.gather(local_times, root=0)
    comm.Barrier()

    if rank == 0:
        return [t for sub in all_times for t in sub]
    return []


# ======================================================================
# Main logic
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

    Parameters
    ----------
    groups : dict
        (dataset, coords, uid) → list of files
    dataset, coords, uid : str or None
        Optional filters
    coords: str or None
        Optional filter
    uid : str or None
        Optional filter
    process_all : bool
        If True and filters not given, process all groups.
        Else: pick the most recent group.

    Returns
    -------
    dict
        Filtered group(s)
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

        Parameters
        ----------
        item : tuple
            (key, list of files)
        Returns
        -------
        float
            Newest modification time.
        """
        _, files = item
        return max(f.stat().st_mtime for f in files)

    selected = max(groups.items(), key=newest_mtime)
    return {selected[0]: selected[1]}


def main(args):

    overall_start = time.time()
    setup_logging(args.log)

    logging.info("==================================================")
    logging.info("Running step2a_mask_and_process_grib.py")
    logging.info("This script will filter geospatial grid data according to a precomputed mask and convert the original grib files to parquet format.")
    logging.info("==================================================")
    logging.info("Configuration used:")
    logging.info(f"\t\tGRIB source directory : {args.grib_dir}")
    logging.info(f"\t\tMask parquet : {args.mask_parquet}")
    logging.info(f"\t\tMask metadata : {args.mask_meta}")
    logging.info(f"\t\tOutput directory : {args.output_dir}")
    logging.info(f"\t\tParallel backend : {args.backend}")
    logging.info(f"\t\tOverwrite existing : {OVERWRITE_EXISTING}")
    logging.info(f"\t\tProcess all groups : {args.process_all_groups}")
    logging.info("==================================================")
    logging.info("Starting processing...")


    # --------------------------------------------------------------
    # Load and validate GRIB directory
    # --------------------------------------------------------------
    if not args.grib_dir.exists():
        raise FileNotFoundError(f"GRIB directory does not exist: {args.grib_dir}")

    grib_files = sorted(args.grib_dir.glob("*.grib"))
    if not grib_files:
        raise RuntimeError(f"No .grib files found in {args.grib_dir}")
    logging.info(
        f"Found [{len(grib_files)}] GRIB files in [{args.grib_dir}]."
    )


    # --------------------------------------------------------------
    # Group GRIB files by (dataset, coords, uid)
    # --------------------------------------------------------------

    logging.info("Grouping GRIB files by (dataset, coords, uid)...")

    all_groups = group_grib_files_by_triplet(grib_files)

    logging.info(f"Detected {len(all_groups)} GRIB groups in {args.grib_dir}:")
    for (dataset, coords, uid), filelist in all_groups.items():
        logging.info(f"Grouping characteristics:")
        logging.info(f"\t\tDataset: [{dataset}]")
        logging.info(f"\t\tCoordinates: [{coords}]")
        logging.info(f"\t\tUID: [{uid}]")
        logging.info(f"\t\tNumber of files: {len(filelist)}")

    # Apply filters + recency selection
    groups = select_groups(
        all_groups,
        dataset=args.dataset,
        coords=args.coords,
        uid=args.uid,
        process_all=args.process_all_groups,
    )

    logging.info(f"After filtering, [{len(groups)}] group(s) will now be processed:")
    for (dataset, coords, uid), filelist in groups.items():
        logging.info(
            f'\t\t[{dataset}_{coords}_{uid}] with [{len(filelist)}] files)'
        )

    # --------------------------------------------------------------
    # Load explicit mask metadata + mask parquet
    # --------------------------------------------------------------
    if not args.mask_meta.exists():
        raise FileNotFoundError(f"Mask metadata not found: {args.mask_meta}")

    if not args.mask_parquet.exists():
        raise FileNotFoundError(f"Mask parquet not found: {args.mask_parquet}")

    with open(args.mask_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    region_token = str(meta.get("region_token", "REGION")).upper()
    logging.info("\t\tUsing region token: %s", region_token)

    # --------------------------------------------------------------
    # Reference variables from first file
    # --------------------------------------------------------------
    first_file = next(iter(groups.values()))[0]
    params = scan_parameters(first_file)
    shortnames = sorted({p["shortName"] for p in params})
    logging.info(f"\t\tWith variables : {shortnames}")

    # --------------------------------------------------------------
    # Verify consistency across all files
    # --------------------------------------------------------------
    verify_variable_consistency(grib_files, set(shortnames))

    # --------------------------------------------------------------
    # Flatten all selected groups into list of files
    # --------------------------------------------------------------
    files_to_process = []
    for _, flist in groups.items():
        files_to_process.extend(sorted(flist))

    logging.info("Processing %d GRIB files.", len(files_to_process))

    # --------------------------------------------------------------
    # Dispatch to backend
    # --------------------------------------------------------------
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
    elif args.backend == "mpi":
        durations = run_mpi(
            files_to_process, args.mask_parquet, shortnames,
            region_token, args.output_dir
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    total = sum(durations)
    n = sum(1 for d in durations if d > 0)
    avg = total / n if n > 0 else 0.0

    # --------------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------------
    total_files = sum(1 for d in durations if d > 0)
    cpu_time_total = sum(durations)
    cpu_time_avg = cpu_time_total / total_files if total_files else 0.0
    wall_time_total = time.time() - overall_start

    # Format times
    cpu_total_fmt = format_hms(cpu_time_total)
    cpu_avg_fmt = format_hms(cpu_time_avg)
    wall_total_fmt = format_hms(wall_time_total)

    # Extract grouping key(s)
    group_keys = list(groups.keys())
    group_str = ", ".join([f"{g[0]}_{g[1]}_{g[2]}" for g in group_keys])

    # Determine worker count for summary
    if args.backend == "processpool":
        worker_count = MAX_WORKERS
    elif args.backend == "mpi":
        worker_count = MPI.COMM_WORLD.Get_size() if MPI is not None else 1
    else:
        worker_count = 1

    logging.info("==================================================")
    logging.info("PROCESS OVERVIEW")
    logging.info("==================================================")
    logging.info("INPUTS")
    logging.info("--------------------------------------------------")
    logging.info(f"\t\tSource directory : {args.grib_dir}")
    logging.info(f"\t\tOutput directory : {args.output_dir}")
    logging.info(f"\t\tMask parquet : {args.mask_parquet}")
    logging.info(f"\t\tMask metadata : {args.mask_meta}")
    logging.info("PROCESSING")
    logging.info("--------------------------------------------------")
    logging.info(f"\t\tGrouping key(s) : {group_str}")
    logging.info(f"\t\tFiles processed : [{total_files}]")
    logging.info(f"\t\tParallel backend : [{args.backend}]")
    logging.info(f"\t\tParallel workers : [{worker_count}]")
    logging.info("TIMINGS")
    logging.info("--------------------------------------------------")
    logging.info(f"\t\tAverage per file : {cpu_avg_fmt}")
    logging.info(f"\t\tCPU total time : {cpu_total_fmt}")
    logging.info(f"\t\tTOTAL WALL TIME : {wall_total_fmt}")
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
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
        help="Directory containing monthly ERA5 .grib files."
    )

    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory to write output monthly Parquet files."
    )

    parser.add_argument(
        "--mask-parquet", type=Path, default=MASK_PARQUET,
        help="Path to mask parquet file."
    )

    parser.add_argument(
        "--mask-meta", type=Path, default=MASK_META,
        help="Path to mask metadata JSON file."
    )

    # ------------ Optional filters ------------
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--coords", type=str, default=None)
    parser.add_argument("--uid", type=str, default=None)

    # ------------ Group selection behavior ------------
    parser.add_argument(
        "--process-all-groups",
        action="store_true",
        default=PROCESS_ALL_GROUPS_DEFAULT,
        help="If multiple GRIB groups exist and no filters provided, process ALL. "
             "Default: only process the most recent group."
    )

    # ------------ Backend & logging ------------
    parser.add_argument(
        "--backend", type=str,
        choices=["none", "processpool", "mpi"],
        default=PARALLEL_BACKEND_DEFAULT
    )
    parser.add_argument(
        "--log", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL
    )

    args = parser.parse_args()
    main(args)
