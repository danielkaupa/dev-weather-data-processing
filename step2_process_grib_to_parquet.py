"""
02_process_grib_to_parquet.py

Process monthly ERA5-style GRIB files → country-only Parquet using a
pre-computed mask + metadata from Script 01.

Assumptions
-----------
- Script 01 has already created:
    masks/{dataset_prefix}_{COUNTRY_TOKEN}_mask_...parquet
    masks/{dataset_prefix}_{COUNTRY_TOKEN}_mask_...json

- The JSON metadata (from Script 01) contains at least:
    - dataset_prefix
    - country_token
    - boundary_path
    - adm_level
    - country_name
    - inclusion_mode
    - fraction_threshold
    - apply_exclusions
    - exclusion_bboxes
    - row_count
    - generated_at

- The Parquet mask file (from Script 01) has at least:
    - 'latitude'
    - 'longitude'
  and may also include things like:
    - 'frac_in_region'
    - 'centroid_in_region'
    - 'cell_area_m2'
  which we simply carry through.

- This script:
    * DOES NOT build any masks.
    * DOES NOT area-weight any variables (we assume Script 01
      already trimmed to "good" cells with your preferred rules).
    * DOES NOT do any plotting.

File naming
-----------
Input GRIB:
    dataset_region_uid_year_month.grib
Example:
    era5-world_N37W68S6E98_d514a3a3c256_2018_01.grib

Output Parquet:
    dataset_COUNTRYTOKEN_uid_year_month.parquet
Example (for INDIA):
    era5-world_INDIA_d514a3a3c256_2018_01.parquet

CLI
---
    python 02_process_grib_to_parquet.py \
        --mask-meta masks/era5-world_INDIA_mask_combined0.9_264612.json \
        --backend processpool

"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Set
import time
import re
import argparse
import multiprocessing
import json

import pandas as pd
import xarray as xr
import polars as pl

# eccodes for variable scanning
from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

# MPI is optional; imported only if backend == "mpi"
try:
    from mpi4py import MPI  # type: ignore
except ImportError:
    MPI = None  # type: ignore


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Input directory containing monthly GRIB files
RAW_GRIB_DIR = Path("../data/raw")

# Output directory for monthly country-only Parquet files
OUTPUT_MONTHLY_DIR = Path("../data/interim")

# Directory where Script 01 saved masks + metadata
MASKS_DIR = Path("masks")

# Default: auto-detect newest mask metadata JSON in MASKS_DIR
DEFAULT_MASK_META_PATH: Path | None = None

# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND_DEFAULT = "processpool"  # "none" | "processpool" | "mpi"

# workers for processpool backend (2/3 of maximum available CPUs, min 2)
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Logging
LOG_LEVEL = logging.INFO

# Overwrite existing monthly Parquet outputs?
OVERWRITE_EXISTING = True


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------


def setup_logging() -> None:
    OUTPUT_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# MASK METADATA & PATH RESOLUTION
# ---------------------------------------------------------------------


def auto_detect_mask_meta() -> Path:
    """
    Auto-detect the newest mask metadata JSON under MASKS_DIR.
    """
    candidates = sorted(
        MASKS_DIR.glob("*_mask_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No mask metadata JSON found under {MASKS_DIR} "
            f"(expected something like '*_mask_*.json')."
        )
    chosen = candidates[0]
    logging.info("Auto-detected mask metadata: %s", chosen)
    return chosen


def load_mask_metadata(mask_meta_path: Path | None) -> tuple[Path, Dict]:
    """
    Load Script 01 metadata JSON, infer mask parquet path, and return:

        (mask_parquet_path, meta_dict)

    If mask_meta_path is None, auto-detect the newest metadata JSON
    under MASKS_DIR.
    """
    if mask_meta_path is None:
        meta_path = auto_detect_mask_meta()
    else:
        meta_path = mask_meta_path

    if not meta_path.exists():
        raise FileNotFoundError(f"Mask metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    mask_parquet = meta_path.with_suffix(".parquet")
    if not mask_parquet.exists():
        raise FileNotFoundError(
            f"Expected mask parquet at {mask_parquet} "
            f"(same stem as metadata JSON)."
        )

    logging.info("Using mask parquet: %s", mask_parquet)
    logging.info("Mask metadata: country_token=%s, dataset_prefix=%s, rows=%s",
                 meta.get("country_token"), meta.get("dataset_prefix"), meta.get("row_count"))

    return mask_parquet, meta


# ---------------------------------------------------------------------
# FILENAME PARSERS
# ---------------------------------------------------------------------


def build_output_filename(path: Path, country_token: str) -> str:
    """
    Convert input GRIB filename:
        dataset_region_uid_year_month.grib
    into:
        dataset_COUNTRYTOKEN_uid_year_month.parquet
    """
    m = re.match(
        r"^(?P<dataset>[^_]+)_"
        r"(?P<region>[^_]+)_"
        r"(?P<uid>[^_]+)_"
        r"(?P<year>\d{4})_"
        r"(?P<month>\d{2})\.grib$",
        path.name,
    )

    if not m:
        raise ValueError(f"Filename pattern not recognized: {path.name}")

    return (
        f"{m.group('dataset')}_"
        f"{country_token}_"
        f"{m.group('uid')}_"
        f"{m.group('year')}_"
        f"{m.group('month')}.parquet"
    )


def parse_year_month_from_name(path: Path) -> tuple[int, int]:
    """
    Parse year and month from a filename like:
      era5-world_N37W68S6E98_d514a3a3c256_2025_04.grib
    """
    stem = path.stem
    parts = stem.split("_")
    year = int(parts[-2])
    month = int(parts[-1])
    return year, month


# ---------------------------------------------------------------------
# VARIABLE SCANNING & CONSISTENCY
# ---------------------------------------------------------------------


def scan_parameters_for_processing(path: Path) -> List[Dict]:
    """
    Scan a GRIB file with ecCodes and return a list of dictionaries:
    [
        { "paramId": 167, "shortName": "2t" },
        { "paramId": 228, "shortName": "tp" },
        ...
    ]
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
                    short_name = codes_get(gid, "shortName")
                    param_id = codes_get(gid, "paramId")

                    if isinstance(short_name, bytes):
                        short_name = short_name.decode("utf-8")

                    params[int(param_id)] = {
                        "paramId": int(param_id),
                        "shortName": str(short_name),
                    }

                finally:
                    codes_release(gid)

    except FileNotFoundError:
        logging.error("File not found during scan: %s", path)
    except Exception as e:  # noqa: BLE001
        logging.exception("Unexpected error while scanning %s: %s", path, e)

    return list(params.values())


def verify_variables_across_files(
    grib_files: List[Path],
    reference_shortnames: Set[str],
) -> None:
    """
    Verify that all GRIB files share the same set of variable shortNames.
    Logs warnings for mismatches.
    """
    logging.info(
        "Verifying variable consistency across %d GRIB files...", len(grib_files)
    )
    mismatches = []

    for path in grib_files:
        params_this = scan_parameters_for_processing(path)
        vars_this = {p["shortName"] for p in params_this}
        if vars_this != reference_shortnames:
            diff_ref = reference_shortnames - vars_this
            diff_this = vars_this - reference_shortnames
            mismatches.append((path, diff_ref, diff_this))

    if not mismatches:
        logging.info("All GRIB files share the same variable set.")
    else:
        logging.warning("Detected files with differing variable sets:")
        for path, missing, extra in mismatches:
            logging.warning("File: %s", path.name)
            if missing:
                logging.warning(
                    "  Missing (in reference but not in this file): %s",
                    sorted(missing),
                )
            if extra:
                logging.warning(
                    "  Extra (in this file but not in reference): %s",
                    sorted(extra),
                )


# ---------------------------------------------------------------------
# GRIB → XARRAY (PER-VARIABLE) → POLARS → COUNTRY CLIP
# ---------------------------------------------------------------------


def open_single_variable_dataset(path: Path, shortname: str) -> xr.Dataset:
    """
    Open a single ERA5 variable from a GRIB file via cfgrib, handling
    forecast-style dimensions (time, step, valid_time) by flattening to
    a single time dimension using valid_time.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": shortname},
        },
    )

    # Forecast-style: time, step, valid_time
    if "step" in ds.dims and "valid_time" in ds.coords:
        # dims might be: time, step, latitude, longitude
        other_dims = [d for d in ds.dims if d not in ("time", "step")]
        stacked = ds.stack(time_step=("time", "step"))

        # Flatten valid_time: treat as the actual time coordinate
        flat_time = stacked["valid_time"].values.ravel()
        flat_time = pd.to_datetime(flat_time)

        data_vars = {}
        for var_name, da in stacked.data_vars.items():
            # move the stacked dim to front for consistent shape
            data = da.transpose("time_step", *other_dims).values
            data_vars[var_name] = (["time", *other_dims], data)

        coords = {"time": flat_time}
        for dim in other_dims:
            coords[dim] = ds[dim]

        new_ds = xr.Dataset(data_vars, coords=coords)
        ds.close()
        return new_ds

    # Analysis-style: just return as-is, but ensure time is datetime64
    if "time" in ds.coords:
        ds = ds.assign_coords(time=pd.to_datetime(ds["time"].values))

    return ds


def load_grib_to_xarray(path: Path, var_shortnames: List[str]) -> xr.Dataset:
    """
    Load a GRIB file by opening each variable separately and merging into
    a single Dataset. Joins on the common time axis (inner join).
    """
    logging.info("Loading GRIB file into xarray (per-variable): %s", path.name)

    datasets: List[xr.Dataset] = []

    for sn in var_shortnames:
        logging.debug("  Opening variable: %s", sn)
        ds_var = open_single_variable_dataset(path, sn)
        datasets.append(ds_var)

    # inner join on time/lat/lon etc.; override attrs to avoid conflicts
    ds_merged = xr.merge(
        datasets, combine_attrs="override", join="inner", compat="override"
    )

    # close the per-variable datasets (already copied data)
    for ds_var in datasets:
        ds_var.close()

    logging.info(
        "Loaded Dataset for %s: dims=%s, vars=%s",
        path.name,
        ds_merged.dims,
        list(ds_merged.data_vars),
    )

    return ds_merged


def process_one_grib_file(
    path: Path,
    mask_path: Path,
    var_shortnames: List[str],
    country_token: str,
) -> float:
    """
    Process a single GRIB file:
    Returns the elapsed time in seconds (wall).
    """
    start = time.time()

    try:
        year, month = parse_year_month_from_name(path)
    except Exception:
        logging.warning("Unable to parse year/month from %s. Skipping.", path)
        return 0.0

    out_name = build_output_filename(path, country_token)
    out_path = OUTPUT_MONTHLY_DIR / out_name

    if out_path.exists() and not OVERWRITE_EXISTING:
        logging.info("Output already exists for %s, skipping.", path.name)
        return 0.0
    elif out_path.exists() and OVERWRITE_EXISTING:
        logging.info("Overwriting existing file for %s", path.name)

    logging.info("Processing %s (%04d-%02d) -> %s",
                 path.name, year, month, out_path.name)

    # 1) Load GRIB → xarray
    ds = load_grib_to_xarray(path, var_shortnames)

    # 2) Flatten to pandas, then to Polars
    df = ds.to_dataframe().reset_index()
    ds.close()

    pl_df = pl.from_pandas(df)

    # 3) Lazy join with country mask (Script 01)
    lf = pl_df.lazy()
    mask_lf = pl.scan_parquet(mask_path)

    coord_cols = ["latitude", "longitude"]
    for c in coord_cols:
        if c not in pl_df.columns:
            raise KeyError(
                f"Expected coordinate column '{c}' in {path.name}, got {pl_df.columns}"
            )

    lf = lf.join(mask_lf, on=coord_cols, how="inner")

    # Validate expected columns are present
    required_cols = ["frac_in_country", "centroid_in_country"]
    missing = [c for c in required_cols if c not in lf.collect_schema().names()]
    if missing:
        raise KeyError(f"Mask missing expected columns: {missing}")



    # 5) Write Parquet lazily
    OUTPUT_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(out_path, compression="snappy")

    elapsed = time.time() - start
    logging.info(
        "Finished %s -> %s in %.2f sec",
        path.name,
        out_path.name,
        elapsed,
    )
    return elapsed


# ---------------------------------------------------------------------
# PARALLEL DISPATCH
# ---------------------------------------------------------------------


def _worker_process_one_grib(args) -> float:
    path, mask_path, var_shortnames, country_token = args
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] [worker] %(message)s",
    )
    return process_one_grib_file(path, mask_path, var_shortnames, country_token)


def run_sequential(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
    country_token: str,
) -> List[float]:
    durations = []
    for p in grib_files:
        durations.append(
            process_one_grib_file(p, mask_path, var_shortnames, country_token)
        )
    return durations


def run_processpool(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
    country_token: str,
) -> List[float]:
    from concurrent.futures import ProcessPoolExecutor   # ← add this import
    args = [(p, mask_path, var_shortnames, country_token) for p in grib_files]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(_worker_process_one_grib, args))



def run_mpi(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
    country_token: str,
) -> List[float]:
    if MPI is None:
        raise RuntimeError("mpi4py not available.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_durations: List[float] = []

    for idx, p in enumerate(grib_files):
        if idx % size == rank:
            elapsed = process_one_grib_file(
                p, mask_path, var_shortnames, country_token
            )
            local_durations.append(elapsed)

    # All processes wait here before gathering
    comm.Barrier()

    # Collect timings
    all_durations = comm.gather(local_durations, root=0)

    # Final global barrier (optional, but nice for logs)
    comm.Barrier()

    if rank == 0:
        return [t for sub in all_durations for t in sub]
    else:
        return []


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main(
    mask_meta_path: Path | None,
    parallel_backend: str,
) -> None:
    setup_logging()

    overall_start = time.time()

    if not RAW_GRIB_DIR.exists():
        raise FileNotFoundError(f"RAW_GRIB_DIR does not exist: {RAW_GRIB_DIR}")

    grib_files = sorted(RAW_GRIB_DIR.glob("*.grib"))
    if not grib_files:
        logging.error("No GRIB files found in %s", RAW_GRIB_DIR)
        return

    logging.info("Found %d GRIB files to process.", len(grib_files))

    # ----- MASK & METADATA -----
    mask_parquet, meta = load_mask_metadata(mask_meta_path)

    country_token = meta.get("country_token", "COUNTRY").upper()
    dataset_prefix_meta = meta.get("dataset_prefix")

    logging.info("Country token from metadata: %s", country_token)
    if dataset_prefix_meta:
        logging.info("Dataset prefix in metadata: %s", dataset_prefix_meta)

    # 1) Variable scan on first file for processing
    first_file = grib_files[0]
    params_first_proc = scan_parameters_for_processing(first_file)
    shortnames_ref_proc = {p["shortName"] for p in params_first_proc}
    var_shortnames_sorted_proc = sorted(shortnames_ref_proc)

    logging.info(
        "Reference file %s has %d variables: %s",
        first_file.name,
        len(shortnames_ref_proc),
        var_shortnames_sorted_proc,
    )

    # 2) Consistency check across all files
    verify_variables_across_files(grib_files, shortnames_ref_proc)

    # 3) Run processing with chosen backend
    logging.info("Starting processing with backend: %s", parallel_backend)

    if parallel_backend == "none":
        durations = run_sequential(
            grib_files,
            mask_parquet,
            var_shortnames_sorted_proc,
            country_token,
        )
    elif parallel_backend == "processpool":
        from concurrent.futures import ProcessPoolExecutor  # local import

        durations = run_processpool(
            grib_files,
            mask_parquet,
            var_shortnames_sorted_proc,
            country_token,
        )
    elif parallel_backend == "mpi":
        durations = run_mpi(
            grib_files,
            mask_parquet,
            var_shortnames_sorted_proc,
            country_token,
        )
    else:
        raise ValueError(
            f"Unknown parallel_backend={parallel_backend}. "
            f"Use 'none', 'processpool', or 'mpi'."
        )

    # --- Final Summary ---
    total_files = len([d for d in durations if d > 0])
    cpu_time_sum = sum(durations)
    cpu_avg = cpu_time_sum / total_files if total_files else 0.0

    overall_elapsed = time.time() - overall_start

    logging.info("===============================================")
    logging.info("SUMMARY:")
    logging.info("  Files processed: %d", total_files)
    logging.info(
        "  CPU time total (approx, per-file sum): %.2f sec (%.2f min)",
        cpu_time_sum,
        cpu_time_sum / 60,
    )
    logging.info("  Avg CPU time per file: %.2f sec", cpu_avg)
    logging.info("-----------------------------------------------")
    logging.info(
        "  TOTAL WALL-CLOCK TIME: %.2f sec (%.2f min)",
        overall_elapsed,
        overall_elapsed / 60,
    )
    logging.info("===============================================")


# ---------------------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Process ERA5-style GRIB files into country-only Parquet using "
            "a pre-computed mask + metadata from Script 01."
        )
    )
    parser.add_argument(
        "--mask-meta",
        type=Path,
        default=DEFAULT_MASK_META_PATH,
        help=(
            "Path to mask metadata JSON from Script 01. "
            "If omitted, the newest '*_mask_*.json' in 'masks/' is used."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["none", "processpool", "mpi"],
        default=PARALLEL_BACKEND_DEFAULT,
        help="Parallel backend to use (default: processpool).",
    )

    args = parser.parse_args()

    main(
        mask_meta_path=args.mask_meta,
        parallel_backend=args.backend,
    )
