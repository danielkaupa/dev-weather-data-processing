"""
02_process_grib_to_parquet.py

Convert ERA5 global GRIB files to India-only Parquet, with exact
fractional overlap per grid cell and area-weighting of extensive variables.

Features:
- Variable scanning on first file; verify all files share same variable set
- Builds or loads a grid-cell fractional mask (frac_in_india, cell_area_m2)
- Robust GRIB loader:
    * Opens each variable separately via cfgrib (filter_by_keys shortName)
    * Flattens forecast "time, step, valid_time" into a single time axis
    * Merges all variables into one xarray.Dataset
    * Preserves original variable names (no renaming)
- Converts Dataset → Polars, joins India mask, filters to frac_in_india > 0
- Area-weights extensive variables only
- Writes monthly Parquet (snappy) via Polars LazyFrame.sink_parquet()
- Parallel backends: "none", "processpool", "mpi"
"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Set
import time

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import re
import xarray as xr
import polars as pl
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Geod

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

# Output directory for monthly India-only Parquet files
OUTPUT_MONTHLY_DIR = Path("../data/interim")

# India boundary GeoJSON from Script 1 (already extracted)
INDIA_BOUNDARY_GEOJSON = Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")

# Cache path for grid mask with fractional overlap
MASK_CACHE_PATH = Path("india_grid_mask.parquet")

# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND = "processpool"  # "none" | "processpool" | "mpi"

# workers for processpool backend (2/3 of maximum available CPUs, min 2)
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Logging
LOG_LEVEL = logging.INFO

OVERWRITE_EXISTING = True   # set True to force regeneration


# ---------------------------------------------------------------------
# VARIABLE CATEGORIES
# ---------------------------------------------------------------------

# Extensive variables (area-weighted by fractional overlap)
EXTENSIVE_VARS = {
    "tp",
    "ssr",
    "ssrd",
    "ssrc",
    "ssrdc",
    "str",
    "strc",
    "strd",
    "strdc",
    "tsr",
    "tsrc",
    "ttr",
    "ttrc",
    "fdir",
    "cdir",
    "uvb",
}

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------


def setup_logging() -> None:
    OUTPUT_MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )



# ---------------------------------------------------------------------
# OPTIONAL: REMOVE ISLAND GRID POINTS BY LAT/LON BOUNDS
# ---------------------------------------------------------------------

def drop_island_points(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove ERA5 grid points corresponding to island regions
    using simple lat/lon bounding boxes.

    Current boxes (tweak as needed):
      - Andaman & Nicobar:   lon 90–95E,  lat < 15N
      - Lakshadweep:         lon 70–74E,  lat 8–14N
    """
    lon = pl.col("longitude")
    lat = pl.col("latitude")

    andaman_nicobar = (lon >= 90.0) & (lon <= 95.0) & (lat < 15.0)
    lakshadweep     = (lon >= 71.0) & (lon <= 74.0) & (lat >= 8.2) & (lat <= 11.7)

    return lf.filter(~(andaman_nicobar | lakshadweep))


def build_output_filename(path: Path) -> str:
    """
    Convert input GRIB filename:
        dataset_region_uniqueid_year_month.grib
    into:
        dataset_INDIA_uniqueid_year_month.parquet
    """
    m = re.match(
        r"^(?P<dataset>[^_]+)_"         # dataset name
        r"(?P<region>[^_]+)_"          # geographic region (ignored)
        r"(?P<uid>[^_]+)_"             # unique id
        r"(?P<year>\d{4})_"            # year
        r"(?P<month>\d{2})\.grib$",    # month
        path.name
    )

    if not m:
        raise ValueError(f"Filename pattern not recognized: {path.name}")

    return (
        f"{m.group('dataset')}_"
        f"INDIA_"
        f"{m.group('uid')}_"
        f"{m.group('year')}_"
        f"{m.group('month')}.parquet"
    )


# ---------------------------------------------------------------------
# VARIABLE SCANNING & CONSISTENCY
# ---------------------------------------------------------------------


def scan_parameters_in_file(path: Path) -> List[Dict]:
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
    logging.info("Verifying variable consistency across %d GRIB files...", len(grib_files))
    mismatches = []

    for path in grib_files:
        params_this = scan_parameters_in_file(path)
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
                logging.warning("  Missing (in reference but not in this file): %s", sorted(missing))
            if extra:
                logging.warning("  Extra (in this file but not in reference): %s", sorted(extra))


# ---------------------------------------------------------------------
# GEOMETRY & FRACTIONAL MASK
# ---------------------------------------------------------------------


def load_india_geometry() -> MultiPolygon:
    gdf = gpd.read_file(INDIA_BOUNDARY_GEOJSON)
    gdf = gdf.to_crs(epsg=4326)
    geom = unary_union(gdf.geometry)
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    if not isinstance(geom, MultiPolygon):
        raise TypeError("India geometry is not a Polygon or MultiPolygon.")
    return geom



def geodesic_area_polygon(geod: Geod, polygon: Polygon) -> float:
    """Return geodesic area (m^2) of a shapely Polygon using pyproj.Geod."""
    lons, lats = polygon.exterior.coords.xy
    area, _ = geod.polygon_area_perimeter(lons, lats)
    return abs(area)


def geodesic_area_geometry(geod: Geod, geom) -> float:
    """Return geodesic area for Polygon/MultiPolygon/GeometryCollection."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
    from shapely.geometry import GeometryCollection

    if isinstance(geom, ShapelyPolygon):
        return geodesic_area_polygon(geod, geom)
    if isinstance(geom, ShapelyMultiPolygon):
        return sum(geodesic_area_polygon(geod, g) for g in geom.geoms)
    if isinstance(geom, GeometryCollection):
        return sum(
            geodesic_area_geometry(geod, g)
            for g in geom.geoms
            if isinstance(g, (ShapelyPolygon, ShapelyMultiPolygon))
        )
    return 0.0


def build_mask_from_sample_grib(sample_grib: Path, sample_shortname: str) -> pl.DataFrame:
    """
    Build a grid mask with fractional overlap for a sample GRIB file.

    Loads only a single variable (sample_shortname) to avoid GRIB edition issues.

    Returns a Polars DataFrame with:
      - latitude
      - longitude
      - frac_in_india (0..1)
      - cell_area_m2
    """
    logging.info(
        "Building India grid mask from sample GRIB: %s (var=%s)",
        sample_grib,
        sample_shortname,
    )

    ds = xr.open_dataset(
        sample_grib,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": sample_shortname},
        },
    )

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        ds.close()
        raise KeyError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat = ds["latitude"]
    lon = ds["longitude"]

    # We assume 1D lat/lon, as is typical for ERA5 global grids
    if lat.ndim != 1 or lon.ndim != 1:
        ds.close()
        raise ValueError(
            "This script currently expects 1D latitude/longitude arrays."
        )

    lat_vals = lat.values
    lon_vals = lon.values

    if len(lat_vals) < 2 or len(lon_vals) < 2:
        ds.close()
        raise ValueError("Latitude/longitude arrays too short to build cell polygons.")

    dlat = abs(lat_vals[1] - lat_vals[0])
    dlon = abs(lon_vals[1] - lon_vals[0])

    geod = Geod(ellps="WGS84")
    india_geom = load_india_geometry()

    rows = []
    total_cells = len(lat_vals) * len(lon_vals)
    logging.info(
        "Building mask for %d grid cells (lat=%d, lon=%d)",
        total_cells,
        len(lat_vals),
        len(lon_vals),
    )

    for lat_c in lat_vals:
        lat_min = float(lat_c - dlat / 2.0)
        lat_max = float(lat_c + dlat / 2.0)

        for lon_c in lon_vals:
            lon_min = float(lon_c - dlon / 2.0)
            lon_max = float(lon_c + dlon / 2.0)

            cell_poly = Polygon(
                [
                    (lon_min, lat_min),
                    (lon_min, lat_max),
                    (lon_max, lat_max),
                    (lon_max, lat_min),
                    (lon_min, lat_min),
                ]
            )

            cell_area = geodesic_area_geometry(geod, cell_poly)
            if cell_area == 0.0:
                frac = 0.0
            else:
                inter = india_geom.intersection(cell_poly)
                if inter.is_empty:
                    frac = 0.0
                else:
                    inter_area = geodesic_area_geometry(geod, inter)
                    frac = max(0.0, min(1.0, inter_area / cell_area))

            rows.append(
                {
                    "latitude": float(lat_c),
                    "longitude": float(lon_c),
                    "frac_in_india": float(frac),
                    "cell_area_m2": float(cell_area),
                }
            )

    ds.close()

    mask_df = pl.DataFrame(rows)
    non_zero = (mask_df["frac_in_india"] > 0).sum()
    logging.info("Mask built. Non-zero cells: %d", non_zero)

    return mask_df


def ensure_mask(sample_grib: Path, sample_shortname: str) -> pl.DataFrame:
    """
    Load India grid mask if it exists; otherwise build from sample GRIB
    and cache to Parquet.
    """
    if MASK_CACHE_PATH.exists():
        logging.info("Loading existing India mask from %s", MASK_CACHE_PATH)
        return pl.read_parquet(MASK_CACHE_PATH)

    MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    mask_df = build_mask_from_sample_grib(sample_grib, sample_shortname)
    logging.info("Saving India mask to %s", MASK_CACHE_PATH)
    mask_df.write_parquet(MASK_CACHE_PATH, compression="snappy")
    return mask_df


# ---------------------------------------------------------------------
# GRIB → XARRAY (PER-VARIABLE) → POLARS → INDIA CLIP
# ---------------------------------------------------------------------


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
    ds_merged = xr.merge(datasets, combine_attrs="override", join="inner", compat="override")    # <--- explicitly set

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
) -> float:
    """
    Process a single GRIB file:
    Returns the elapsed time in seconds.
    """
    start = time.time()

    try:
        year, month = parse_year_month_from_name(path)
    except Exception:
        logging.warning("Unable to parse year/month from %s. Skipping.", path)
        return 0.0

    out_name = build_output_filename(path)
    out_path = OUTPUT_MONTHLY_DIR / out_name

    if out_path.exists() and not OVERWRITE_EXISTING:
        logging.info("Output already exists for %s, skipping.", path.name)
        return
    elif out_path.exists() and OVERWRITE_EXISTING:
        logging.info("Overwriting existing file for %s", path.name)


    logging.info("Processing %s -> %s", path.name, out_path.name)

    # 1) Load GRIB → xarray
    ds = load_grib_to_xarray(path, var_shortnames)

    # 2) Flatten to pandas, then to Polars
    df = ds.to_dataframe().reset_index()
    ds.close()

    pl_df = pl.from_pandas(df)

    # 3) Lazy join with India mask
    lf = pl_df.lazy()
    mask_lf = pl.scan_parquet(mask_path)

    coord_cols = ["latitude", "longitude"]
    for c in coord_cols:
        if c not in pl_df.columns:
            raise KeyError(
                f"Expected coordinate column '{c}' in {path.name}, got {pl_df.columns}"
            )

    lf = (
        lf.join(mask_lf, on=coord_cols, how="inner")
          .filter(pl.col("frac_in_india") > 0.0)
    )

    # 3b) Drop island grid points by explicit lat/lon boxes
    lf = drop_island_points(lf)

    # 4) Area-weight extensive variables
    for var in EXTENSIVE_VARS:
        if var in pl_df.columns:
            lf = lf.with_columns(
                (pl.col(var) * pl.col("frac_in_india")).alias(var)
            )

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
    path, mask_path, var_shortnames = args
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] [worker] %(message)s",
    )
    return process_one_grib_file(path, mask_path, var_shortnames)


def run_sequential(grib_files, mask_path, var_shortnames) -> List[float]:
    durations = []
    for p in grib_files:
        durations.append(process_one_grib_file(p, mask_path, var_shortnames))
    return durations


def run_processpool(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
) -> List[float]:
    args = [(p, mask_path, var_shortnames) for p in grib_files]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(_worker_process_one_grib, args))


def run_mpi(grib_files, mask_path, var_shortnames) -> List[float]:
    if MPI is None:
        raise RuntimeError("mpi4py not available.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_durations = []

    for idx, p in enumerate(grib_files):
        if idx % size == rank:
            elapsed = process_one_grib_file(p, mask_path, var_shortnames)
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


def main() -> None:
    setup_logging()

    overall_start = time.time()   # <-- NEW LINE

    if not RAW_GRIB_DIR.exists():
        raise FileNotFoundError(f"RAW_GRIB_DIR does not exist: {RAW_GRIB_DIR}")

    grib_files = sorted(RAW_GRIB_DIR.glob("*.grib"))
    if not grib_files:
        logging.error("No GRIB files found in %s", RAW_GRIB_DIR)
        return

    logging.info("Found %d GRIB files to process.", len(grib_files))

    # 1) Variable scan on first file
    first_file = grib_files[0]
    params_first = scan_parameters_in_file(first_file)
    shortnames_ref = {p["shortName"] for p in params_first}
    var_shortnames_sorted = sorted(shortnames_ref)

    logging.info(
        "Reference file %s has %d variables: %s",
        first_file.name,
        len(shortnames_ref),
        var_shortnames_sorted,
    )

    # 2) Consistency check across all files
    verify_variables_across_files(grib_files, shortnames_ref)

    # 3) Build or load fractional India mask
    # Prefer "2t"/"t2m" if present; otherwise first shortname in set
    if "2t" in shortnames_ref:
        sample_var = "2t"
    elif "t2m" in shortnames_ref:
        sample_var = "t2m"
    else:
        sample_var = var_shortnames_sorted[0]

    mask_df = ensure_mask(first_file, sample_var)
    if not MASK_CACHE_PATH.exists():
        # ensure it is saved
        MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        mask_df.write_parquet(MASK_CACHE_PATH, compression="snappy")

    logging.info("Starting processing with backend: %s", PARALLEL_BACKEND)

    if PARALLEL_BACKEND == "none":
        durations = run_sequential(grib_files, MASK_CACHE_PATH, var_shortnames_sorted)
    elif PARALLEL_BACKEND == "processpool":
        durations = run_processpool(grib_files, MASK_CACHE_PATH, var_shortnames_sorted)
    elif PARALLEL_BACKEND == "mpi":
        durations = run_mpi(grib_files, MASK_CACHE_PATH, var_shortnames_sorted)
    else:
        raise ValueError(...)

    # --- Final Summary ---
    total_files = len([d for d in durations if d > 0])
    cpu_time_sum = sum(durations)
    cpu_avg = cpu_time_sum / total_files if total_files else 0

    overall_elapsed = time.time() - overall_start   # <-- NEW LINE

    logging.info("===============================================")
    logging.info("SUMMARY:")
    logging.info("  Files processed: %d", total_files)
    logging.info("  CPU time total: %.2f sec (%.2f min)", cpu_time_sum, cpu_time_sum / 60)
    logging.info("  Avg CPU time per file: %.2f sec", cpu_avg)
    logging.info("-----------------------------------------------")
    logging.info("  TOTAL WALL-CLOCK TIME: %.2f sec (%.2f min)",
                 overall_elapsed, overall_elapsed / 60)
    logging.info("===============================================")


if __name__ == "__main__":
    main()
