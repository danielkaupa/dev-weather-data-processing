"""
02_process_grib_to_parquet.py

Monster script that:

1) Builds ERA5 India grid masks (fractional + 3 variants):
   - Base fractional: frac_in_india (0..1), cell_area_m2, centroid_in_india
   - Liberal mask:      keep all cells with any overlap        (frac_in_india > 0)
   - Strict mask:       keep only fully inside cells           (frac_in_india >= STRICT_THRESHOLD)
   - Conservative mask: keep fully inside + partial with
                        centroid inside India

2) Creates an 2x4 static matplotlib comparison plot:
   Top row:
     - Original global ERA5 grid
     - Liberal grid points
     - Strict grid points
     - Conservative grid points
   Bottom row:
     - Same 4 but with India boundary overlayed

3) Processes monthly ERA5 GRIB files → India-only Parquet using a chosen mask:
   - Variable scanning on first file; verify all files share same var set
   - Robust GRIB loader:
       * Opens each variable separately via cfgrib (filter_by_keys shortName)
       * Flattens forecast "time, step, valid_time" into single time axis
       * Merges variables into one xarray.Dataset
   - Converts Dataset → Polars, joins chosen India mask, filters to mask rows
   - Area-weights extensive variables only (by frac_in_india)
   - Writes monthly Parquet (snappy) via Polars LazyFrame.sink_parquet()
   - Parallel backends: "none", "processpool", "mpi"

CLI flags:
   --skip-masks         : don't (re)build masks
   --skip-plot          : don't generate PNG comparison
   --skip-processing    : don't process GRIB files
   --mask-mode MODE     : liberal | strict | conservative

"""

from __future__ import annotations

from pathlib import Path
import logging
from typing import Dict, List, Set
import time
import re
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import xarray as xr
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, Point
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

# Base fractional mask (with frac_in_india + centroid flag)
MASK_CACHE_PATH = Path("india_grid_mask.parquet")

# Derived masks
LIBERAL_MASK_PATH = Path("india_mask_liberal.parquet")
STRICT_MASK_PATH = Path("india_mask_strict.parquet")
CONSERVATIVE_MASK_PATH = Path("india_mask_conservative.parquet")

# Threshold for "fully inside" (to avoid float noise)
STRICT_THRESHOLD = 0.999  # treat ≥ 99.9% area as "fully inside"

# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND = "processpool"  # "none" | "processpool" | "mpi"

# workers for processpool backend (2/3 of maximum available CPUs, min 2)
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Mask mode for processing: "liberal", "strict", "conservative"
MASK_MODE_DEFAULT = "liberal"

# Logging
LOG_LEVEL = logging.INFO

# Overwrite existing monthly Parquet outputs?
OVERWRITE_EXISTING = True

# Drop island grid points (Lakshadweep, Andaman & Nicobar)?
DROP_ISLANDS = True

# Visualisation PNG
GRID_PNG_PATH = Path("grid_masks_comparison.png")


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
# VAR HELPERS FOR SAMPLE GRID
# ---------------------------------------------------------------------


def scan_parameters_in_file(path: Path):
    """Return list of {'shortName': ..., 'paramId': ...} from a GRIB file."""
    params: Dict[int, Dict] = {}
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
    return list(params.values())


def select_example_var(path: Path) -> str:
    """Pick a safe variable present in the GRIB file for grid extraction."""
    params = scan_parameters_in_file(path)
    sns = {p["shortName"] for p in params}

    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    # fallback: first one alphabetically
    return sorted(list(sns))[0]


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


def build_fractional_mask(sample_grib: Path, sample_shortname: str) -> pl.DataFrame:
    """
    Build a grid mask with fractional overlap for a sample GRIB file.

    Loads only a single variable (sample_shortname) to avoid GRIB edition issues.

    Returns a Polars DataFrame with:
      - latitude
      - longitude
      - frac_in_india (0..1)
      - cell_area_m2
      - centroid_in_india (bool)
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
        raise ValueError("This script currently expects 1D latitude/longitude arrays.")

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

            # centroid-based membership (point in polygon)
            centroid = Point(float(lon_c), float(lat_c))
            centroid_inside = india_geom.contains(centroid)

            rows.append(
                {
                    "latitude": float(lat_c),
                    "longitude": float(lon_c),
                    "frac_in_india": float(frac),
                    "cell_area_m2": float(cell_area),
                    "centroid_in_india": bool(centroid_inside),
                }
            )

    ds.close()

    mask_df = pl.DataFrame(rows)
    non_zero = (mask_df["frac_in_india"] > 0).sum()
    logging.info("Base mask built. Non-zero cells (any overlap): %d", non_zero)

    return mask_df


def ensure_all_masks(sample_grib: Path, sample_shortname: str) -> Dict[str, Path]:
    """
    Ensure that:
      - fractional base mask exists
      - liberal / strict / conservative masks exist

    Returns dict:
      {
        "liberal": LIBERAL_MASK_PATH,
        "strict": STRICT_MASK_PATH,
        "conservative": CONSERVATIVE_MASK_PATH
      }
    """
    if MASK_CACHE_PATH.exists():
        logging.info("Loading existing fractional mask from %s", MASK_CACHE_PATH)
        mask_df = pl.read_parquet(MASK_CACHE_PATH)
    else:
        MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        mask_df = build_fractional_mask(sample_grib, sample_shortname)
        logging.info("Saving fractional mask to %s", MASK_CACHE_PATH)
        mask_df.write_parquet(MASK_CACHE_PATH, compression="snappy")

    # Liberal: any overlap
    if not LIBERAL_MASK_PATH.exists():
        logging.info("Building liberal mask: any frac_in_india > 0")
        liberal_df = mask_df.filter(pl.col("frac_in_india") > 0.0)
        liberal_df.write_parquet(LIBERAL_MASK_PATH, compression="snappy")

    # Strict: fully inside (fraction ≈ 1.0)
    if not STRICT_MASK_PATH.exists():
        logging.info(
            "Building strict mask: frac_in_india >= %.4f", STRICT_THRESHOLD
        )
        strict_df = mask_df.filter(pl.col("frac_in_india") >= STRICT_THRESHOLD)
        strict_df.write_parquet(STRICT_MASK_PATH, compression="snappy")

    # Conservative:
    #   keep:
    #     - fully inside (frac_in_india >= STRICT_THRESHOLD), OR
    #     - partial but with centroid inside (frac_in_india > 0 & centroid_in_india)
    if not CONSERVATIVE_MASK_PATH.exists():
        logging.info("Building conservative mask")
        conservative_df = mask_df.filter(
            (pl.col("frac_in_india") >= STRICT_THRESHOLD)
            | ((pl.col("frac_in_india") > 0.0) & pl.col("centroid_in_india"))
        )
        conservative_df.write_parquet(CONSERVATIVE_MASK_PATH, compression="snappy")

    return {
        "liberal": LIBERAL_MASK_PATH,
        "strict": STRICT_MASK_PATH,
        "conservative": CONSERVATIVE_MASK_PATH,
    }


# ---------------------------------------------------------------------
# OPTIONAL: REMOVE ISLAND GRID POINTS BY LAT/LON BOUNDS
# ---------------------------------------------------------------------


def drop_island_points(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Remove ERA5 grid points corresponding to island regions
    using simple lat/lon bounding boxes.

    Current boxes (tweak as needed):
      - Andaman & Nicobar:   lon 90–95E,  lat < 15N
      - Lakshadweep:         lon 71–74E,  lat ~ 8.2–11.7N
    """
    if not DROP_ISLANDS:
        return lf

    lon = pl.col("longitude")
    lat = pl.col("latitude")

    andaman_nicobar = (lon >= 90.0) & (lon <= 95.0) & (lat < 15.0)
    lakshadweep = (lon >= 71.0) & (lon <= 74.0) & (lat >= 8.2) & (lat <= 11.7)

    return lf.filter(~(andaman_nicobar | lakshadweep))


# ---------------------------------------------------------------------
# FILENAME PARSERS
# ---------------------------------------------------------------------


def build_output_filename(path: Path) -> str:
    """
    Convert input GRIB filename:
        dataset_region_uniqueid_year_month.grib
    into:
        dataset_INDIA_uniqueid_year_month.parquet
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
        f"INDIA_"
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
# GRIB → XARRAY (PER-VARIABLE) → POLARS → INDIA CLIP
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

    out_name = build_output_filename(path)
    out_path = OUTPUT_MONTHLY_DIR / out_name

    if out_path.exists() and not OVERWRITE_EXISTING:
        logging.info("Output already exists for %s, skipping.", path.name)
        return 0.0
    elif out_path.exists() and OVERWRITE_EXISTING:
        logging.info("Overwriting existing file for %s", path.name)

    logging.info("Processing %s -> %s", path.name, out_path.name)

    # 1) Load GRIB → xarray
    ds = load_grib_to_xarray(path, var_shortnames)

    # 2) Flatten to pandas, then to Polars
    df = ds.to_dataframe().reset_index()
    ds.close()

    pl_df = pl.from_pandas(df)

    # 3) Lazy join with selected India mask
    lf = pl_df.lazy()
    mask_lf = pl.scan_parquet(mask_path)

    coord_cols = ["latitude", "longitude"]
    for c in coord_cols:
        if c not in pl_df.columns:
            raise KeyError(
                f"Expected coordinate column '{c}' in {path.name}, got {pl_df.columns}"
            )

    lf = lf.join(mask_lf, on=coord_cols, how="inner")

    # Optional drop islands
    lf = drop_island_points(lf)

    # 4) Area-weight extensive variables (using frac_in_india)
    for var in EXTENSIVE_VARS:
        if var in pl_df.columns:
            lf = lf.with_columns(
                (pl.col(var) * pl.col("frac_in_india")).alias(var)
            )

    # 5) Filter out cells with zero weight (just in case)
    lf = lf.filter(pl.col("frac_in_india") > 0.0)

    # 6) Write Parquet lazily
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


def run_sequential(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
) -> List[float]:
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


def run_mpi(
    grib_files: List[Path],
    mask_path: Path,
    var_shortnames: List[str],
) -> List[float]:
    if MPI is None:
        raise RuntimeError("mpi4py not available.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_durations: List[float] = []

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
# ORIGINAL GRID + PLOTTING MASKS (2 x 4)
# ---------------------------------------------------------------------


def load_original_grid_points(path: Path, example_var: str):
    """
    Load original grid as flat arrays of lon/lat using a single variable.
    """
    logging.info(
        "Loading original grid from GRIB %s using var %s", path.name, example_var
    )

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": example_var},
        },
    )

    lat = ds["latitude"]
    lon = ds["longitude"]

    lon2d, lat2d = xr.broadcast(lon, lat)

    lats = lat2d.values.ravel()
    lons = lon2d.values.ravel()

    ds.close()
    return lons, lats


def plot_masks_comparison(
    sample_grib: Path,
    mask_paths: Dict[str, Path],
    output_png: Path = GRID_PNG_PATH,
):
    """
    Create a 2x4 static matplotlib comparison plot:

    Top row:
      - Original global grid
      - Liberal mask points
      - Strict mask points
      - Conservative mask points

    Bottom row:
      - Same 4, with India boundary overlayed
    """
    logging.info("Creating grid masks comparison PNG: %s", output_png)

    example_var = select_example_var(sample_grib)
    lon_orig, lat_orig = load_original_grid_points(sample_grib, example_var)

    # Load masks
    liberal_df = pl.read_parquet(mask_paths["liberal"])
    strict_df = pl.read_parquet(mask_paths["strict"])
    conservative_df = pl.read_parquet(mask_paths["conservative"])

    lon_lib = liberal_df["longitude"].to_numpy()
    lat_lib = liberal_df["latitude"].to_numpy()

    lon_str = strict_df["longitude"].to_numpy()
    lat_str = strict_df["latitude"].to_numpy()

    lon_con = conservative_df["longitude"].to_numpy()
    lat_con = conservative_df["latitude"].to_numpy()

    # India boundary
    india_gdf = gpd.read_file(INDIA_BOUNDARY_GEOJSON).to_crs(epsg=4326)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    (ax1, ax2, ax3, ax4), (bx1, bx2, bx3, bx4) = axes

    # Top row: no boundary
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original ERA5 Grid (Global Region)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    ax2.scatter(lon_lib, lat_lib, s=4, color="blue")
    ax2.set_title("Liberal Mask (any overlap)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    ax3.scatter(lon_str, lat_str, s=4, color="green")
    ax3.set_title(f"Strict Mask (frac ≥ {STRICT_THRESHOLD:.3f})")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    ax4.scatter(lon_con, lat_con, s=4, color="orange")
    ax4.set_title("Conservative Mask")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")

    # Bottom row: with India boundary
    india_gdf.plot(ax=bx1, facecolor="none", edgecolor="red", linewidth=1.0)
    bx1.scatter(lon_orig, lat_orig, s=1, color="black")
    bx1.set_title("Original Grid + India Boundary")
    bx1.set_xlabel("Longitude")
    bx1.set_ylabel("Latitude")

    india_gdf.plot(ax=bx2, facecolor="none", edgecolor="red", linewidth=1.0)
    bx2.scatter(lon_lib, lat_lib, s=4, color="blue")
    bx2.set_title("Liberal Mask + Boundary")
    bx2.set_xlabel("Longitude")
    bx2.set_ylabel("Latitude")

    india_gdf.plot(ax=bx3, facecolor="none", edgecolor="red", linewidth=1.0)
    bx3.scatter(lon_str, lat_str, s=4, color="green")
    bx3.set_title("Strict Mask + Boundary")
    bx3.set_xlabel("Longitude")
    bx3.set_ylabel("Latitude")

    india_gdf.plot(ax=bx4, facecolor="none", edgecolor="red", linewidth=1.0)
    bx4.scatter(lon_con, lat_con, s=4, color="orange")
    bx4.set_title("Conservative Mask + Boundary")
    bx4.set_xlabel("Longitude")
    bx4.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close(fig)

    logging.info("Saved grid masks comparison PNG to %s", output_png)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main(
    skip_masks: bool,
    skip_plot: bool,
    skip_processing: bool,
    mask_mode: str,
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

    # Sample file for masks + visualisation
    first_file = grib_files[0]

    # ----- MASK BUILDING -----
    if skip_masks:
        logging.info("Skipping mask building as requested.")
        mask_paths = {
            "liberal": LIBERAL_MASK_PATH,
            "strict": STRICT_MASK_PATH,
            "conservative": CONSERVATIVE_MASK_PATH,
        }
    else:
        # For mask building, pick "2t"/"t2m" or first shortname
        params_first = scan_parameters_for_processing(first_file)
        shortnames_ref = {p["shortName"] for p in params_first}
        var_shortnames_sorted = sorted(shortnames_ref)

        if "2t" in shortnames_ref:
            sample_var = "2t"
        elif "t2m" in shortnames_ref:
            sample_var = "t2m"
        else:
            sample_var = var_shortnames_sorted[0]

        mask_paths = ensure_all_masks(first_file, sample_var)

    # ----- PLOTTING -----
    if not skip_plot:
        try:
            plot_masks_comparison(first_file, mask_paths, GRID_PNG_PATH)
        except Exception as e:  # noqa: BLE001
            logging.exception("Error while plotting grid masks comparison: %s", e)

    # ----- PROCESSING -----
    if skip_processing:
        logging.info("Skipping GRIB → Parquet processing as requested.")
        return

    # 1) Variable scan on first file for processing
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

    # Which mask to use for processing?
    if mask_mode not in ("liberal", "strict", "conservative"):
        raise ValueError(
            f"Invalid mask_mode={mask_mode}. Use 'liberal', 'strict', or 'conservative'."
        )
    selected_mask_path = mask_paths[mask_mode]
    logging.info("Using '%s' mask for processing: %s", mask_mode, selected_mask_path)

    logging.info("Starting processing with backend: %s", PARALLEL_BACKEND)

    if PARALLEL_BACKEND == "none":
        durations = run_sequential(
            grib_files,
            selected_mask_path,
            var_shortnames_sorted_proc,
        )
    elif PARALLEL_BACKEND == "processpool":
        durations = run_processpool(
            grib_files,
            selected_mask_path,
            var_shortnames_sorted_proc,
        )
    elif PARALLEL_BACKEND == "mpi":
        durations = run_mpi(
            grib_files,
            selected_mask_path,
            var_shortnames_sorted_proc,
        )
    else:
        raise ValueError(
            f"Unknown PARALLEL_BACKEND={PARALLEL_BACKEND}. "
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
            "Build ERA5 India masks (fractional + liberal/strict/conservative), "
            "visualise them, and process GRIB → India-only Parquet."
        )
    )
    parser.add_argument(
        "--skip-masks",
        action="store_true",
        help="Skip building masks (assumes mask parquet files already exist).",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the 2x4 grid comparison PNG.",
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip GRIB → Parquet processing.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["liberal", "strict", "conservative"],
        default=MASK_MODE_DEFAULT,
        help="Which mask variant to use for GRIB processing.",
    )

    args = parser.parse_args()

    main(
        skip_masks=args.skip_masks,
        skip_plot=args.skip_plot,
        skip_processing=args.skip_processing,
        mask_mode=args.mask_mode,
    )
