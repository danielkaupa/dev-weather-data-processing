"""
02_process_grib_to_parquet.py

Convert ERA5 global GRIB files to India-only Parquet, with exact
fractional overlap per grid cell and area-weighting of extensive variables.

Features:
- Uses India polygon from extract_india_boundary.py
- Builds or loads a grid-cell fractional mask (frac_in_india, area)
- Applies fractional area weighting to extensive variables only
- Drops cells outside India (frac == 0)
- Writes monthly Parquet (wide format, snappy compression)
- Configurable parallel backend: "none", "processpool", "mpi"
- Variable scanning using eccodes for robustness
"""

from __future__ import annotations

from pathlib import Path
import logging
import json
from typing import Dict, List

import xarray as xr
import polars as pl
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from pyproj import Geod

from cfgrib.dataset import DatasetBuildError

# eccodes for variable scanning
from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

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

# India boundary GeoJSON from Script 1
INDIA_BOUNDARY_GEOJSON = Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")

# Variable metadata JSON (shortName -> {paramId, fullName, units})
VARIABLE_METADATA_JSON = Path("era5-world_variable_metadata.json")

# Cache path for grid mask with fractional overlap
MASK_CACHE_PATH = Path("india_grid_mask.parquet")

# Parallel backend: "none", "processpool", or "mpi"
PARALLEL_BACKEND = "processpool"  # "none" | "processpool" | "mpi"

# Max workers for processpool backend
MAX_WORKERS = 2
# max(multiprocessing.cpu_count() - 1, 1)

# Logging
LOG_LEVEL = logging.INFO

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
# UTILS: VARIABLE SCANNING
# ---------------------------------------------------------------------


def scan_variables_in_file(path: Path) -> List[Dict]:
    """
    Scan a GRIB file with ecCodes and return a list of dictionaries:
    [
        { "paramId": 167, "shortName": "2t" },
        { "paramId": 228, "shortName": "tp" },
        ...
    ]

    Uses low-level eccodes API.
    """
    variables: Dict[int, Dict] = {}
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

                    # decode if bytes
                    if isinstance(short_name, bytes):
                        short_name = short_name.decode("utf-8")

                    variables[param_id] = {
                        "paramId": int(param_id),
                        "shortName": str(short_name),
                    }

                finally:
                    codes_release(gid)

    except FileNotFoundError:
        logging.error("File not found during scan: %s", path)
    except Exception as e:  # noqa: BLE001
        logging.exception("Unexpected error while scanning %s: %s", path, e)

    return list(variables.values())


def load_variable_metadata() -> Dict[str, Dict]:
    if not VARIABLE_METADATA_JSON.exists():
        logging.warning("Variable metadata JSON not found at %s", VARIABLE_METADATA_JSON)
        return {}
    with VARIABLE_METADATA_JSON.open("r") as f:
        metadata = json.load(f)
    return metadata


# ---------------------------------------------------------------------
# UTILS: GEOMETRY & MASK
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


def build_mask_from_sample_grib(sample_grib: Path) -> pl.DataFrame:
    ...
    ds = xr.open_dataset(
        sample_grib,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": "2t"},  # or tp, 10u, etc.
        },
    )

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        ds.close()
        raise KeyError("Dataset must contain 'latitude' and 'longitude' coordinates.")

    lat = ds["latitude"]
    lon = ds["longitude"]
    ...
    mask_df = pl.DataFrame(rows)
    logging.info("Mask built. Non-zero cells: %d", (mask_df["frac_in_india"] > 0).sum())

    ds.close()  # <-- add this
    return mask_df


def ensure_mask(grib_files: List[Path]) -> pl.DataFrame:
    """
    Load India grid mask if it exists; otherwise build from first GRIB file
    and cache to Parquet.
    """
    if MASK_CACHE_PATH.exists():
        logging.info("Loading existing India mask from %s", MASK_CACHE_PATH)
        return pl.read_parquet(MASK_CACHE_PATH)

    if not grib_files:
        raise ValueError("No GRIB files found to build mask from.")

    sample_grib = grib_files[0]
    mask_df = build_mask_from_sample_grib(sample_grib)

    MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Saving India mask to %s", MASK_CACHE_PATH)
    mask_df.write_parquet(MASK_CACHE_PATH, compression="snappy")

    return mask_df


# ---------------------------------------------------------------------
# GRIB → POLARS → INDIA CLIP
# ---------------------------------------------------------------------

def parse_year_month_from_name(path: Path) -> tuple[int, int]:
    """
    Example input:
      era5-world_N37W68S6E98_d514a3a3c256_2025_04.grib
    """
    stem = path.stem
    parts = stem.split("_")
    year = int(parts[-2])
    month = int(parts[-1])
    return year, month


def _load_grib_full(path: Path) -> xr.Dataset:
    """
    Try to load the entire GRIB as a single xarray Dataset.
    This may fail with DatasetBuildError if there are mixed GRIB editions.
    """
    return xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )


def _load_grib_by_variable_merge(path: Path) -> xr.Dataset:
    """
    Robust fallback: load each variable separately with filter_by_keys
    and merge them into a single Dataset.

    Uses scan_variables_in_file(path) to know which shortNames to load.
    """
    logging.warning(
        "Falling back to per-variable loading for %s due to cfgrib edition mismatch.",
        path.name,
    )

    var_infos = scan_variables_in_file(path)
    shortnames = sorted({v["shortName"] for v in var_infos})

    if not shortnames:
        raise RuntimeError(f"No variables found via eccodes in {path}")

    datasets = []
    for sn in shortnames:
        try:
            logging.debug("Loading variable %s from %s", sn, path.name)
            ds_var = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {"shortName": sn},
                },
            )
            datasets.append(ds_var)
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "Failed to load variable %s from %s: %s", sn, path.name, e
            )

    if not datasets:
        raise RuntimeError(
            f"Failed to load any variables from {path} via per-variable merging."
        )

    # Merge all variable-specific datasets into one
    ds_merged = xr.merge(datasets, combine_attrs="override")

    # Close individual ds to free resources
    for ds_var in datasets:
        ds_var.close()

    return ds_merged


def load_grib_to_polars(path: Path) -> pl.DataFrame:
    """
    Load a GRIB file into a Polars DataFrame, robust to mixed GRIB editions.

    Strategy:
      1. Try loading the full file in one go (fast path).
      2. If DatasetBuildError is raised, fall back to per-variable loading.
    """
    try:
        ds = _load_grib_full(path)
    except DatasetBuildError:
        logging.warning(
            "DatasetBuildError (mixed 'edition') when loading %s. "
            "Switching to robust per-variable merge.",
            path.name,
        )
        ds = _load_grib_by_variable_merge(path)

    # Flatten to pandas then to polars
    df = ds.to_dataframe().reset_index()
    ds.close()

    # Normalize time column
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    elif "valid_time" in df.columns:
        df = df.rename(columns={"valid_time": "timestamp"})
    else:
        raise KeyError(f"No 'time' or 'valid_time' column found in {path}")

    # Convert to Polars
    pl_df = pl.from_pandas(df)

    # Ensure timestamp is Datetime
    pl_df = pl_df.with_columns(
        pl.col("timestamp").cast(pl.Datetime(time_unit="us"), strict=False)
    )

    return pl_df



def process_one_grib_file(
    path: Path,
    mask_path: Path,
    variable_metadata: Dict[str, Dict],
) -> None:

    try:
        year, month = parse_year_month_from_name(path)
    except Exception:
        logging.warning("Unable to parse year/month from %s. Skipping.", path)
        return

    out_name = f"era5_india_{year}_{month:02d}.parquet"
    out_path = OUTPUT_MONTHLY_DIR / out_name

    if out_path.exists():
        logging.info("Output already exists for %s, skipping.", path.name)
        return

    logging.info("Processing %s -> %s", path.name, out_path.name)

    # ------------------------------------------------------------------
    # STEP 1 — Load GRIB → Polars (eager, unavoidable)
    # ------------------------------------------------------------------
    pl_df = load_grib_to_polars(path)

    # Immediately convert to lazy
    lf = pl_df.lazy()

    # ------------------------------------------------------------------
    # STEP 2 — Lazy load mask
    # ------------------------------------------------------------------
    mask_lf = pl.scan_parquet(mask_path)

    coord_cols = ["latitude", "longitude"]

    # ------------------------------------------------------------------
    # STEP 3 — Lazy Join + Filter
    # ------------------------------------------------------------------
    lf = (
        lf.join(mask_lf, on=coord_cols, how="inner")
          .filter(pl.col("frac_in_india") > 0)
    )

    # ------------------------------------------------------------------
    # STEP 4 — Lazy area-weighting of extensive variables
    # ------------------------------------------------------------------
    for var in EXTENSIVE_VARS:
        if var in pl_df.columns:
            lf = lf.with_columns(
                (pl.col(var) * pl.col("frac_in_india")).alias(var)
            )

    # ------------------------------------------------------------------
    # STEP 5 — Write Parquet lazily
    # ------------------------------------------------------------------
    lf.sink_parquet(out_path, compression="snappy")

    logging.info("Finished %s", path.name)



# ---------------------------------------------------------------------
# PARALLEL DISPATCH
# ---------------------------------------------------------------------


def run_sequential(grib_files: List[Path], mask_path: Path, var_meta: Dict[str, Dict]):
    for p in grib_files:
        process_one_grib_file(p, mask_path, var_meta)


def run_processpool(grib_files: List[Path], mask_path: Path, var_meta: Dict[str, Dict]):
    # var_meta small enough to be pickled; mask_path is global
    args = [(p, mask_path, var_meta) for p in grib_files]

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(_worker_process_one_grib, args))


def _worker_process_one_grib(args):
    path, mask_path, var_meta = args
    # re-init logging in worker
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] [worker] %(message)s",
    )
    process_one_grib_file(path, mask_path, var_meta)


def run_mpi(grib_files: List[Path], mask_path: Path, var_meta: Dict[str, Dict]):
    if MPI is None:
        raise RuntimeError(
            "mpi4py is not available but PARALLEL_BACKEND=='mpi'. "
            "Install mpi4py or change PARALLEL_BACKEND."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simple workload split: each rank processes files where index % size == rank
    for idx, p in enumerate(grib_files):
        if idx % size != rank:
            continue
        # Each rank runs its own process_one_grib_file
        logging.info(
            "Rank %d processing file %d/%d: %s",
            rank,
            idx + 1,
            len(grib_files),
            p.name,
        )
        process_one_grib_file(p, mask_path, var_meta)

    # Barrier to ensure all ranks complete
    comm.Barrier()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main() -> None:
    setup_logging()

    if not RAW_GRIB_DIR.exists():
        raise FileNotFoundError(f"RAW_GRIB_DIR does not exist: {RAW_GRIB_DIR}")

    grib_files = sorted(RAW_GRIB_DIR.glob("*.grib"))
    if not grib_files:
        logging.error("No GRIB files found in %s", RAW_GRIB_DIR)
        return

    var_meta = load_variable_metadata()

    # Build or load mask
    mask_df = ensure_mask(grib_files)
    # Save was already done if we built it; just ensure path exists
    if not MASK_CACHE_PATH.exists():
        MASK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        mask_df.write_parquet(MASK_CACHE_PATH, compression="snappy")

    logging.info("Starting processing with backend: %s", PARALLEL_BACKEND)

    if PARALLEL_BACKEND == "none":
        run_sequential(grib_files, MASK_CACHE_PATH, var_meta)
    elif PARALLEL_BACKEND == "processpool":
        run_processpool(grib_files, MASK_CACHE_PATH, var_meta)
    elif PARALLEL_BACKEND == "mpi":
        run_mpi(grib_files, MASK_CACHE_PATH, var_meta)
    else:
        raise ValueError(
            f"Unknown PARALLEL_BACKEND={PARALLEL_BACKEND}. "
            f"Use 'none', 'processpool', or 'mpi'."
        )

    logging.info("All files processed.")


if __name__ == "__main__":
    main()