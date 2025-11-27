"""
01_build_country_mask.py

Build a grid mask for a target country on an ERA5-style lat/lon grid.

Features
--------
- Country-agnostic: works with any boundary file (ADM0/ADM1/etc.).
- Supports shapefile (.shp) and GeoJSON (.geojson) boundaries.
- Uses an ERA5 GRIB file to derive the lat/lon grid & cell polygons.
- Inclusion rules:
    * fractional: frac_in_region >= FRACTION_THRESHOLD
    * centroid:  centroid of cell inside region
    * combined:  both conditions must hold
- Exclusions:
    * Custom lat/lon bounding boxes (e.g. islands).
- Mask naming:
    * {dataset_prefix}_{country_token}_mask_{mode_and_thresholds}_{excl_hash}.parquet
- Safe reuse:
    * If mask with same config name already exists, can reuse or overwrite.
- Optional preview PNG: shows mask grid + boundary outline.

Grid resolution & cost
----------------------
The script:
1. Takes one sample GRIB file (ERA5-style) from RAW_GRIB_DIR.
2. Reads 1D latitude/longitude from a single variable via cfgrib.
3. Restricts to the bounding box of the country polygon before looping
   to reduce computation time.

You can later wrap main() into a CLI that passes in these configs.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import logging
import multiprocessing
import time
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
import xarray as xr
from eccodes import (
    CodesInternalError,
    codes_get,
    codes_grib_new_from_file,
    codes_release,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


# ---------------------------------------------------------------------
# USER CONFIGURATION (EASY TO TURN INTO CLI OPTIONS LATER)
# ---------------------------------------------------------------------

# Directory containing ERA5 GRIB files (we use one as a grid template)
RAW_GRIB_DIR = Path("../data/raw")

# Country boundary file (can be .shp or .geojson)
BOUNDARY_PATH = Path("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp")

# ADM level hint for heuristics (0 = national, 1 = states, etc.)
ADM_LEVEL: Optional[int] = 0

# Country name in the boundary dataset (if None, assume boundary file
# is already a single-country subset).
COUNTRY_NAME: Optional[str] = "India"

# Where to write the final mask parquet files
MASK_OUTPUT_DIR = Path("masks")

# Dataset prefix for file naming.
# For ERA5 this should match the GRIB prefix (e.g. "era5-world")
DATASET_PREFIX: Optional[str] = None  # if None, inferred from GRIB filename

# Inclusion mode: "fractional", "centroid", or "combined"
INCLUSION_MODE = "combined"

# Initialise to None, then resolve based on mode
FRACTION_THRESHOLD: Optional[float] = None
USE_CENTROID: Optional[bool] = None

if INCLUSION_MODE == "fractional":
    FRACTION_THRESHOLD = 0.90  # user-defined
    USE_CENTROID = False
elif INCLUSION_MODE == "centroid":
    FRACTION_THRESHOLD = None
    USE_CENTROID = True
elif INCLUSION_MODE == "combined":
    FRACTION_THRESHOLD = 0.90  # user-defined
    USE_CENTROID = True
else:
    raise ValueError(f"Unknown INCLUSION_MODE: {INCLUSION_MODE}")

# Exclusion bounding boxes: list of dicts with lon/lat bounds
# Example: remove Andaman & Nicobar and Lakshadweep for India.
EXCLUSION_BBOXES: List[Dict[str, float]] = [
    {
        "name": "andaman_nicobar",
        "lon_min": 90.0,
        "lon_max": 95.0,
        "lat_min": -10.0,
        "lat_max": 15.0,
    },
    {
        "name": "lakshadweep",
        "lon_min": 71.0,
        "lon_max": 74.0,
        "lat_min": 8.2,
        "lat_max": 11.7,
    },
]

# Whether to apply the exclusion bounding boxes
APPLY_EXCLUSIONS = True

# Reuse existing mask if a file with the same config-derived name exists?
REUSE_EXISTING_IF_FOUND = False
OVERWRITE_EXISTING = True

# If True, generate a preview PNG with grid + boundary overlay
GENERATE_PREVIEW_PNG = True

# Logging
LOG_LEVEL = logging.INFO

# Parallelisation settings
MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)
PARALLEL_BACKEND = "processpool"  # "none" | "processpool" | "mpi"
PARALLEL_MIN_CELLS = 10_000               #100_000  # min cells to trigger parallel
PARALLEL_CHUNK_SIZE = 5_000               #50_000

# Equal-area CRS for area computation
EA_CRS = "EPSG:6933"


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------


def setup_logging() -> None:
    """
    Configure logging for the mask-building script.

    This function:
    - Ensures that the output directory exists.
    - Sets a global logging configuration with time, level, and message.

    Returns
    -------
    None
    """
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# HELPER: SCAN GRIB PARAMETERS TO FIND A SAFE VARIABLE
# ---------------------------------------------------------------------


def scan_parameters_in_file(path: Path) -> List[Dict[str, str]]:
    """
    Scan a GRIB file with ecCodes and return a list of parameter descriptors.

    Parameters
    ----------
    path : pathlib.Path
        Path to the GRIB file.

    Returns
    -------
    list of dict
        List of parameter dictionaries, each containing:
        - 'paramId' : int
        - 'shortName' : str
    """
    params: Dict[int, Dict[str, str]] = {}

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


def select_example_var(grib_path: Path) -> str:
    """
    Select a representative variable present in the GRIB file.

    The function prefers common near-surface temperature fields such as
    ``2t`` or ``t2m``, and otherwise falls back to the first shortName
    in alphabetical order.

    Parameters
    ----------
    grib_path : pathlib.Path
        Path to the GRIB file.

    Returns
    -------
    str
        Selected GRIB shortName.

    Raises
    ------
    RuntimeError
        If no GRIB parameters are found in the file.
    """
    params = scan_parameters_in_file(grib_path)
    if not params:
        raise RuntimeError(f"No GRIB messages found in {grib_path}")

    sns = {p["shortName"] for p in params}
    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    return sorted(sns)[0]


# ---------------------------------------------------------------------
# LOAD GRID FROM GRIB (LAT/LON ONLY)
# ---------------------------------------------------------------------


def load_grid_from_grib(
    grib_path: Path,
    shortname: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ERA5-like 1D latitude and longitude arrays from a GRIB file.

    Parameters
    ----------
    grib_path : pathlib.Path
        Path to the GRIB file.
    shortname : str, optional
        GRIB ``shortName`` of the variable used to read the grid. If
        ``None``, a representative variable is selected automatically
        via :func:`select_example_var`.

    Returns
    -------
    lat_vals : numpy.ndarray
        1D array of latitude values.
    lon_vals : numpy.ndarray
        1D array of longitude values.

    Raises
    ------
    KeyError
        If latitude/longitude coordinates cannot be inferred from the
        dataset.
    ValueError
        If the latitude or longitude arrays are not 1D.
    """
    if shortname is None:
        shortname = select_example_var(grib_path)
        logging.info("Selected variable '%s' for grid extraction", shortname)

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": shortname},
        },
    )

    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
    elif "lat" in ds.coords and "lon" in ds.coords:
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
    else:
        coords = list(ds.coords)
        ds.close()
        raise KeyError(
            f"Could not find lat/lon coords in {grib_path.name}; "
            f"found coordinates: {coords}"
        )

    ds.close()

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError(
            "Expected 1D latitude/longitude arrays (global regular grid)."
        )

    return lat_vals, lon_vals


# ---------------------------------------------------------------------
# BOUNDARY LOADER (COUNTRY-AGNOSTIC, SHP/GEOJSON)
# ---------------------------------------------------------------------


def infer_country_field(
    gdf: gpd.GeoDataFrame,
    adm_level: Optional[int],
) -> List[str]:
    """
    Suggest candidate field names for matching the country name.

    The function returns candidate column names in priority order based
    on the administrative level hint. Only columns that actually exist
    in the GeoDataFrame are returned.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing boundary geometries.
    adm_level : int or None
        Administrative level hint (0 = national, 1 = state/province,
        etc.). If ``None``, only generic candidates are considered.

    Returns
    -------
    list of str
        Candidate column names, ordered from most to least preferred.
    """
    candidates_common = ["shapeName", "ADMIN", "NAME", "name", "COUNTRY", "country"]

    if adm_level == 0:
        candidates = ["shapeName", "ADM0_NAME", "NAME_EN", "NAME_0"] + candidates_common
    elif adm_level == 1:
        candidates = ["shapeName", "ADM1_NAME", "NAME_1"] + candidates_common
    else:
        candidates = candidates_common

    return [c for c in candidates if c in gdf.columns]


def load_country_geometry(
    path: Path,
    country_name: Optional[str],
    adm_level: Optional[int],
) -> Tuple[MultiPolygon, str]:
    """
    Load a boundary dataset (SHP/GeoJSON) and extract the target region.

    Parameters
    ----------
    path : pathlib.Path
        Path to the boundary file (e.g. SHP or GeoJSON).
    country_name : str or None
        Name of the country/region within the dataset. If ``None``,
        the entire file is assumed to represent the target region.
    adm_level : int or None
        Administrative level hint for selecting the attribute used to
        match ``country_name``.

    Returns
    -------
    region_geom : shapely.geometry.MultiPolygon
        Geometry of the selected region in EPSG:4326.
    country_token : str
        Sanitised string token for use in filenames (e.g. ``INDIA``).

    Raises
    ------
    KeyError
        If no plausible attribute fields are found for matching.
    ValueError
        If the requested ``country_name`` cannot be found in the
        boundary file.
    TypeError
        If the resulting geometry is not a Polygon or MultiPolygon.
    """
    logging.info("Loading boundary file: %s", path)
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)

    if country_name is None:
        if len(gdf) > 1:
            logging.warning(
                "Boundary file has %d features but COUNTRY_NAME is None. "
                "Using union of all geometries as the region.",
                len(gdf),
            )
        subset = gdf
        country_token = "region"
    else:
        candidates = infer_country_field(gdf, adm_level)
        if not candidates:
            raise KeyError(
                f"No candidate country fields found for ADM{adm_level}. "
                f"Available columns: {list(gdf.columns)}. "
                f"Please inspect the boundary file and update the script config."
            )

        matched_field: Optional[str] = None
        subset: Optional[gpd.GeoDataFrame] = None
        for field in candidates:
            if country_name in set(gdf[field]):
                subset = gdf[gdf[field] == country_name]
                matched_field = field
                break

        if subset is None or subset.empty:
            raise ValueError(
                f"Could not find COUNTRY_NAME='{country_name}' in any of the "
                f"candidate fields {candidates}. Available columns: {list(gdf.columns)}. "
                f"Please check the boundary file and update COUNTRY_NAME / ADM_LEVEL."
            )

        logging.info(
            "Filtered boundary on %s == '%s' (%d feature(s)).",
            matched_field,
            country_name,
            len(subset),
        )
        country_token = country_name.upper().replace(" ", "_")

    geom = unary_union(subset.geometry)
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    if not isinstance(geom, MultiPolygon):
        raise TypeError("Region geometry is not a Polygon or MultiPolygon.")

    return geom, country_token


# ---------------------------------------------------------------------
# INTERNAL HELPERS FOR GRID PREPARATION & AREA FRACTIONS
# ---------------------------------------------------------------------


def _prepare_grid_cells(
    sample_grib: Path,
    region_geom: MultiPolygon,
    example_var: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare flattened grid cells and centroids for the region bounding box.

    This helper:
    1. Loads the 1D latitude and longitude arrays from the GRIB file.
    2. Restricts them to the bounding box around ``region_geom``.
    3. Constructs all cell polygons and centroids within this sub-grid.

    Parameters
    ----------
    sample_grib : pathlib.Path
        Path to the sample GRIB file used to define the grid.
    region_geom : shapely.geometry.MultiPolygon
        Target region geometry in EPSG:4326.
    example_var : str, optional
        GRIB ``shortName`` to use for grid extraction. If ``None``,
        a variable is selected automatically.

    Returns
    -------
    lat_flat : numpy.ndarray
        Flattened latitude coordinates of grid cell centers.
    lon_flat : numpy.ndarray
        Flattened longitude coordinates of grid cell centers.
    cell_polys : numpy.ndarray
        Vectorised array of cell polygons (WGS84).
    centroid_in_region : numpy.ndarray of bool
        Boolean mask indicating whether each cell centroid lies inside
        the region.
    """
    lat_vals, lon_vals = load_grid_from_grib(sample_grib, example_var)
    dlat = abs(lat_vals[1] - lat_vals[0])
    dlon = abs(lon_vals[1] - lon_vals[0])

    minx, miny, maxx, maxy = region_geom.bounds
    logging.info("Region bounds: lon[%.2f, %.2f], lat[%.2f, %.2f]", minx, maxx, miny, maxy)

    lat_mask = (lat_vals >= (miny - dlat)) & (lat_vals <= (maxy + dlat))
    lon_mask = (lon_vals >= (minx - dlon)) & (lon_vals <= (maxx + dlon))

    lat_sub = lat_vals[lat_mask]
    lon_sub = lon_vals[lon_mask]

    n_lat, n_lon = len(lat_sub), len(lon_sub)
    total_cells = n_lat * n_lon
    logging.info("Sub-grid: %d lat × %d lon = %d cells", n_lat, n_lon, total_cells)

    lat_2d = np.broadcast_to(lat_sub[:, None], (n_lat, n_lon))
    lon_2d = np.broadcast_to(lon_sub, (n_lat, n_lon))

    lat_flat = lat_2d.ravel()
    lon_flat = lon_2d.ravel()

    half_dlat = dlat / 2.0
    half_dlon = dlon / 2.0

    cell_polys = shapely.box(
        lon_flat - half_dlon,
        lat_flat - half_dlat,
        lon_flat + half_dlon,
        lat_flat + half_dlat,
        ccw=True,
    )

    centroid_points = shapely.points(lon_flat, lat_flat)
    centroid_in_region = shapely.within(centroid_points, region_geom)

    return lat_flat, lon_flat, cell_polys, centroid_in_region


def _compute_fraction_and_area_for_cells(
    cell_polys: np.ndarray,
    region_geom: MultiPolygon,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fractional coverage and area for a batch of grid cells.

    Parameters
    ----------
    cell_polys : numpy.ndarray
        Vectorised shapely polygons representing grid cells in EPSG:4326.
    region_geom : shapely.geometry.MultiPolygon
        Region geometry in EPSG:4326.

    Returns
    -------
    frac_vals : numpy.ndarray
        Fraction of each cell area that lies within the region, clipped
        to the interval [0, 1].
    cell_areas : numpy.ndarray
        Area of each cell in square metres (equal-area projection).
    """
    gdf_cells = gpd.GeoSeries(cell_polys, crs="EPSG:4326").to_crs(EA_CRS)
    gdf_region = gpd.GeoSeries([region_geom], crs="EPSG:4326").to_crs(EA_CRS)
    region_geom_ea = gdf_region.iloc[0]

    cells_ea = gdf_cells.geometry.values
    cell_areas = shapely.area(cells_ea)

    intersections = shapely.intersection(cells_ea, region_geom_ea)
    inter_areas = shapely.area(intersections)

    frac_vals = np.zeros_like(cell_areas, dtype=float)
    mask_nonzero = cell_areas > 0
    frac_vals[mask_nonzero] = (
        inter_areas[mask_nonzero] / cell_areas[mask_nonzero]
    ).clip(0, 1)

    return frac_vals, cell_areas


def _fraction_area_chunk(
    args: Tuple[np.ndarray, MultiPolygon],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Worker function for parallel area/fraction computation.

    Parameters
    ----------
    args : tuple
        Tuple of (cell_polys_chunk, region_geom).

    Returns
    -------
    frac_vals : numpy.ndarray
        Fractional coverage values for the chunk.
    cell_areas : numpy.ndarray
        Cell areas in square metres for the chunk.
    """
    cell_polys_chunk, region_geom = args
    return _compute_fraction_and_area_for_cells(cell_polys_chunk, region_geom)


# ---------------------------------------------------------------------
# BUILD BASE MASK (FRACTION + CENTROID) FOR A COUNTRY
# ---------------------------------------------------------------------


def build_base_mask_optimized(
    sample_grib: Path,
    region_geom: MultiPolygon,
    example_var: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build a base grid mask using a fully vectorised (single-process) approach.

    Parameters
    ----------
    sample_grib : pathlib.Path
        Path to the sample GRIB file used to define the grid.
    region_geom : shapely.geometry.MultiPolygon
        Target region geometry in EPSG:4326.
    example_var : str, optional
        GRIB ``shortName`` for grid extraction. If ``None``, a variable
        is selected automatically.

    Returns
    -------
    polars.DataFrame
        DataFrame containing the base mask with columns:
        - 'latitude' : float
        - 'longitude' : float
        - 'frac_in_region' : float
        - 'centroid_in_region' : bool
        - 'cell_area_m2' : float
    """
    logging.info("Building base mask (optimised, single-process) from %s", sample_grib)

    lat_flat, lon_flat, cell_polys, centroid_in_region = _prepare_grid_cells(
        sample_grib, region_geom, example_var
    )

    frac_vals, cell_areas = _compute_fraction_and_area_for_cells(
        cell_polys, region_geom
    )

    return pl.DataFrame(
        {
            "latitude": lat_flat,
            "longitude": lon_flat,
            "frac_in_region": frac_vals,
            "centroid_in_region": centroid_in_region,
            "cell_area_m2": cell_areas,
        }
    )


def build_base_mask_parallel(
    sample_grib: Path,
    region_geom: MultiPolygon,
    example_var: Optional[str] = None,
    min_cells_parallel: int = PARALLEL_MIN_CELLS,
    max_workers: int = MAX_WORKERS,
) -> pl.DataFrame:
    """
    Build a base grid mask using a process pool for large grids.

    For small grids (below ``min_cells_parallel``), this function falls
    back to the optimised single-process implementation.

    Parameters
    ----------
    sample_grib : pathlib.Path
        Path to the sample GRIB file used to define the grid.
    region_geom : shapely.geometry.MultiPolygon
        Target region geometry in EPSG:4326.
    example_var : str, optional
        GRIB ``shortName`` for grid extraction. If ``None``, a variable
        is selected automatically.
    min_cells_parallel : int, optional
        Minimum number of cells above which parallel processing is used.
    max_workers : int, optional
        Maximum number of worker processes to use.

    Returns
    -------
    polars.DataFrame
        DataFrame containing the base mask with columns:
        - 'latitude' : float
        - 'longitude' : float
        - 'frac_in_region' : float
        - 'centroid_in_region' : bool
        - 'cell_area_m2' : float
    """
    logging.info("Building base mask (parallel-aware) from %s", sample_grib)

    lat_flat, lon_flat, cell_polys, centroid_in_region = _prepare_grid_cells(
        sample_grib, region_geom, example_var
    )

    n_cells = len(cell_polys)
    if n_cells < min_cells_parallel:
        logging.info(
            "Grid has %d cells (< %d); using single-process implementation.",
            n_cells,
            min_cells_parallel,
        )
        frac_vals, cell_areas = _compute_fraction_and_area_for_cells(
            cell_polys, region_geom
        )
    else:
        logging.info(
            "Grid has %d cells (>= %d); using process pool with up to %d workers.",
            n_cells,
            min_cells_parallel,
            max_workers,
        )
        chunk_size = max(PARALLEL_CHUNK_SIZE, n_cells // max_workers)
        chunks: List[np.ndarray] = []
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunks.append(cell_polys[start:end])

        with multiprocessing.Pool(processes=max_workers) as pool:
            results = pool.map(
                _fraction_area_chunk,
                [(chunk, region_geom) for chunk in chunks],
            )

        frac_list, area_list = zip(*results)
        frac_vals = np.concatenate(frac_list)
        cell_areas = np.concatenate(area_list)

    return pl.DataFrame(
        {
            "latitude": lat_flat,
            "longitude": lon_flat,
            "frac_in_region": frac_vals,
            "centroid_in_region": centroid_in_region,
            "cell_area_m2": cell_areas,
        }
    )


def build_base_mask(
    sample_grib: Path,
    region_geom: MultiPolygon,
    example_var: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build a base grid mask, selecting backend according to configuration.

    The function dispatches to either the single-process optimised
    implementation or the process-pool parallel implementation based on
    the module-level ``PARALLEL_BACKEND`` constant.

    Parameters
    ----------
    sample_grib : pathlib.Path
        Path to the sample GRIB file used to define the grid.
    region_geom : shapely.geometry.MultiPolygon
        Target region geometry in EPSG:4326.
    example_var : str, optional
        GRIB ``shortName`` for grid extraction. If ``None``, a variable
        is selected automatically.

    Returns
    -------
    polars.DataFrame
        DataFrame containing the base mask with columns:
        - 'latitude' : float
        - 'longitude' : float
        - 'frac_in_region' : float
        - 'centroid_in_region' : bool
        - 'cell_area_m2' : float

    Raises
    ------
    ValueError
        If ``PARALLEL_BACKEND`` is set to an unsupported value.
    """
    if PARALLEL_BACKEND == "processpool":
        return build_base_mask_parallel(sample_grib, region_geom, example_var)
    if PARALLEL_BACKEND == "none":
        return build_base_mask_optimized(sample_grib, region_geom, example_var)
    if PARALLEL_BACKEND == "mpi":
        raise ValueError("MPI backend is not implemented in this script.")
    raise ValueError(f"Unknown PARALLEL_BACKEND: {PARALLEL_BACKEND}")


# ---------------------------------------------------------------------
# APPLY INCLUSION & EXCLUSION RULES TO BASE MASK
# ---------------------------------------------------------------------


def apply_inclusion_rules(
    base_mask: pl.DataFrame,
    mode: str,
    fraction_threshold: Optional[float],
    use_centroid: bool,
) -> pl.DataFrame:
    """
    Apply inclusion rules to a base mask.

    Parameters
    ----------
    base_mask : polars.DataFrame
        Base mask containing at least the columns ``'frac_in_region'``
        and ``'centroid_in_region'``.
    mode : {'fractional', 'centroid', 'combined'}
        Inclusion mode:
        - 'fractional' : include cells with fractional coverage above
          ``fraction_threshold``.
        - 'centroid' : include cells whose centroid lies within the
          region.
        - 'combined' : include cells that satisfy both conditions.
    fraction_threshold : float or None
        Fractional coverage threshold. Required for 'fractional' and
        'combined' modes.
    use_centroid : bool
        Whether to use centroid-based inclusion for 'centroid' or
        'combined' modes.

    Returns
    -------
    polars.DataFrame
        Filtered mask satisfying the inclusion criteria.

    Raises
    ------
    ValueError
        If the mode is unknown, or if the required threshold is not
        provided for the chosen mode.
    """
    if mode not in {"fractional", "centroid", "combined"}:
        raise ValueError(f"Unknown inclusion mode: {mode}")

    df = base_mask

    if mode in {"fractional", "combined"}:
        if fraction_threshold is None:
            raise ValueError(
                "fraction_threshold must be set for 'fractional' or 'combined' mode."
            )
        df = df.filter(pl.col("frac_in_region") >= fraction_threshold)

    if mode in {"centroid", "combined"} and use_centroid:
        df = df.filter(pl.col("centroid_in_region") == True)  # noqa: E712

    return df


def apply_exclusions(
    mask_df: pl.DataFrame,
    bboxes: List[Dict[str, float]],
    apply: bool = True,
) -> pl.DataFrame:
    """
    Apply exclusion bounding boxes to the mask (e.g. to drop islands).

    Parameters
    ----------
    mask_df : polars.DataFrame
        Input mask with columns ``'longitude'`` and ``'latitude'``.
    bboxes : list of dict
        List of bounding boxes. Each dictionary must contain:
        - 'lon_min', 'lon_max', 'lat_min', 'lat_max' (floats).
    apply : bool, optional
        If ``False``, the input DataFrame is returned unchanged.

    Returns
    -------
    polars.DataFrame
        Mask with rows removed that fall inside any of the exclusion
        bounding boxes.
    """
    if not apply or not bboxes:
        return mask_df

    lf = mask_df.lazy()
    lon = pl.col("longitude")
    lat = pl.col("latitude")

    exclusion_expr = None
    for bbox in bboxes:
        box_expr = (
            (lon >= float(bbox["lon_min"]))
            & (lon <= float(bbox["lon_max"]))
            & (lat >= float(bbox["lat_min"]))
            & (lat <= float(bbox["lat_max"]))
        )
        exclusion_expr = box_expr if exclusion_expr is None else (exclusion_expr | box_expr)

    if exclusion_expr is None:
        return mask_df

    before = mask_df.height
    result = lf.filter(~exclusion_expr).collect()
    logging.info(
        "Applied %d exclusion bounding box(es): %d -> %d rows",
        len(bboxes),
        before,
        result.height,
    )
    return result


# ---------------------------------------------------------------------
# MASK NAMING & METADATA
# ---------------------------------------------------------------------


def infer_dataset_prefix(grib_path: Path) -> str:
    """
    Infer a dataset prefix from a GRIB filename.

    Example
    -------
    ``"era5-world_N37W68S6E98_d514a3a3c256_2024_03.grib"`` → ``"era5-world"``

    Parameters
    ----------
    grib_path : pathlib.Path
        Path to the GRIB file.

    Returns
    -------
    str
        Inferred dataset prefix.
    """
    stem = grib_path.stem
    parts = stem.split("_")
    return parts[0]


def build_exclusion_hash(
    bboxes: List[Dict[str, float]],
    apply: bool,
) -> str:
    """
    Build a short hash representing the exclusion configuration.

    Parameters
    ----------
    bboxes : list of dict
        List of bounding boxes used for exclusions.
    apply : bool
        Whether exclusions are applied.

    Returns
    -------
    str
        Hash string summarising the exclusion configuration, or
        ``"noexcl"`` if no exclusions are applied.
    """
    if not apply or not bboxes:
        return "noexcl"

    key = repr(sorted(bboxes, key=lambda d: d.get("name", "")))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]


def build_mode_tag(
    mode: str,
    fraction_threshold: Optional[float],
) -> str:
    """
    Build a compact tag representing the inclusion mode and thresholds.

    Parameters
    ----------
    mode : {'fractional', 'centroid', 'combined'}
        Inclusion mode.
    fraction_threshold : float or None
        Fraction threshold for 'fractional' or 'combined' modes. Ignored
        for 'centroid'.

    Returns
    -------
    str
        Mode tag suitable for inclusion in filenames.

    Raises
    ------
    ValueError
        If an unknown mode is provided or a required threshold is
        missing.
    """
    if mode == "fractional":
        if fraction_threshold is None:
            raise ValueError("fraction_threshold must be provided for 'fractional' mode.")
        return f"fractional{fraction_threshold}"
    if mode == "centroid":
        return "centroid"
    if mode == "combined":
        if fraction_threshold is None:
            raise ValueError("fraction_threshold must be provided for 'combined' mode.")
        return f"combined{fraction_threshold}"
    raise ValueError(f"Unknown mode: {mode}")


def build_mask_filename(
    dataset_prefix: str,
    country_token: str,
    mode_tag: str,
    exclusion_hash: str,
) -> Path:
    """
    Construct the output filename for a mask.

    Parameters
    ----------
    dataset_prefix : str
        Dataset prefix (e.g. ``"era5-world"``).
    country_token : str
        Sanitised country token (e.g. ``"INDIA"``).
    mode_tag : str
        Tag representing inclusion mode and thresholds.
    exclusion_hash : str
        Short hash representing the exclusion configuration.

    Returns
    -------
    pathlib.Path
        Output filename under ``MASK_OUTPUT_DIR`` with a ``.parquet``
        suffix.
    """
    fname = f"{dataset_prefix}_{country_token}_mask_{mode_tag}_{exclusion_hash}.parquet"
    return MASK_OUTPUT_DIR / fname


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------


def main() -> None:
    """
    Run the mask-building pipeline end-to-end.

    Steps
    -----
    1. Configure logging.
    2. Find a sample GRIB file and infer the dataset prefix.
    3. Load the target region boundary.
    4. Build the base mask (grid-level fraction/centroid info).
    5. Apply inclusion rules (fractional/centroid/combined).
    6. Apply exclusion bounding boxes (optional).
    7. Handle reuse/overwrite of existing mask file.
    8. Save the final mask and metadata (JSON).
    9. Optionally generate a preview PNG via ``visualise_masks.preview_mask``.

    Returns
    -------
    None
    """
    setup_logging()
    overall_start = time.time()

    # --- 1. Find a sample GRIB file ---------------------------------
    grib_files = sorted(RAW_GRIB_DIR.glob("*.grib"))
    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found under {RAW_GRIB_DIR}")

    sample_grib = grib_files[0]
    logging.info("Using sample GRIB for grid definition: %s", sample_grib.name)

    dataset_prefix = DATASET_PREFIX or infer_dataset_prefix(sample_grib)
    logging.info("Dataset prefix: %s", dataset_prefix)

    # --- 2. Load region boundary ------------------------------------
    region_geom, country_token = load_country_geometry(
        BOUNDARY_PATH,
        COUNTRY_NAME,
        ADM_LEVEL,
    )

    # --- 3. Build base mask -----------------------------------------
    base_mask = build_base_mask(sample_grib, region_geom)

    # --- 4. Apply inclusion rules -----------------------------------
    included_mask = apply_inclusion_rules(
        base_mask,
        INCLUSION_MODE,
        FRACTION_THRESHOLD,
        USE_CENTROID if USE_CENTROID is not None else False,
    )

    # --- 5. Apply exclusions ----------------------------------------
    excluded_mask = apply_exclusions(
        included_mask,
        EXCLUSION_BBOXES,
        apply=APPLY_EXCLUSIONS,
    )

    # --- 6. Build final filename & handle existing ------------------
    mode_tag = build_mode_tag(INCLUSION_MODE, FRACTION_THRESHOLD)
    excl_hash = build_exclusion_hash(EXCLUSION_BBOXES, APPLY_EXCLUSIONS)
    mask_path = build_mask_filename(dataset_prefix, country_token, mode_tag, excl_hash)

    if mask_path.exists():
        if REUSE_EXISTING_IF_FOUND and not OVERWRITE_EXISTING:
            logging.info(
                "Mask already exists and REUSE_EXISTING_IF_FOUND=True; loading: %s",
                mask_path.name,
            )
            loaded = pl.read_parquet(mask_path)
            logging.info("Loaded existing mask with %d rows.", loaded.height)
            final_mask = loaded
        elif OVERWRITE_EXISTING:
            logging.info(
                "Mask exists but OVERWRITE_EXISTING=True; will overwrite: %s",
                mask_path.name,
            )
            final_mask = excluded_mask
        else:
            raise FileExistsError(
                f"Mask file already exists: {mask_path}. "
                f"Set REUSE_EXISTING_IF_FOUND=True or OVERWRITE_EXISTING=True."
            )
    else:
        final_mask = excluded_mask

    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mask
    final_mask.write_parquet(mask_path, compression="snappy")
    logging.info("Final mask saved to %s (rows=%d).", mask_path, final_mask.height)

    # Save metadata JSON alongside
    meta = {
        "dataset_prefix": dataset_prefix,
        "country_token": country_token,
        "boundary_path": str(BOUNDARY_PATH),
        "adm_level": ADM_LEVEL,
        "country_name": COUNTRY_NAME,
        "inclusion_mode": INCLUSION_MODE,
        "fraction_threshold": FRACTION_THRESHOLD,
        "apply_exclusions": APPLY_EXCLUSIONS,
        "exclusion_bboxes": EXCLUSION_BBOXES,
        "row_count": final_mask.height,
        "parallel_backend": PARALLEL_BACKEND,
        "max_workers": MAX_WORKERS,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = mask_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Mask metadata saved to %s", meta_path)

    # --- 8. Optional preview plot -----------------------------------
    if GENERATE_PREVIEW_PNG:
        from visualise_masks import preview_mask

        preview_png_path = mask_path.with_suffix(".png")
        preview_mask(
            final_mask,
            region_geom,
            preview_png_path,
            title=f"{dataset_prefix}_{country_token}_{INCLUSION_MODE}",
        )
        logging.info("Preview PNG saved to %s", preview_png_path.name)

    # --- 9. Summary -------------------------------------------------
    total_elapsed = time.time() - overall_start
    logging.info("===============================================")
    logging.info("Mask build complete.")
    logging.info("  Output: %s", mask_path.name)
    logging.info("  Rows in mask: %d", final_mask.height)
    logging.info(
        "  Total wall time: %.2f sec (%.2f min)",
        total_elapsed,
        total_elapsed / 60.0,
    )
    logging.info("===============================================")


if __name__ == "__main__":
    main()
