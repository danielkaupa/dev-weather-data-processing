"""
step1d_build_country_mask.py
============================

Build a region mask on an ERA5-style latitude–longitude grid.

This script:

1. Loads a *pre-extracted* boundary file (GeoJSON/SHP) representing one
   country or region. No internal filtering on attribute columns is
   performed; the entire geometry is treated as the target region.
2. Loads a sample GRIB file to derive the ERA5 grid (1D latitude and
   longitude arrays).
3. Constructs grid-cell polygons and computes:
   - fractional overlap with the region (in an equal-area CRS),
   - whether the cell centroid lies within the region.
4. Applies inclusion rules:
   - ``fractional`` : based on fractional coverage,
   - ``centroid``   : based on centroid inclusion only,
   - ``combined``   : requires both criteria.
5. Applies optional exclusion bounding boxes (e.g., to remove islands).
   Exclusions can be supplied via a JSON file or via hardcoded defaults.
6. Produces:
   - final mask as a Parquet file,
   - mask metadata as a JSON file,
   - optional preview PNG of the mask and boundary.

The script is **region-agnostic**: the region can be a national
boundary, a state/province, a watershed, or any polygon shape.

Example
-------
Run with defaults and a pre-extracted boundary file::

    python step1d_build_country_mask.py \
        --grib-dir ../data/raw \
        --boundary-file boundaries/india.geojson

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import polars as pl
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from eccodes import (
    CodesInternalError,
    codes_get,
    codes_grib_new_from_file,
    codes_release,
)

# ============================================================================
# DEFAULTS (override via CLI if desired)
# ============================================================================

# Inclusion mode and threshold
# "fractional"  : include cells with fraction >= threshold
# "centroid"    : include cells whose centroid lies in region
# "combined"    : require both conditions
DEFAULT_INCLUSION_MODE = "combined"  # "fractional" | "centroid" | "combined"
DEFAULT_FRACTION_THRESHOLD = 0.90

# Reuse / overwrite existing masks
DEFAULT_REUSE_EXISTING = False
DEFAULT_OVERWRITE_EXISTING = True

# Generate image of mask + boundary
DEFAULT_GENERATE_IMAGE = False

DEFAULT_PARALLEL_BACKEND = "processpool"  # "none" | "processpool"
DEFAULT_MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)
DEFAULT_PARALLEL_MIN_CELLS = 10_000
DEFAULT_PARALLEL_CHUNK_SIZE = 5_000

# Equal-area CRS for area and fraction computation
EA_CRS = "EPSG:6933"

# --------------------------------------------------------------------------
# Optional hardcoded exclusion bounding boxes
#
# These are used only if no JSON file is provided via --exclusion-bbox-json.
# Each dict must contain:
#     "lon_min", "lon_max", "lat_min", "lat_max"
# A "name" field may also be provided for readability; it is only used
# when computing the exclusion hash and does not affect masking logic.
# --------------------------------------------------------------------------
HARDCODED_EXCLUSION_BBOXES: List[Dict[str, float]] = [
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


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(level: str = "INFO") -> None:
    """
    Configure global logging for the script.

    Parameters
    ----------
    level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, optional
        Logging verbosity level. Defaults to ``"INFO"``.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ============================================================================
# GRIB GRID LOADING
# ============================================================================

def scan_parameters_in_file(path: Path) -> List[Dict]:
    """
    Scan a GRIB file and return unique parameter entries.

    Parameters
    ----------
    path : pathlib.Path
        Path to the GRIB file.

    Returns
    -------
    list of dict
        List of parameter dictionaries, each containing:
        - ``"paramId"`` : int
        - ``"shortName"`` : str
    """
    params: Dict[int, Dict[str, str]] = {}

    with open(path, "rb") as f:
        while True:
            try:
                gid = codes_grib_new_from_file(f)
            except CodesInternalError:
                break
            if not gid:
                break

            try:
                sn = codes_get(gid, "shortName")
                if isinstance(sn, bytes):
                    sn = sn.decode("utf-8")
                pid = int(codes_get(gid, "paramId"))
                params[pid] = {"paramId": pid, "shortName": str(sn)}
            finally:
                codes_release(gid)

    return list(params.values())


def select_example_var(grib_path: Path) -> str:
    """
    Choose a representative GRIB shortName for grid extraction.

    Preference order:
    1. ``"2t"``
    2. ``"t2m"``
    3. First shortName in alphabetical order.

    Parameters
    ----------
    grib_path : pathlib.Path
        Path to the GRIB file.

    Returns
    -------
    str
        Selected shortName.
    """
    params = scan_parameters_in_file(grib_path)
    sns = {p["shortName"] for p in params}
    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    return sorted(sns)[0]


def load_grid_from_grib(
    grib: Path,
    shortname: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ERA5-style 1D latitude and longitude arrays from a GRIB file.

    Parameters
    ----------
    grib : pathlib.Path
        GRIB file path.
    shortname : str, optional
        GRIB shortName used when reading the dataset. If ``None``,
        a shortName is selected automatically via
        :func:`select_example_var`.

    Returns
    -------
    lat_vals : numpy.ndarray
        1D array of latitude values.
    lon_vals : numpy.ndarray
        1D array of longitude values.

    Raises
    ------
    KeyError
        If latitude/longitude coordinates cannot be found.
    ValueError
        If the latitude or longitude arrays are not 1D.
    """
    if shortname is None:
        shortname = select_example_var(grib)
        logging.info("Selected variable '%s' for grid extraction", shortname)

    ds = xr.open_dataset(
        grib,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": shortname}},
    )

    if "latitude" in ds and "longitude" in ds:
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
    elif "lat" in ds and "lon" in ds:
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
    else:
        coords = list(ds.coords)
        ds.close()
        raise KeyError(
            f"Could not find lat/lon coords in {grib.name}; "
            f"available coordinates: {coords}"
        )

    ds.close()

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("Expected 1D lat/lon arrays (regular grid).")

    return lat_vals, lon_vals


# ============================================================================
# BOUNDARY LOADING
# ============================================================================

def load_boundary_geometry(path: Path) -> Tuple[MultiPolygon, str]:
    """
    Load a boundary (GeoJSON/SHP) and return a MultiPolygon region.

    Parameters
    ----------
    path : pathlib.Path
        Path to the boundary file.

    Returns
    -------
    region_geom : shapely.geometry.MultiPolygon
        Unified geometry representing the region, reprojected to WGS84
        (EPSG:4326).
    region_token : str
        Uppercase token derived from the file stem, suitable for use in
        filenames (spaces replaced by underscores).

    Raises
    ------
    TypeError
        If the unified geometry is neither a Polygon nor a MultiPolygon.
    """
    logging.info("Loading boundary: %s", path)
    gdf = gpd.read_file(path).to_crs(epsg=4326)

    geom = unary_union(gdf.geometry)
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])

    if not isinstance(geom, MultiPolygon):
        raise TypeError("Boundary must be a Polygon or MultiPolygon geometry.")

    token = path.stem.upper().replace(" ", "_")
    return geom, token


# ============================================================================
# GRID PREPARATION
# ============================================================================

def prepare_grid_cells(
    grib_path: Path,
    region_geom: MultiPolygon,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare grid cells (polygons + centroids) over the region bounding box.

    The function:
    1. Loads 1D latitude and longitude arrays from the GRIB file.
    2. Restricts them to a slightly expanded bounding box around the
       region geometry to reduce unnecessary computation.
    3. Constructs rectangular polygons representing grid cells.
    4. Determines whether each cell centroid lies inside the region.

    Parameters
    ----------
    grib_path : pathlib.Path
        Path to the GRIB file used to define the grid.
    region_geom : shapely.geometry.MultiPolygon
        Region geometry in EPSG:4326.

    Returns
    -------
    lat_flat : numpy.ndarray
        Flattened latitude array of grid cell centers.
    lon_flat : numpy.ndarray
        Flattened longitude array of grid cell centers.
    cell_polys : numpy.ndarray of shapely.Polygon
        Array of grid cell polygons in EPSG:4326.
    centroid_in_region : numpy.ndarray of bool
        Boolean mask indicating centroid inclusion within the region.
    """
    lat_vals, lon_vals = load_grid_from_grib(grib_path)
    dlat = abs(lat_vals[1] - lat_vals[0])
    dlon = abs(lon_vals[1] - lon_vals[0])

    minx, miny, maxx, maxy = region_geom.bounds

    lat_mask = (lat_vals >= miny - dlat) & (lat_vals <= maxy + dlat)
    lon_mask = (lon_vals >= minx - dlon) & (lon_vals <= maxx + dlon)

    lat_sub = lat_vals[lat_mask]
    lon_sub = lon_vals[lon_mask]

    lat2d = np.broadcast_to(lat_sub[:, None], (len(lat_sub), len(lon_sub)))
    lon2d = np.broadcast_to(lon_sub, (len(lat_sub), len(lon_sub)))

    lat_flat = lat2d.ravel()
    lon_flat = lon2d.ravel()

    half_dlat = dlat / 2.0
    half_dlon = dlon / 2.0

    cell_polys = shapely.box(
        lon_flat - half_dlon,
        lat_flat - half_dlat,
        lon_flat + half_dlon,
        lat_flat + half_dlat,
        ccw=True,
    )

    centroids = shapely.points(lon_flat, lat_flat)
    centroid_in_region = shapely.within(centroids, region_geom)

    return lat_flat, lon_flat, cell_polys, centroid_in_region


def compute_fraction_area(
    cell_polys: np.ndarray,
    region_geom: MultiPolygon,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fractional coverage and area for grid cells.

    The computation is performed in an equal-area projection to ensure
    consistent area calculations.

    Parameters
    ----------
    cell_polys : numpy.ndarray of shapely.Polygon
        Array of grid cell polygons in EPSG:4326.
    region_geom : shapely.geometry.MultiPolygon
        Region geometry in EPSG:4326.

    Returns
    -------
    frac_vals : numpy.ndarray
        Fractional coverage of each cell area that lies inside the
        region, clipped to [0, 1].
    cell_areas : numpy.ndarray
        Area of each cell in square metres.
    """
    gdf_cells = gpd.GeoSeries(cell_polys, crs="EPSG:4326").to_crs(EA_CRS)
    region_ea = gpd.GeoSeries([region_geom], crs="EPSG:4326").to_crs(EA_CRS).iloc[0]

    cells_ea = gdf_cells.geometry.values
    cell_areas = shapely.area(cells_ea)

    intersections = shapely.intersection(cells_ea, region_ea)
    inter_areas = shapely.area(intersections)

    frac_vals = np.zeros_like(cell_areas)
    mask = cell_areas > 0
    frac_vals[mask] = (inter_areas[mask] / cell_areas[mask]).clip(0, 1)

    return frac_vals, cell_areas


# ============================================================================
# INCLUSION / EXCLUSION
# ============================================================================

def apply_inclusion(
    df: pl.DataFrame,
    mode: str,
    threshold: float,
    use_centroid: bool,
) -> pl.DataFrame:
    """
    Apply inclusion rules to a base mask.

    Parameters
    ----------
    df : polars.DataFrame
        Input DataFrame containing at least the columns
        ``"frac_in_region"`` and ``"centroid_in_region"``.
    mode : {"fractional", "centroid", "combined"}
        Inclusion mode:
        - ``"fractional"`` : require ``frac_in_region >= threshold``.
        - ``"centroid"``   : require centroid to lie in region.
        - ``"combined"``   : require both conditions.
    threshold : float
        Fractional coverage threshold used for ``"fractional"`` and
        ``"combined"`` modes.
    use_centroid : bool
        Whether to use centroid inclusion (relevant for
        ``"centroid"`` or ``"combined"`` modes).

    Returns
    -------
    polars.DataFrame
        Filtered DataFrame satisfying the inclusion criteria.

    Raises
    ------
    ValueError
        If an unknown inclusion mode is provided.
    """
    if mode not in {"fractional", "centroid", "combined"}:
        raise ValueError(f"Unknown inclusion mode: {mode}")

    out = df

    if mode in {"fractional", "combined"}:
        out = out.filter(pl.col("frac_in_region") >= threshold)

    if mode in {"centroid", "combined"} and use_centroid:
        out = out.filter(pl.col("centroid_in_region") == True)  # noqa: E712

    return out


def apply_exclusions(
    df: pl.DataFrame,
    bboxes: List[Dict[str, float]],
) -> pl.DataFrame:
    """
    Apply exclusion bounding boxes to the mask.

    Cells whose (longitude, latitude) lie inside any specified bounding
    box are removed.

    Parameters
    ----------
    df : polars.DataFrame
        Input mask DataFrame with ``"longitude"`` and ``"latitude"``
        columns.
    bboxes : list of dict
        List of bounding box dictionaries. Each dict must contain:
        - ``"lon_min"``, ``"lon_max"``, ``"lat_min"``, ``"lat_max"``.
        An optional ``"name"`` field is allowed for readability, but it
        is ignored in the masking logic.

    Returns
    -------
    polars.DataFrame
        Mask DataFrame with rows removed for which the cell center lies
        inside any exclusion bounding box. If ``bboxes`` is empty, the
        input DataFrame is returned unchanged.
    """
    if not bboxes:
        return df

    lf = df.lazy()
    lon = pl.col("longitude")
    lat = pl.col("latitude")

    expr = None
    for b in bboxes:
        box_expr = (
            (lon >= float(b["lon_min"]))
            & (lon <= float(b["lon_max"]))
            & (lat >= float(b["lat_min"]))
            & (lat <= float(b["lat_max"]))
        )
        expr = box_expr if expr is None else (expr | box_expr)

    before = df.height
    out = lf.filter(~expr).collect()

    logging.info(
        "Applied %d exclusion bounding box(es): %d → %d rows",
        len(bboxes),
        before,
        out.height,
    )

    return out


# ============================================================================
# FILENAME HELPERS
# ============================================================================

def infer_dataset_prefix(grib_file: Path) -> str:
    """
    Infer the dataset prefix from a GRIB filename.

    The expected pattern is roughly::

        <dataset>_<coords>_<uid>_<year>_<month>.grib

    In this case, the dataset prefix is taken as the first component.

    Parameters
    ----------
    grib_file : pathlib.Path
        GRIB file path.

    Returns
    -------
    str
        Dataset prefix (e.g. ``"era5-world"``).
    """
    return grib_file.stem.split("_")[0]


def build_exclusion_hash(bboxes: List[Dict[str, float]]) -> str:
    """
    Build a short hash representing the exclusion configuration.

    Parameters
    ----------
    bboxes : list of dict
        Exclusion bounding boxes. May be empty.

    Returns
    -------
    str
        A 6-character hexadecimal hash string summarising the exclusion
        configuration. If no bounding boxes are provided, returns
        ``"noexcl"``.
    """
    if not bboxes:
        return "noexcl"

    key = repr(sorted(bboxes, key=lambda d: d.get("name", "")))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]


def build_mode_tag(mode: str, threshold: float) -> str:
    """
    Build a compact mode tag for inclusion mode and threshold.

    Parameters
    ----------
    mode : {"fractional", "centroid", "combined"}
        Inclusion mode.
    threshold : float
        Fractional coverage threshold (used for labeling).

    Returns
    -------
    str
        Mode tag string suitable for filenames.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported values.
    """
    if mode == "fractional":
        return f"fractional{threshold}"
    if mode == "centroid":
        return "centroid"
    if mode == "combined":
        return f"combined{threshold}"
    raise ValueError(f"Invalid inclusion mode: {mode}")


# ============================================================================
# CLI
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description="Build a region mask on an ERA5-style regular grid."
    )

    # Core inputs
    p.add_argument("--grib-dir", type=Path, required=True,
                   help="Directory containing GRIB files (one is used as grid template).")
    p.add_argument("--boundary-file", type=Path, required=True,
                   help="Path to region boundary file (GeoJSON or SHP).")

    # Output directories
    p.add_argument("--mask-dir", type=Path, default=Path("masks"),
                   help="Directory for saving mask Parquet files.")
    p.add_argument("--metadata-dir", type=Path, default=Path("metadata"),
                   help="Directory for saving mask metadata JSON files.")
    p.add_argument("--image-dir", type=Path, default=Path("images"),
                   help="Directory for saving preview PNGs.")

    # Dataset naming
    p.add_argument("--dataset-prefix", type=str, default=None,
                   help="Dataset prefix for filenames (if omitted, inferred from GRIB).")

    # Inclusion rules
    p.add_argument("--inclusion-mode", type=str,
                   default=DEFAULT_INCLUSION_MODE,
                   choices=["fractional", "centroid", "combined"],
                   help="Inclusion mode for mask generation.")
    p.add_argument("--fraction-threshold", type=float,
                   default=DEFAULT_FRACTION_THRESHOLD,
                   help="Fractional coverage threshold for inclusion.")

    # Exclusions
    p.add_argument(
        "--exclusion-bbox-json",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file containing exclusion bounding boxes. "
            "If provided, overrides the hardcoded exclusions."
        ),
    )

    # Reuse / overwrite
    p.add_argument("--reuse-existing", action="store_true",
                   default=DEFAULT_REUSE_EXISTING,
                   help="Reuse existing mask if it already exists.")
    p.add_argument("--overwrite-existing", action="store_true",
                   default=DEFAULT_OVERWRITE_EXISTING,
                   help="Overwrite existing mask if it already exists.")

    # Preview image
    p.add_argument("--generate-image", action="store_true",
                   default=DEFAULT_GENERATE_IMAGE,
                   help="Generate a PNG preview of the mask and boundary.")

    # Parallelisation
    p.add_argument("--parallel-backend", type=str,
                   default=DEFAULT_PARALLEL_BACKEND,
                   choices=["none", "processpool"],
                   help="Backend for area/fraction computations.")
    p.add_argument("--parallel-min-cells", type=int,
                   default=DEFAULT_PARALLEL_MIN_CELLS,
                   help="Minimum grid cells to trigger parallel processing.")
    p.add_argument("--parallel-chunk-size", type=int,
                   default=DEFAULT_PARALLEL_CHUNK_SIZE,
                   help="Number of cells per chunk for parallel processing.")

    # Logging level
    p.add_argument("--log", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level.")

    return p


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """
    Run the mask-building pipeline end-to-end.

    Steps
    -----
    1. Parse CLI arguments and configure logging.
    2. Locate a sample GRIB file and infer dataset prefix.
    3. Load the region boundary and unify geometries.
    4. Prepare grid cells and centroids within the region bounding box.
    5. Compute cell areas and fractional coverage within the region.
    6. Apply inclusion rules (fractional/centroid/combined).
    7. Load exclusion bounding boxes (JSON overrides hardcoded) and
       apply them to the mask.
    8. Construct output filenames and handle reuse/overwrite logic.
    9. Save the final mask (Parquet) and metadata (JSON).
    10. Optionally generate a preview PNG.

    Returns
    -------
    None
    """
    args = build_arg_parser().parse_args()
    setup_logging(args.log)

    # ---------------------------------------------------------------
    # 1. Find sample GRIB file
    # ---------------------------------------------------------------
    grib_files = sorted(args.grib_dir.glob("*.grib"))
    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found in {args.grib_dir}")

    sample_grib = grib_files[0]
    logging.info("Using sample GRIB file: %s", sample_grib.name)

    dataset_prefix = args.dataset_prefix or infer_dataset_prefix(sample_grib)
    logging.info("Dataset prefix: %s", dataset_prefix)

    # ---------------------------------------------------------------
    # 2. Load boundary
    # ---------------------------------------------------------------
    region_geom, region_token = load_boundary_geometry(args.boundary_file)

    # ---------------------------------------------------------------
    # 3. Prepare grid cells
    # ---------------------------------------------------------------
    lat_flat, lon_flat, cell_polys, centroid_in_region = prepare_grid_cells(
        sample_grib,
        region_geom,
    )
    n_cells = len(cell_polys)
    logging.info("Prepared %d candidate grid cells.", n_cells)

    # ---------------------------------------------------------------
    # 4. Fraction + area computation
    # ---------------------------------------------------------------
    if args.parallel_backend == "none" or n_cells < args.parallel_min_cells:
        logging.info("Using single-process area computation.")
        frac_vals, cell_areas = compute_fraction_area(cell_polys, region_geom)
    else:
        logging.info(
            "Using process pool (%d workers), chunk size = %d.",
            DEFAULT_MAX_WORKERS,
            args.parallel_chunk_size,
        )
        chunks: List[np.ndarray] = [
            cell_polys[i:i + args.parallel_chunk_size]
            for i in range(0, n_cells, args.parallel_chunk_size)
        ]

        with multiprocessing.Pool(processes=DEFAULT_MAX_WORKERS) as pool:
            results = pool.starmap(
                compute_fraction_area,
                [(chunk, region_geom) for chunk in chunks],
            )

        frac_vals = np.concatenate([f for f, _ in results])
        cell_areas = np.concatenate([a for _, a in results])

    base_df = pl.DataFrame(
        {
            "latitude": lat_flat,
            "longitude": lon_flat,
            "frac_in_region": frac_vals,
            "centroid_in_region": centroid_in_region,
            "cell_area_m2": cell_areas,
        }
    )

    # ---------------------------------------------------------------
    # 5. Apply inclusion rules
    # ---------------------------------------------------------------
    included = apply_inclusion(
        base_df,
        mode=args.inclusion_mode,
        threshold=args.fraction_threshold,
        use_centroid=(args.inclusion_mode in {"centroid", "combined"}),
    )

    # ---------------------------------------------------------------
    # 6. Determine exclusions (JSON overrides hardcoded)
    # ---------------------------------------------------------------
    if args.exclusion_bbox_json is not None:
        logging.info("Loading exclusion bounding boxes from JSON: %s", args.exclusion_bbox_json)
        excl_bboxes = json.loads(args.exclusion_bbox_json.read_text())
    else:
        excl_bboxes = HARDCODED_EXCLUSION_BBOXES
        if excl_bboxes:
            logging.info(
                "Using %d hardcoded exclusion bounding box(es).",
                len(excl_bboxes),
            )
        else:
            logging.info("No exclusion bounding boxes provided.")

    final_mask = apply_exclusions(included, excl_bboxes)

    # ---------------------------------------------------------------
    # 7. Build filenames
    # ---------------------------------------------------------------
    mode_tag = build_mode_tag(args.inclusion_mode, args.fraction_threshold)
    excl_hash = build_exclusion_hash(excl_bboxes)

    args.mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = (
        args.mask_dir /
        f"{dataset_prefix}_{region_token}_mask_{mode_tag}_{excl_hash}.parquet"
    )

    # ---------------------------------------------------------------
    # 8. Reuse / overwrite logic
    # ---------------------------------------------------------------
    if mask_path.exists():
        if args.reuse_existing and not args.overwrite_existing:
            logging.info("Reusing existing mask: %s", mask_path)
            final_mask = pl.read_parquet(mask_path)
        elif args.overwrite_existing:
            logging.info("Overwriting existing mask: %s", mask_path)
        else:
            raise FileExistsError(
                f"Mask already exists: {mask_path}. "
                f"Use --overwrite-existing or --reuse-existing."
            )

    final_mask.write_parquet(mask_path, compression="snappy")
    logging.info("Saved mask → %s", mask_path)

    # ---------------------------------------------------------------
    # 9. Save metadata
    # ---------------------------------------------------------------
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset_prefix": dataset_prefix,
        "region_token": region_token,
        "boundary_file": str(args.boundary_file),
        "inclusion_mode": args.inclusion_mode,
        "fraction_threshold": args.fraction_threshold,
        "exclusion_bboxes": excl_bboxes,
        "row_count": final_mask.height,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_filename = (
        f"{dataset_prefix}_{region_token}_mask_{mode_tag}_{excl_hash}.json"
    )
    meta_path = args.metadata_dir / meta_filename
    meta_path.write_text(json.dumps(meta, indent=2))
    logging.info("Saved metadata → %s", meta_path)

    # ---------------------------------------------------------------
    # 10. Optional preview image
    # ---------------------------------------------------------------
    if args.generate_image:
        from step1e_visualise_masks import preview_mask

        args.image_dir.mkdir(parents=True, exist_ok=True)
        png_filename = (
            f"{dataset_prefix}_{region_token}_mask_{mode_tag}_{excl_hash}.png"
        )
        png_path = args.image_dir / png_filename

        preview_mask(final_mask, region_geom, png_path)
        logging.info(f"Saved image of mask for {region_token} as PNG to [{png_path}]")

    logging.info("Mask build complete. Rows in final mask: %d", final_mask.height)


if __name__ == "__main__":
    main()
