"""
01_build_country_mask.py

Build a grid mask for a target country on an ERA5-style lat/lon grid.

Features
--------
- Country-agnostic: works with any boundary file (ADM0/ADM1/etc.)
- Supports shapefile (.shp) and GeoJSON (.geojson) boundaries
- Uses ERA5 GRIB file to derive the lat/lon grid & cell polygons
- Inclusion rules:
    * fractional: frac_in_region >= FRACTION_THRESHOLD
    * centroid:  centroid of cell inside region
- Exclusions:
    * Custom lat/lon bounding boxes (e.g. islands)
    * (Hooks in place for more advanced criteria later)
- Mask naming:
    * {dataset_prefix}_{country_token}_mask_{mode_and_thresholds}_{excl_hash}.parquet
- Safe reuse:
    * If mask with same config name already exists, can reuse or overwrite
- Optional preview PNG: shows mask grid + boundary outline

Grid resolution & cost
----------------------
The script:
1. Takes one sample GRIB file (ERA5-style) from RAW_GRIB_DIR.
2. Reads 1D latitude/longitude from a single variable via cfgrib.
3. Restricts to the bounding box of the country polygon before looping
   to reduce computation time.

You can later wrap the main() into a CLI that passes in these configs.
"""

from __future__ import annotations

from pathlib import Path
import logging
import time
import re
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import polars as pl
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from pyproj import Geod

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# USER CONFIGURATION (EASY TO TURN INTO CLI OPTIONS LATER)
# ---------------------------------------------------------------------

# Directory containing ERA5 GRIB files (we use one as a grid template)
RAW_GRIB_DIR = Path("../data/raw")

# Country boundary file (can be .shp or .geojson)
# For India, this might be:
#   Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")
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

# INITIALISE TO NONE
FRACTION_THRESHOLD = None
USE_CENTROID = None

# Inclusion mode: "fractional", "centroid", or "combined"
INCLUSION_MODE = "combined"

# RESOLVE MODE → SETTINGS
if INCLUSION_MODE == "fractional":
    FRACTION_THRESHOLD = 0.90        # user-defined
    USE_CENTROID = False

elif INCLUSION_MODE == "centroid":
    FRACTION_THRESHOLD = None
    USE_CENTROID = True

elif INCLUSION_MODE == "combined":
    FRACTION_THRESHOLD = 0.90        # user-defined
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
#   - If True and file exists -> load and return it (no rebuild)
#   - If False and file exists:
#         * if OVERWRITE_EXISTING is True -> rebuild & overwrite
#         * else -> raise an error
REUSE_EXISTING_IF_FOUND = True
OVERWRITE_EXISTING = False

# If True, generate a preview PNG with grid + boundary overlay
GENERATE_PREVIEW_PNG = True

# Logging
LOG_LEVEL = logging.INFO


# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------


def setup_logging() -> None:
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# HELPER: SCAN GRIB PARAMETERS TO FIND A SAFE VARIABLE
# ---------------------------------------------------------------------


def scan_parameters_in_file(path: Path) -> List[Dict]:
    """
    Scan a GRIB file with ecCodes and return a list of dictionaries:
    [
      { "paramId": 167, "shortName": "2t" },
      ...
    ]
    """
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


def select_example_var(grib_path: Path) -> str:
    """
    Pick a safe variable present in the GRIB file for grid extraction.
    Prefers 2m temperature, otherwise chooses first shortName alphabetically.
    """
    params = scan_parameters_in_file(grib_path)
    sns = {p["shortName"] for p in params}
    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    return sorted(list(sns))[0]


# ---------------------------------------------------------------------
# LOAD GRID FROM GRIB (LAT/LON ONLY)
# ---------------------------------------------------------------------


def load_grid_from_grib(grib_path: Path, shortname: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ERA5-like 1D lat/lon coordinates from a GRIB file using cfgrib.

    Returns:
        lat_vals, lon_vals (both 1D numpy arrays)
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

    # ERA5 typically uses "latitude" and "longitude"
    if "latitude" in ds.coords and "longitude" in ds.coords:
        lat_vals = ds["latitude"].values
        lon_vals = ds["longitude"].values
    elif "lat" in ds.coords and "lon" in ds.coords:
        lat_vals = ds["lat"].values
        lon_vals = ds["lon"].values
    else:
        raise KeyError(f"Could not find lat/lon coords in {grib_path.name}; found: {list(ds.coords)}")

    ds.close()

    if lat_vals.ndim != 1 or lon_vals.ndim != 1:
        raise ValueError("Expected 1D latitude/longitude arrays (global regular grid).")

    return lat_vals, lon_vals


# ---------------------------------------------------------------------
# BOUNDARY LOADER (COUNTRY-AGNOSTIC, SHP/GEOJSON)
# ---------------------------------------------------------------------


def infer_country_field(gdf: gpd.GeoDataFrame, adm_level: Optional[int]) -> List[str]:
    """
    Suggest candidate field names to search for COUNTRY_NAME based on ADM level.
    We don't pick a single field here; instead we return candidates in priority order.
    """
    candidates_common = ["shapeName", "ADMIN", "NAME", "name", "COUNTRY", "country"]

    if adm_level == 0:
        candidates = ["shapeName", "ADM0_NAME", "NAME_EN", "NAME_0"] + candidates_common
    elif adm_level == 1:
        candidates = ["shapeName", "ADM1_NAME", "NAME_1"] + candidates_common
    else:
        candidates = candidates_common

    # Keep only those that actually exist in the dataframe
    return [c for c in candidates if c in gdf.columns]


def load_country_geometry(
    path: Path,
    country_name: Optional[str],
    adm_level: Optional[int],
) -> Tuple[MultiPolygon, str]:
    """
    Load a boundary dataset (shp/geojson) and return:
        - a MultiPolygon geometry representing the target country/region
        - a country token for naming (sanitized string)

    If country_name is None:
        assumes the file already contains only the target region.
    If country_name is provided:
        heuristically chooses a field to match based on ADM level.
    """
    logging.info("Loading boundary file: %s", path)
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)

    if country_name is None:
        # Assume entire file is one region
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

        matched_field = None
        subset = None
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

    # Union to a single MultiPolygon
    geom = unary_union(subset.geometry)
    if isinstance(geom, Polygon):
        geom = MultiPolygon([geom])
    if not isinstance(geom, MultiPolygon):
        raise TypeError("Region geometry is not a Polygon or MultiPolygon.")

    return geom, country_token


# ---------------------------------------------------------------------
# GEODESIC AREA UTILITIES
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# BUILD BASE MASK (FRACTION + CENTROID) FOR A COUNTRY
# ---------------------------------------------------------------------


def build_base_mask(
    sample_grib: Path,
    region_geom: MultiPolygon,
    example_var: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build a base grid mask with:
      - latitude
      - longitude
      - frac_in_region (0..1)
      - centroid_in_region (bool)
      - cell_area_m2

    Uses one sample GRIB to define the grid, and the region_geom to compute
    fractional overlap and centroid inclusion.

    To keep runtime reasonable, we restrict to a bounding box around the
    region plus one grid cell margin.
    """
    logging.info("Building base mask from GRIB: %s", sample_grib)

    lat_vals, lon_vals = load_grid_from_grib(sample_grib, example_var)

    if len(lat_vals) < 2 or len(lon_vals) < 2:
        raise ValueError("Latitude/longitude arrays too short to build cell polygons.")

    # Grid spacing
    dlat = abs(lat_vals[1] - lat_vals[0])
    dlon = abs(lon_vals[1] - lon_vals[0])

    minx, miny, maxx, maxy = region_geom.bounds
    logging.info("Region bounds: lon[%.2f, %.2f], lat[%.2f, %.2f]", minx, maxx, miny, maxy)

    # Restrict lat/lon to a padded bbox around the region to reduce compute
    lat_mask = (lat_vals >= (miny - dlat)) & (lat_vals <= (maxy + dlat))
    lon_mask = (lon_vals >= (minx - dlon)) & (lon_vals <= (maxx + dlon))

    lat_sub = lat_vals[lat_mask]
    lon_sub = lon_vals[lon_mask]

    logging.info(
        "Sub-grid size for mask: %d lat × %d lon = %d cells",
        len(lat_sub),
        len(lon_sub),
        len(lat_sub) * len(lon_sub),
    )

    geod = Geod(ellps="WGS84")

    rows = []
    total_cells = len(lat_sub) * len(lon_sub)
    processed = 0
    t0 = time.time()

    for lat_c in lat_sub:
        lat_min = float(lat_c - dlat / 2.0)
        lat_max = float(lat_c + dlat / 2.0)

        for lon_c in lon_sub:
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
                centroid_in = False
            else:
                inter = region_geom.intersection(cell_poly)
                if inter.is_empty:
                    frac = 0.0
                else:
                    inter_area = geodesic_area_geometry(geod, inter)
                    frac = max(0.0, min(1.0, inter_area / cell_area))

                centroid_pt = cell_poly.centroid
                centroid_in = region_geom.contains(centroid_pt)

            rows.append(
                {
                    "latitude": float(lat_c),
                    "longitude": float(lon_c),
                    "frac_in_region": float(frac),
                    "centroid_in_region": bool(centroid_in),
                    "cell_area_m2": float(cell_area),
                }
            )

            processed += 1
            if processed % 50000 == 0:
                elapsed = time.time() - t0
                logging.info(
                    "  Processed %d / %d cells (%.1f%%) in %.1f sec",
                    processed,
                    total_cells,
                    100.0 * processed / total_cells,
                    elapsed,
                )

    mask_df = pl.DataFrame(rows)
    logging.info(
        "Base mask built with %d rows (sub-grid cells). Non-zero frac cells: %d",
        mask_df.height,
        (mask_df["frac_in_region"] > 0).sum(),
    )
    return mask_df


# ---------------------------------------------------------------------
# APPLY INCLUSION & EXCLUSION RULES TO BASE MASK
# ---------------------------------------------------------------------


def apply_inclusion_rules(base_mask: pl.DataFrame, mode: str,
                          fraction_threshold: float | None, use_centroid: bool):
    lf = base_mask.lazy()

    if mode == "fractional":
        if fraction_threshold is None:
            raise ValueError("Fractional mode requires a fraction_threshold")
        lf = lf.filter(pl.col("frac_in_region") >= fraction_threshold)

    elif mode == "centroid":
        lf = lf.filter(pl.col("centroid_in_region") == True)

    elif mode == "combined":
        if fraction_threshold is None:
            raise ValueError("Combined mode requires fraction_threshold")
        lf = lf.filter(
            (pl.col("centroid_in_region") == True) &
            (pl.col("frac_in_region") >= fraction_threshold)
        )

    return lf.collect()



def apply_exclusions(
    mask_df: pl.DataFrame,
    bboxes: List[Dict[str, float]],
    apply: bool = True,
) -> pl.DataFrame:
    """
    Apply exclusion bounding boxes to the mask (e.g. to drop islands).
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
    lf = lf.filter(~exclusion_expr)
    result = lf.collect()
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
    Infer dataset prefix from a GRIB filename.
    Example: "era5-world_N37W68S6E98_d514a3a3c256_2024_03.grib"
             -> "era5-world"
    """
    stem = grib_path.stem
    parts = stem.split("_")
    if len(parts) < 1:
        raise ValueError(f"Unexpected GRIB filename pattern: {grib_path.name}")
    return parts[0]


def build_exclusion_hash(bboxes: List[Dict[str, float]], apply: bool) -> str:
    """
    Build a short hash representing the exclusion configuration.
    """
    if not apply or not bboxes:
        return "noexcl"
    # Use a stable representation
    key = repr(sorted(bboxes, key=lambda d: d.get("name", "")))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:6]


def build_mode_tag(mode: str, fraction_threshold: float) -> str:
    """
    Build a compact tag for the mode + thresholds for filenames.
    """
    if mode == "fractional":
        return f"fractional{str(fraction_threshold)}"
    if mode == "centroid":
        return "centroid"
    if mode == "combined":
        return f"combined{str(fraction_threshold)}"
    raise ValueError(f"Unknown mode: {mode}")


def build_mask_filename(
    dataset_prefix: str,
    country_token: str,
    mode_tag: str,
    exclusion_hash: str,
) -> Path:
    """
    Construct mask filename.
    """
    fname = f"{dataset_prefix}_{country_token}_mask_{mode_tag}_{exclusion_hash}.parquet"
    return MASK_OUTPUT_DIR / fname


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------


def main():
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

    # --- 3. Build base mask (or reuse, if you later cache it) -------
    # For now we always build base mask; could be cached separately.
    base_mask = build_base_mask(sample_grib, region_geom)

    # --- 4. Apply inclusion rules -----------------------------------
    included_mask = apply_inclusion_rules(
        base_mask,
        INCLUSION_MODE,
        FRACTION_THRESHOLD,
        USE_CENTROID,
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
            logging.info("Mask already exists and REUSE_EXISTING_IF_FOUND=True; loading: %s", mask_path.name)
            loaded = pl.read_parquet(mask_path)
            logging.info("Loaded existing mask with %d rows.", loaded.height)
            final_mask = loaded
        elif OVERWRITE_EXISTING:
            logging.info("Mask exists but OVERWRITE_EXISTING=True; will overwrite: %s", mask_path.name)
            final_mask = excluded_mask
        else:
            raise FileExistsError(
                f"Mask file already exists: {mask_path}. "
                f"Set REUSE_EXISTING_IF_FOUND=True or OVERWRITE_EXISTING=True."
            )
    else:
        final_mask = excluded_mask

    # --- 7. Save mask & metadata ------------------------------------
    # Ensure output dir exists
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save mask
    final_mask.write_parquet(mask_path, compression="snappy")
    logging.info("Final mask saved to %s (rows=%d).", mask_path, final_mask.height)

    # Save metadata JSON alongside (optional but handy)
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
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = mask_path.with_suffix(".json")
    import json

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Mask metadata saved to %s", meta_path)

    # --- 8. Optional preview plot -----------------------------------
    from visualise_masks import preview_mask

    if GENERATE_PREVIEW_PNG:
        preview_png_path = mask_path.with_suffix(".png")
        preview_mask(
            final_mask,
            region_geom,
            preview_png_path,
            title=f"{dataset_prefix}_{country_token}_{INCLUSION_MODE}"
        )
        logging.info("Preview PNG saved to %s", preview_png_path.name)


    # --- 9. Summary -------------------------------------------------
    total_elapsed = time.time() - overall_start
    logging.info("===============================================")
    logging.info("Mask build complete.")
    logging.info("  Output: %s", mask_path.name)
    logging.info("  Rows in mask: %d", final_mask.height)
    logging.info("  Total wall time: %.2f sec (%.2f min)",
                 total_elapsed, total_elapsed / 60.0)
    logging.info("===============================================")


if __name__ == "__main__":
    main()
