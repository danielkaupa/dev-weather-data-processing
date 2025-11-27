"""
step1e_visualise_masks.py
================================

Static visualisation utilities for ERA5 grid and mask comparison.

This module provides:

Functions
---------
preview_mask(mask_df, region_geom, out_path, title)
    Plot mask grid points with region boundary overlay.

compare_original_vs_mask(grib_file, mask_file, boundary_file, out_png)
    Compare an original ERA5 grid with a mask grid (2×2 figure).

compare_multiple_masks(grib_file, mask_files, boundary_file, out_png, max_cols)
    Visualise the original grid and multiple mask files in a grid layout.

Notes
-----
- This file is importable by other modules (e.g., mask builder).
- Can also be run directly to generate a default multi-mask comparison.
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from datetime import datetime
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)



def timestamp_for_filename() -> str:
    """
    Return timestamp string formatted as YYYY-MM-DD_HH-MM.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

# ======================================================================
# USER CONFIGURATION (only used when __main__ is executed)
# ======================================================================

GRIB_FILE = Path("../data/raw/era5-world_N37W68S6E98_d514a3a3c256_2025_04.grib")
BOUNDARY_FILE = Path("geoBoundariesCGAZ_ADM0/India.geojson")

MASK_FILES = [
    Path("masks/era5-world_INDIA_mask_centroid_264612.parquet"),
    Path("masks/era5-world_INDIA_BOUNDARY_mask_centroid_264612.parquet"),
    Path("masks/era5-world_INDIA_mask_combined0.8_264612.parquet"),
]
OUTPUT_DIR = Path("images")
OUTPUT_PNG = OUTPUT_DIR / Path(f"multi_mask_comparison_{timestamp_for_filename()}.png")
BOUNDARY_COUNTRY_NAME = "India"
MAX_COLUMNS = 4


# ======================================================================
# CORE HELPERS
# ======================================================================

def scan_parameters_in_file(path: Path) -> list[dict]:
    """
    Scan a GRIB file and extract unique parameter descriptors.

    Parameters
    ----------
    path : Path
        GRIB file path.

    Returns
    -------
    list of dict
        Each entry is ``{"paramId": int, "shortName": str}``.
    """
    params = {}
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

    return list(params.values())


def select_example_var(path: Path) -> str:
    """
    Select a representative shortName for grid extraction.

    Parameters
    ----------
    path : Path
        GRIB file path.

    Returns
    -------
    str
        Selected shortName (prefers ``"2t"`` or ``"t2m"``).
    """
    params = scan_parameters_in_file(path)
    sns = {p["shortName"] for p in params}

    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"

    return sorted(sns)[0]


def load_country_boundary(path: Path) -> MultiPolygon:
    """
    Load a region boundary file (already extracted for one country).

    Parameters
    ----------
    path : Path
        Path to GeoJSON/SHP containing a single region or multiple polygons
        belonging to the same region.

    Returns
    -------
    MultiPolygon
        Region geometry in EPSG:4326 CRS.
    """
    gdf = gpd.read_file(path).to_crs(epsg=4326)

    geom = unary_union(gdf.geometry)
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom

    raise TypeError("Boundary geometry must be Polygon or MultiPolygon.")


# ======================================================================
# GRID LOADERS
# ======================================================================

def load_original_grid(path: Path, example_var: str | None = None):
    """
    Load the full ERA5 grid for a single variable.

    Parameters
    ----------
    path : Path
        GRIB file path.
    example_var : str, optional
        If None, selects a representative variable.

    Returns
    -------
    lon2d : ndarray
        2D array of longitudes.
    lat2d : ndarray
        2D array of latitudes.
    """
    if example_var is None:
        example_var = select_example_var(path)

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": example_var}},
    )

    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])
    arr_lon = lon2d.values
    arr_lat = lat2d.values

    ds.close()
    return arr_lon, arr_lat


def load_mask_grid(mask_path: Path):
    """
    Load mask coordinates from a parquet mask file.

    Parameters
    ----------
    mask_path : Path

    Returns
    -------
    lon : ndarray(float)
    lat : ndarray(float)
    """
    df = pl.read_parquet(mask_path)
    return df["longitude"].to_numpy(), df["latitude"].to_numpy()


# ======================================================================
# PLOTTERS
# ======================================================================

def preview_mask(mask_df: pl.DataFrame, region_geom: MultiPolygon,
                 out_path: Path, title: str = "Mask Preview") -> None:
    """
    Plot mask grid points with region boundary overlay.

    Parameters
    ----------
    mask_df : pl.DataFrame
        Mask containing ``longitude`` and ``latitude`` columns.
    region_geom : MultiPolygon
        Region boundary.
    out_path : Path
        Output PNG path.
    title : str
        Plot title.
    """
    print(f"[INFO] Saving preview PNG → {out_path}")

    fig, ax = plt.subplots(figsize=(10, 10))
    gpd.GeoSeries([region_geom], crs="EPSG:4326").boundary.plot(ax=ax, color="red")

    ax.scatter(mask_df["longitude"], mask_df["latitude"], c="blue", s=6, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved preview: {out_path}")


def compare_original_vs_mask(
    grib_file: Path,
    mask_file: Path,
    boundary_file: Path,
    out_png: Path,
) -> None:
    """
    Produce a 2×2 comparison plot of an ERA5 grid vs. its mask version.

    Parameters
    ----------
    grib_file : Path
    mask_file : Path
    boundary_file : Path
    out_png : Path
    """
    example_var = select_example_var(grib_file)
    lon_orig, lat_orig = load_original_grid(grib_file, example_var)
    lon_mask, lat_mask = load_mask_grid(mask_file)

    region_geom = load_country_boundary(boundary_file)
    boundary = gpd.GeoDataFrame(geometry=[region_geom], crs="EPSG:4326")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Original
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original ERA5 Grid")

    ax2.scatter(lon_orig, lat_orig, s=1, color="black")
    boundary.boundary.plot(ax=ax2, color="red")
    ax2.set_title("Original + Boundary")

    # Mask
    ax3.scatter(lon_mask, lat_mask, s=4, color="blue")
    ax3.set_title("Mask Grid")

    ax4.scatter(lon_mask, lat_mask, s=4, color="blue")
    boundary.boundary.plot(ax=ax4, color="red")
    ax4.set_title("Mask + Boundary")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

    print(f"[OK] Saved comparison → {out_png}")


def compare_multiple_masks(
    grib_file: Path,
    mask_files: list[Path],
    boundary_file: Path,
    out_png: Path,
    max_cols: int = 4,
) -> None:
    """
    Visualise the original grid plus multiple mask grids in a tiled layout.

    Parameters
    ----------
    grib_file : Path
    mask_files : list of Path
    boundary_file : Path
    out_png : Path
    max_cols : int
        Maximum number of columns before wrapping.
    """
    if len(mask_files) == 0:
        raise ValueError("mask_files list cannot be empty.")

    example_var = select_example_var(grib_file)
    lon_orig, lat_orig = load_original_grid(grib_file, example_var)

    region_geom = load_country_boundary(boundary_file)
    boundary = gpd.GeoDataFrame(geometry=[region_geom], crs="EPSG:4326")

    total_panels = 1 + len(mask_files)
    cols = min(total_panels, max_cols)
    rows = 2 * math.ceil(total_panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    # ---- Panel 1: Original ----
    ax1 = axes[0][0]
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original Grid")

    ax2 = axes[1][0]
    ax2.scatter(lon_orig, lat_orig, s=1, color="black")
    boundary.boundary.plot(ax=ax2, color="red")
    ax2.set_title("Original + Boundary")

    # ---- Panels 2..N: Masks ----
    col_index = 1
    for mask_file in mask_files:
        lon_mask, lat_mask = load_mask_grid(mask_file)

        row_base = (col_index // cols) * 2
        col_base = col_index % cols

        ax = axes[row_base][col_base]
        ax.scatter(lon_mask, lat_mask, s=4, color="blue")
        ax.set_title(f"{mask_file.stem} (mask only)")

        ax = axes[row_base + 1][col_base]
        ax.scatter(lon_mask, lat_mask, s=4, color="blue")
        boundary.boundary.plot(ax=ax, color="red")
        ax.set_title(f"{mask_file.stem} + boundary")

        col_index += 1

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    print(f"[OK] Saved multi-mask visualisation → {out_png}")


# ======================================================================
# MAIN EXECUTION (only when run directly)
# ======================================================================

if __name__ == "__main__":
    print("\n[INFO] Running multi-mask visualiser...\n")

    compare_multiple_masks(
        grib_file=GRIB_FILE,
        mask_files=MASK_FILES,
        boundary_file=BOUNDARY_FILE,
        out_png=OUTPUT_PNG,
        max_cols=MAX_COLUMNS,
    )

    print(f"[INFO] Visual saved → {OUTPUT_PNG}\n")
