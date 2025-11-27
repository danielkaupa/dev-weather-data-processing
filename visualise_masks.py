"""
02_visualise_grids_static.py

General static visualisation module.

Provides:
    preview_mask(mask_df, region_geom, out_path)
    compare_original_vs_mask(grib_file, mask_file, boundary_file, out_png)
    compare_multiple_masks(grib_file, mask_files, boundary_file, out_png)

This script can be imported by other modules (e.g. mask builder) OR run directly.
"""

import math
import xarray as xr
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

# ============================================================
# USER CONFIGURATION (No CLI yet — configure directly here)
# ============================================================

# Path to a GRIB file (used for original grid visualisation)
GRIB_FILE = Path("../data/raw/era5-world_N37W68S6E98_d514a3a3c256_2018_01.grib")

# Path to country boundary file (GeoJSON or SHP)
BOUNDARY_FILE = Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")

# List of mask parquet files to compare
MASK_FILES = [
    Path("masks/era5-world_INDIA_mask_combined0.9_264612.parquet"),
    Path("masks/era5-world_INDIA_mask_fractional0.9_264612.parquet"),
    Path("masks/era5-world_INDIA_mask_centroid_264612.parquet"),
    Path("masks/era5-world_INDIA_mask_fractional0.8_264612.parquet"),
    Path("masks/era5-world_INDIA_mask_combined0.8_264612.parquet"),
]

# Where to save the output visual
OUTPUT_PNG = Path("multi_mask_comparison.png")

# Maximum number of columns (applies to both preview and multi-mask layouts)
MAX_COLUMNS = 4

BOUNDARY_COUNTRY_NAME = "India"
BOUNDARY_ADM_LEVEL = 0


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def infer_country_field(gdf: gpd.GeoDataFrame, adm_level: int | None):
    common = ["shapeName", "ADMIN", "NAME", "name", "COUNTRY", "country"]

    if adm_level == 0:
        candidates = ["shapeName", "ADM0_NAME", "NAME_EN", "NAME_0"] + common
    elif adm_level == 1:
        candidates = ["shapeName", "ADM1_NAME", "NAME_1"] + common
    else:
        candidates = common

    return [c for c in candidates if c in gdf.columns]

def load_country_boundary(path: Path, country_name: str | None, adm_level: int | None):
    """
    Load a single country's boundary from a multipolygon boundary file.
    """
    gdf = gpd.read_file(path).to_crs(epsg=4326)

    # If only one geometry exists → assume already filtered
    if country_name is None:
        if len(gdf) > 1:
            print("[WARN] Boundary file contains multiple regions but no country_name given.")
        return gdf.union_all()

    # Try to find matching field
    candidates = infer_country_field(gdf, adm_level)
    if not candidates:
        raise KeyError(f"No usable country-name column found. Columns are: {list(gdf.columns)}")

    for field in candidates:
        if country_name in set(gdf[field]):
            subset = gdf[gdf[field] == country_name]
            if subset.empty:
                continue
            return subset.union_all()

    raise ValueError(
        f"Country '{country_name}' not found in any candidate fields {candidates} "
        f"for file {path}"
    )

def scan_parameters_in_file(path: Path):
    """Return list of {'shortName': ..., 'paramId': ...} from a GRIB file."""
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
                params[pid] = {"paramId": pid, "shortName": sn}
            finally:
                codes_release(gid)
    return list(params.values())


def select_example_var(path: Path) -> str:
    """Pick a safe variable present in the GRIB file for grid extraction."""
    params = scan_parameters_in_file(path)
    sns = {p["shortName"] for p in params}

    if "2t" in sns: return "2t"
    if "t2m" in sns: return "t2m"
    return sorted(list(sns))[0]


# -------------------------------------------------------------------
# LOADERS
# -------------------------------------------------------------------

def load_original_grid(path: Path, example_var: str = "2t"):
    """Extract 1 variable only (prevents cfgrib edition clashes)"""
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": example_var}},
    )
    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])
    ds.close()
    return lon2d.values, lat2d.values


def load_mask_grid(mask_path: Path):
    df = pl.read_parquet(mask_path)
    return df["longitude"].to_numpy(), df["latitude"].to_numpy()


# -------------------------------------------------------------------
# PLOTTERS — Simple Preview
# -------------------------------------------------------------------

def preview_mask(mask_df: pl.DataFrame, region_geom, out_path: Path,
                 title: str = "Mask Preview"):
    """
    Draw a simple scatter of mask grid points with boundary overlay.
    """
    print(f"[INFO] Saving preview PNG → {out_path}")

    gdf_region = gpd.GeoDataFrame(geometry=[region_geom], crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(10, 10))

    # boundary
    gdf_region.boundary.plot(ax=ax, color="red", linewidth=1)

    # mask points
    ax.scatter(mask_df["longitude"], mask_df["latitude"], c="blue", s=6, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved preview: {out_path}")


# -------------------------------------------------------------------
# PLOTTERS — Original vs Mask (2×2)
# -------------------------------------------------------------------

def compare_original_vs_mask(grib_file: Path, mask_file: Path,
                             boundary_file: Path, out_png: Path):

    example_var = select_example_var(grib_file)
    lon_orig, lat_orig = load_original_grid(grib_file, example_var)
    lon_mask, lat_mask = load_mask_grid(mask_file)

    boundary_geom = load_country_boundary(
        boundary_file,
        BOUNDARY_COUNTRY_NAME,
        BOUNDARY_ADM_LEVEL
    )
    boundary = gpd.GeoDataFrame(geometry=[boundary_geom], crs="EPSG:4326")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # PLOT 1: Original ERA5 Grid
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original ERA5 Grid")
    ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")

    # PLOT 2: Original + Boundary
    ax2.scatter(lon_orig, lat_orig, s=1, color="black")
    gdf_boundary.plot(ax=ax2, facecolor="none", edgecolor="red")
    ax2.set_title("Original + Boundary")

    # PLOT 3: Mask (trimmed grid)
    ax3.scatter(lon_mask, lat_mask, s=4, color="blue")
    ax3.set_title("Mask Grid")
    ax3.set_xlabel("Longitude"); ax3.set_ylabel("Latitude")

    # PLOT 4: Mask + Boundary
    ax4.scatter(lon_mask, lat_mask, s=4, color="blue")
    gdf_boundary.plot(ax=ax4, facecolor="none", edgecolor="red")
    ax4.set_title("Mask + Boundary")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[OK] Saved comparison figure → {out_png}")


# -------------------------------------------------------------------
# PLOTTERS — MULTI-MASK VISUALISATION
# -------------------------------------------------------------------

def compare_multiple_masks(
    grib_file: Path,
    mask_files: list[Path],
    boundary_file: Path,
    out_png: Path,
    max_cols: int = 4
):
    """
    Visualise up to N masks side-by-side in a 2×K grid per row pair.

    For each mask:
        Row 1: gridpoints only
        Row 2: gridpoints + boundary

    Layout adjusts automatically: max 4 columns, then makes new rows.
    """

    example_var = select_example_var(grib_file)
    lon_orig, lat_orig = load_original_grid(grib_file, example_var)

    # load boundaries once
    boundary_geom = load_country_boundary(
        boundary_file,
        BOUNDARY_COUNTRY_NAME,
        BOUNDARY_ADM_LEVEL
    )
    boundary = gpd.GeoDataFrame(geometry=[boundary_geom], crs="EPSG:4326")
    N = len(mask_files)
    if N == 0:
        raise ValueError("No mask files provided.")

    # compute number of grid columns per "panel"
    num_panels = N + 1   # +1 for "original"
    cols = min(num_panels, max_cols)
    rows = 2 * math.ceil(num_panels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.reshape(rows, cols)

    # -----------------------
    # Column 1 = Original
    # -----------------------
    # Row 0: original grid only
    ax = axes[0][0]
    ax.scatter(lon_orig, lat_orig, s=1, color="black")
    ax.set_title("Original Grid")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

    # Row 1: original + boundary
    ax = axes[1][0]
    ax.scatter(lon_orig, lat_orig, s=1, color="black")
    boundary.plot(ax=ax, facecolor="none", edgecolor="red")
    ax.set_title("Original + Boundary")

    # -----------------------
    # Columns 2..N+1 = masks
    # -----------------------
    col_index = 1
    for mask_file in mask_files:
        lon_mask, lat_mask = load_mask_grid(mask_file)

        row_base = (col_index // cols) * 2
        col_base = col_index % cols

        # Row: grid only
        ax = axes[row_base][col_base]
        ax.scatter(lon_mask, lat_mask, s=4, color="blue")
        ax.set_title(mask_file.stem + " (mask only)")

        # Row: grid + boundary
        ax = axes[row_base + 1][col_base]
        ax.scatter(lon_mask, lat_mask, s=4, color="blue")
        boundary.plot(ax=ax, facecolor="none", edgecolor="red")
        ax.set_title(mask_file.stem + " + boundary")

        col_index += 1

    # -----------------------
    # Save
    # -----------------------
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    print(f"[OK] Saved multi-mask visualisation → {out_png}")

if __name__ == "__main__":
    print("\n[INFO] Running multi-mask visualiser...\n")

    compare_multiple_masks(
        grib_file=GRIB_FILE,
        mask_files=MASK_FILES,
        boundary_file=BOUNDARY_FILE,
        out_png=OUTPUT_PNG,
        max_cols=MAX_COLUMNS
    )

    print(f"[INFO] Visual saved → {OUTPUT_PNG}\n")
