"""
02_visualise_grids_static.py

Create 2×2 static matplotlib comparison plots showing:
1. Original global ERA5 grid
2. Original grid + India boundary overlay
3. Trimmed India-only grid
4. Trimmed India-only grid + India boundary overlay
"""

import xarray as xr
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pathlib import Path

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)


# -------------------------------------------------------------------
# HARD-CODED INPUTS (EDIT AS NEEDED)
# -------------------------------------------------------------------
GRIB_FILE = Path("../data/raw/era5-world_N37W68S6E98_d514a3a3c256_2024_03.grib")
PARQUET_FILE = Path("../data/interim/era5-world_INDIA_d514a3a3c256_2024_03.parquet")
INDIA_BOUNDARY = Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")

OUTPUT_PNG = Path("grid_comparison.png")

# -------------------------------------------------------------------
# VAR HELPERS
# -------------------------------------------------------------------
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

    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    # fallback: first one alphabetically
    return sorted(list(sns))[0]


# -------------------------------------------------------------------
# SAFE LAT/LON LOADER (ERA5-compatible)
# -------------------------------------------------------------------
def get_lat_lon(ds):
    """Return 1D latitude & longitude from an ERA5 GRIB dataset."""
    if "latitude" in ds.coords and "longitude" in ds.coords:
        return ds["latitude"].values, ds["longitude"].values
    if "lat" in ds.coords and "lon" in ds.coords:
        return ds["lat"].values, ds["lon"].values
    raise KeyError(f"Could not find lat/lon coords. Found: {list(ds.coords)}")


# -------------------------------------------------------------------
# LOAD ORIGINAL GRID (full ERA5)
# -------------------------------------------------------------------

def load_original_grid(path: Path, example_var: str = "2t"):
    """
    Load only one variable from the GRIB file to extract the 1D lat/lon grid.
    Avoids cfgrib edition conflicts.
    """
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": example_var},
        }
    )

    # ERA5 global files give lat/lon as 1D coords
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # broadcast to 2D mesh
    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])

    ds.close()
    return lon2d.values, lat2d.values



# -------------------------------------------------------------------
# LOAD TRIMMED GRID (India-only Parquet)
# -------------------------------------------------------------------
def load_trimmed_grid(path: Path):
    print(f"[INFO] Loading trimmed grid from {path}")
    df = pl.read_parquet(path)
    lats = df["latitude"].to_numpy()
    lons = df["longitude"].to_numpy()
    return lons, lats


# -------------------------------------------------------------------
# PLOTTER UTIL
# -------------------------------------------------------------------
def add_india_outline(ax, india_gdf):
    """Overlay India boundary."""
    india_gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.0)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    example_var = select_example_var(GRIB_FILE)
    print(f"[INFO] Using variable '{example_var}' for grid extraction")

    lon_orig, lat_orig = load_original_grid(GRIB_FILE, example_var)
    lon_trim, lat_trim = load_trimmed_grid(PARQUET_FILE)

    india_gdf = gpd.read_file(INDIA_BOUNDARY).to_crs(epsg=4326)

    # Create 2×2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # ---- PLOT 1: Original grid -------------------------------------
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original ERA5 Grid (Global Region)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # ---- PLOT 2: Original + India Boundary -------------------------
    ax2.scatter(lon_orig, lat_orig, s=1, color="black")
    add_india_outline(ax2, india_gdf)
    ax2.set_title("Original Grid + India Boundary Overlay")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # ---- PLOT 3: Trimmed Grid (India Only) -------------------------
    ax3.scatter(lon_trim, lat_trim, s=4, color="blue")
    ax3.set_title("Trimmed ERA5 Grid (India Only)")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # ---- PLOT 4: Trimmed Grid + India Boundary ---------------------
    ax4.scatter(lon_trim, lat_trim, s=4, color="blue")
    add_india_outline(ax4, india_gdf)
    ax4.set_title("Trimmed Grid + India Boundary Overlay")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"[OK] Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
