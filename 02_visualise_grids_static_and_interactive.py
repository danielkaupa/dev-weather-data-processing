"""
02_visualise_grids_static_and_interactive.py

Creates BOTH:
1) 2×2 matplotlib static comparison PNG
2) Folium interactive HTML map

Comparisons:
- Original ERA5 global grid
- Original grid + India boundary
- Trimmed India-only grid
- Trimmed grid + India boundary
"""

import xarray as xr
import polars as pl
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from pathlib import Path
import pandas as pd
import folium

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

# -------------------------------------------------------------------
# HARD-CODED INPUTS
# -------------------------------------------------------------------
GRIB_FILE = Path("../data/raw/era5-world_N37W68S6E98_d514a3a3c256_2024_03.grib")
PARQUET_FILE = Path("../data/interim/era5-world_INDIA_d514a3a3c256_2024_03.parquet")
INDIA_BOUNDARY = Path("geoBoundariesCGAZ_ADM0/india_boundary.geojson")

OUTPUT_PNG = Path("grid_comparison.png")
OUTPUT_HTML = Path("grid_comparison.html")


# -------------------------------------------------------------------
# GRIB SCANNING HELPERS
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
                if isinstance(sn, bytes):
                    sn = sn.decode("utf-8")
                pid = codes_get(gid, "paramId")
                params[pid] = {"paramId": pid, "shortName": sn}
            finally:
                codes_release(gid)
    return list(params.values())


def select_example_var(path: Path) -> str:
    """Pick a safe ERA5 variable to load lat/lon grid from GRIB."""
    params = scan_parameters_in_file(path)
    sns = {p["shortName"] for p in params}

    if "2t" in sns:
        return "2t"
    if "t2m" in sns:
        return "t2m"
    return sorted(list(sns))[0]


# -------------------------------------------------------------------
# LOAD ORIGINAL GRID
# -------------------------------------------------------------------
def load_original_grid(path: Path, example_var: str):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"shortName": example_var},
        }
    )

    lon2d, lat2d = xr.broadcast(ds["longitude"], ds["latitude"])
    ds.close()

    df = pd.DataFrame({
        "latitude": lat2d.values.ravel(),
        "longitude": lon2d.values.ravel()
    })
    return df


# -------------------------------------------------------------------
# LOAD TRIMMED GRID
# -------------------------------------------------------------------
def load_trimmed_grid(path: Path):
    df = pl.read_parquet(path)
    return df.select(["latitude", "longitude"]).to_pandas()


# -------------------------------------------------------------------
# FOLIUM MAP
# -------------------------------------------------------------------
def make_folium_map(original_df, trimmed_df, india_gdf, output_html):
    print(f"[INFO] Creating interactive Folium map → {output_html}")

    center_lat = trimmed_df["latitude"].mean()
    center_lon = trimmed_df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)

    # Add boundary
    folium.GeoJson(
        india_gdf,
        name="India",
        style_function=lambda x: {
            "color": "red",
            "weight": 3,
            "fillOpacity": 0
        },
    ).add_to(m)

    # Original grid (light gray)
    for _, r in original_df.iterrows():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=1,
            color="lightgray",
            fill=False,
        ).add_to(m)

    # Trimmed grid (green)
    for _, r in trimmed_df.iterrows():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=2,
            color="green",
            fill=True,
            fill_opacity=1.0,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"[OK] Saved Folium map → {output_html}")


# -------------------------------------------------------------------
# MATPLOTLIB FIGURE
# -------------------------------------------------------------------
def make_static_png(lon_orig, lat_orig, lon_trim, lat_trim, india_gdf, output_png):
    print(f"[INFO] Creating static PNG → {output_png}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax1, ax2), (ax3, ax4) = axes

    # 1 — Original Grid
    ax1.scatter(lon_orig, lat_orig, s=1, color="black")
    ax1.set_title("Original ERA5 Grid")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # 2 — Original + Boundary
    ax2.scatter(lon_orig, lat_orig, s=1, color="black")
    india_gdf.plot(ax=ax2, facecolor="none", edgecolor="red", linewidth=1)
    ax2.set_title("Original Grid + India Boundary")

    # 3 — Trimmed Grid
    ax3.scatter(lon_trim, lat_trim, s=4, color="blue")
    ax3.set_title("Trimmed ERA5 Grid (India Only)")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # 4 — Trimmed + Boundary
    ax4.scatter(lon_trim, lat_trim, s=4, color="blue")
    india_gdf.plot(ax=ax4, facecolor="none", edgecolor="red", linewidth=1)
    ax4.set_title("Trimmed Grid + India Boundary")

    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    print(f"[OK] Saved static PNG → {output_png}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    example_var = select_example_var(GRIB_FILE)
    print(f"[INFO] Using GRIB variable '{example_var}' for grid extraction")

    # Load grids
    original_df = load_original_grid(GRIB_FILE, example_var)
    trimmed_df = load_trimmed_grid(PARQUET_FILE)
    india_gdf = gpd.read_file(INDIA_BOUNDARY).to_crs(epsg=4326)

    # Extract arrays for PNG
    lon_orig = original_df["longitude"]
    lat_orig = original_df["latitude"]
    lon_trim = trimmed_df["longitude"]
    lat_trim = trimmed_df["latitude"]

    # Create both outputs
    make_static_png(lon_orig, lat_orig, lon_trim, lat_trim, india_gdf, OUTPUT_PNG)
    make_folium_map(original_df, trimmed_df, india_gdf, OUTPUT_HTML)


if __name__ == "__main__":
    main()
