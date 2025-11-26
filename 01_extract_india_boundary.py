"""
01_extract_india_boundary.py

Extract the India polygon from the global geoBoundaries ADM0 shapefile
and save it as a standalone GeoJSON for reuse.
"""

from pathlib import Path
import logging

import geopandas as gpd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Path to global geoBoundaries ADM0 shapefile (or directory containing it)
GEBOUNDARIES_ADM0_PATH = Path("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp")


RAW_GRIB_DIR = Path("../data/raw")
GEOBOUNDARIES_ADM0_SHAPEFILE = Path("geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp")
MASK_OUTPUT = Path("india_grid_mask.parquet")
# Output directory and file for India boundary
OUTPUT_DIR = Path("geoBoundariesCGAZ_ADM0")
INDIA_GEOJSON_PATH = OUTPUT_DIR / "india_boundary.geojson"

LOG_LEVEL = logging.INFO

# Name field and value for India in the shapefile
# Adjust if your shapefile uses a different column name
COUNTRY_NAME_FIELD = "shapeName"
COUNTRY_NAME_VALUE = "India"

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------


def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------------------


def extract_india_boundary() -> None:
    logging.info("Loading global boundaries from %s", GEBOUNDARIES_ADM0_PATH)
    gdf_world = gpd.read_file(GEBOUNDARIES_ADM0_PATH)

    if COUNTRY_NAME_FIELD not in gdf_world.columns:
        raise KeyError(
            f"Expected column '{COUNTRY_NAME_FIELD}' not found in shapefile. "
            f"Available columns: {list(gdf_world.columns)}"
        )

    logging.info(
        "Filtering for %s == %s", COUNTRY_NAME_FIELD, COUNTRY_NAME_VALUE
    )
    gdf_india = gdf_world[gdf_world[COUNTRY_NAME_FIELD] == COUNTRY_NAME_VALUE]

    if gdf_india.empty:
        raise ValueError(
            f"No records found where {COUNTRY_NAME_FIELD} == {COUNTRY_NAME_VALUE}"
        )

    # Ensure WGS84 (lat/lon)
    gdf_india = gdf_india.to_crs(epsg=4326)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Saving India boundary to %s", INDIA_GEOJSON_PATH)
    gdf_india.to_file(INDIA_GEOJSON_PATH, driver="GeoJSON")

    logging.info("Done. Saved India boundary with %d feature(s).", len(gdf_india))


def main() -> None:
    setup_logging()
    extract_india_boundary()


if __name__ == "__main__":
    main()
