"""
step1_extract_country_boundary.py
===================

Extract a country boundary polygon from a global geoBoundaries ADM0 shapefile
and save it as a standalone GeoJSON file. The script is fully generalised, so
the user may specify *any* country name and any attribute field used for
country names in a given shapefile.

Typical usage
-------------
    python step1_extract_country_boundary.py \
        --country "India" \
        --field "shapeName" \
        --shapefile "geoBoundariesCGAZ_ADM0/geoBoundariesCGAZ_ADM0.shp" \
        --outdir "geojson_boundaries"

Requirements
------------
- geopandas
- Python 3.8+
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the script.

    Parameters
    ----------
    level : int, optional
        Logging level to use. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# Core Logic
# ---------------------------------------------------------------------
def extract_boundary(
    shapefile_path: Path,
    country_name: str,
    name_field: str,
    output_dir: Path,
) -> Path:
    """
    Extract a country's boundary from a global boundaries shapefile and
    save it as a GeoJSON file.

    Parameters
    ----------
    shapefile_path : Path
        Path to the global ADM0 shapefile or directory containing it.
    country_name : str
        Value in ``name_field`` identifying the country (e.g., ``"India"``).
    name_field : str
        Column in the shapefile storing country names (e.g., ``"shapeName"``).
    output_dir : Path
        Directory where the output GeoJSON will be saved.

    Returns
    -------
    Path
        The path to the saved GeoJSON file.

    Raises
    ------
    FileNotFoundError
        If the shapefile cannot be found.
    KeyError
        If ``name_field`` does not exist in the shapefile.
    ValueError
        If no matching country is found.
    """
    logging.info("Loading global boundaries from '%s'.", shapefile_path)

    if not shapefile_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

    # Load the boundaries into a GeoDataFrame
    gdf_world = gpd.read_file(shapefile_path)

    # Ensure the expected name field exists
    if name_field not in gdf_world.columns:
        raise KeyError(
            f"Expected column '{name_field}' not found. "
            f"Available columns: {list(gdf_world.columns)}"
        )

    logging.info("Filtering for records where '%s' == '%s'.", name_field, country_name)
    gdf_country = gdf_world[gdf_world[name_field] == country_name]

    if gdf_country.empty:
        raise ValueError(
            f"No records found for country_name='{country_name}' "
            f"using name_field='{name_field}'."
        )

    # Standardise projection to WGS84 (lat/lon)
    gdf_country = gdf_country.to_crs(epsg=4326)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct output file path
    safe_country = country_name.replace(" ", "_")
    output_path = output_dir / f"{safe_country}.geojson"

    logging.info("Saving boundary to '%s'.", output_path)
    gdf_country.to_file(output_path, driver="GeoJSON")

    logging.info("Success: Saved boundary with %d feature(s).", len(gdf_country))
    return output_path


# ---------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Create a command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Extract a country boundary from a global ADM0 shapefile."
    )

    parser.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Path to the global ADM0 shapefile.",
    )
    parser.add_argument(
        "--country",
        type=str,
        required=True,
        help="Country name to extract (as it appears in the chosen name field).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="shapeName",
        help="Column name storing country names (default: 'shapeName').",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("boundaries"),
        help="Output directory for the extracted GeoJSON (default: './boundaries').",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    return parser


def main() -> None:
    """Entry point for command-line execution."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(getattr(logging, args.log.upper()))

    # Execute extraction
    extract_boundary(
        shapefile_path=args.shapefile,
        country_name=args.country,
        name_field=args.field,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
