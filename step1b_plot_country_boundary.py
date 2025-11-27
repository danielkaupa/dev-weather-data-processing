"""
step1_plot_country_boundary.pt
================

Load a country boundary GeoJSON and produce a simple verification plot.
This script is generalisable to any boundary file, not only India.

Typical usage
-------------
    python step1_plot_country_boundary.py \
        --geojson "boundaries/India_boundary.geojson" \
        --title "India Boundary"

Optionally save a PNG:
    python step1_plot_country_boundary.py \
        --geojson "France_boundary.geojson" \
        --title "France" \
        --save "france_plot.png"

Requirements
------------
- geopandas
- matplotlib
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the script.

    Parameters
    ----------
    level : int, optional
        Logging level to use (default is ``logging.INFO``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# Core Plotting Logic
# ---------------------------------------------------------------------
def plot_boundary(geojson_path: Path, title: str, save_path: Path | None = None) -> None:
    """
    Load a GeoJSON boundary file and generate a quick verification plot.

    Parameters
    ----------
    geojson_path : Path
        Path to the GeoJSON file containing the boundary polygon(s).
    title : str
        Title for the matplotlib plot.
    save_path : Path or None, optional
        If provided, the plot will be saved to this file instead of only
        displaying it. Should include an image extension (e.g. ``.png``).

    Raises
    ------
    FileNotFoundError
        If the GeoJSON path does not exist.
    ValueError
        If the file contains no geometries.
    """
    logging.info("Loading GeoJSON boundary from '%s'.", geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    # Load GeoDataFrame
    gdf = gpd.read_file(geojson_path)

    if gdf.empty:
        raise ValueError("GeoJSON contains no features.")

    logging.info("CRS: %s", gdf.crs)
    logging.info("Number of features: %d", len(gdf))
    logging.info("Geometry types: %s", gdf.geom_type.unique().tolist())

    # Warn if CRS is not WGS84
    if gdf.crs and gdf.crs.to_string() != "EPSG:4326":
        logging.warning(
            "The boundary is not in WGS84 (EPSG:4326). Plot may appear distorted."
        )

    # -----------------------------
    # Create the plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 10))

    # Basic polygon plot
    gdf.plot(
        ax=ax,
        edgecolor="black",
        facecolor="lightblue",
        linewidth=0.8,
    )

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()

    # Save or show
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logging.info("Saved boundary plot to '%s'.", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Plot a boundary GeoJSON for quick visual verification."
    )

    parser.add_argument(
        "--geojson",
        type=Path,
        required=True,
        help="Path to the boundary GeoJSON file.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Boundary Verification Plot",
        help="Plot title (default: 'Boundary Verification Plot').",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save the plot as an image (e.g., output.png).",
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

    # Initialize logging
    setup_logging(getattr(logging, args.log.upper()))

    # Execute plot function
    plot_boundary(
        geojson_path=args.geojson,
        title=args.title,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
