#!/usr/bin/env python3
"""
step2b_data_profile.py
======================

Lazy/streaming profiler for Parquet/CSV files using Polars `LazyFrame`.

Features
--------
- Uses `pl.scan_parquet` / `pl.scan_csv` (no full in-memory load).
- Column-wise statistics computed via streaming aggregations.
- Random sample values per column (3 non-null values).
- Console table summary (always printed).
- Optional text and/or CSV report on disk.

Defaults
--------
- Input file can be set in the script (`INPUT_FILE`) but is usually
  provided via CLI.
- Output directory defaults to the directory of the input file.
- Report filename base defaults to: ``profile_report_{input_stem}``.
- Report format defaults to "both" (text + CSV).

Example
-------
Run with explicit file:

    python step2b_data_profile.py ../data/interim/era5-world_INDIA_2025_06.parquet

Specify output directory and only CSV:

    python step2b_data_profile.py ../data/interim/era5-world_INDIA_2025_06.parquet \\
        --out-dir reports \\
        --report-format csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


# ======================================================================
# User defaults (used if CLI arguments not provided)
# ======================================================================

# Example default input file (you can change this or just use CLI)
INPUT_FILE = Path("../data/interim/era5-world_INDIA_d514a3a3c256_2025_06.parquet")

# Default output directory:
# - If None, will use the directory of the input file.
OUTPUT_DIR: Optional[Path] = None

# Format of reports: "txt", "csv", or "both"
DEFAULT_REPORT_FORMAT = "both"

# Logging level (string)
DEFAULT_LOG_LEVEL = "INFO"

# Optional row limit (None = full dataset, lazily)
# If set (e.g. 100_000), the profiler will only consider the first N rows.
DEFAULT_MAX_ROWS: Optional[int] = None


# ======================================================================
# Logging
# ======================================================================

def setup_logging(level: str) -> None:
    """
    Configure CLI logging.

    Parameters
    ----------
    level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        Console log level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ======================================================================
# Column summaries (lazy)
# ======================================================================

def lazy_random_sample(lf: pl.LazyFrame, col: str, n: int = 3) -> List[Any]:
    """
    Return up to ``n`` random non-null sample values from a column.

    This uses lazy/streaming operations:

    1. Adds a random column.
    2. Filters out nulls.
    3. Sorts by the random column.
    4. Takes the first ``n`` rows and collects.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame pointing to the underlying dataset.
    col : str
        Column name to sample from.
    n : int, optional
        Number of random samples to return (default is 3).

    Returns
    -------
    list
        A list of up to ``n`` random non-null values. If sampling fails,
        returns ``["<error sampling>"]``.
    """
    try:
        lf_sample = (
            lf.select([
                pl.col(col).alias(col),
                pl.random().alias("_rand"),
            ])
            .filter(pl.col(col).is_not_null())
            .sort("_rand")
            .select(pl.col(col))
            .head(n)
        )
        df = lf_sample.collect(streaming=True)
        return df[col].to_list()
    except Exception:  # noqa: BLE001
        return ["<error sampling>"]


def summarize_column_lazy(
    lf: pl.LazyFrame,
    col: str,
    dtype: pl.DataType,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a per-column summary using lazy streaming aggregations.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame representing the dataset (possibly already head-limited).
    col : str
        Column name to summarize.
    dtype : polars.DataType
        Column data type from the schema.
    max_rows : int or None, optional
        If not ``None``, indicates that only the first ``max_rows`` rows
        of the dataset were considered. Used to interpret percentages and
        approximate memory usage.

    Returns
    -------
    dict
        Dictionary with summary information:

        - ``column`` : str
        - ``dtype`` : str
        - ``memory_bytes_approx`` : int
        - ``null_pct`` : float
        - ``unique_count`` : int
        - ``unique_pct`` : float or None
        - ``sample_values`` : list
        - Optional numeric stats:
            - ``min``, ``max``, ``mean``
        - Optional string stats:
            - ``avg_length``, ``max_length``
    """
    # Base aggregations for all types
    agg_exprs: Dict[str, pl.Expr] = {
        "null_count": pl.col(col).null_count(),
        "unique_count": pl.col(col).n_unique(),
    }

    # Numeric stats
    if dtype.is_numeric():
        agg_exprs.update(
            {
                "min": pl.col(col).min(),
                "max": pl.col(col).max(),
                "mean": pl.col(col).mean(),
            }
        )

    # String stats
    elif dtype == pl.Utf8:
        length = pl.col(col).str.len_chars()
        agg_exprs.update(
            {
                "avg_length": length.mean(),
                "max_length": length.max(),
            }
        )

    # Convert dict → list of aliased expressions for Polars
    exprs = [expr.alias(name) for name, expr in agg_exprs.items()]

    # Streaming aggregation
    agg_df = lf.select(exprs).collect(streaming=True)
    stats = agg_df.to_dicts()[0]

    # Random sample values (via lazy)
    samples = lazy_random_sample(lf, col, n=3)

    # Row count: either given by max_rows or computed lazily
    if max_rows is not None:
        total_rows = max_rows
    else:
        total_rows = lf.select(pl.len()).collect(streaming=True).item()

    # Percentages
    null_count = stats["null_count"]
    unique_count = stats["unique_count"]

    null_pct = (null_count / total_rows * 100.0) if total_rows > 0 else 0.0
    unique_pct = (
        unique_count / total_rows * 100.0 if total_rows > 0 else None
    )

    # Approximate memory usage based on dtype and non-null count
    try:
        # Create a single-element Series of this dtype and inspect size
        dtype_size = pl.Series([None], dtype=dtype).estimated_size()
        approx_mem = dtype_size * (total_rows - null_count)
    except Exception:  # noqa: BLE001
        approx_mem = 0

    summary: Dict[str, Any] = {
        "column": col,
        "dtype": str(dtype),
        "memory_bytes_approx": approx_mem,
        "null_pct": null_pct,
        "unique_count": unique_count,
        "unique_pct": unique_pct,
        "sample_values": samples,
    }

    if dtype.is_numeric():
        summary.update(
            {
                "min": stats.get("min"),
                "max": stats.get("max"),
                "mean": stats.get("mean"),
            }
        )

    if dtype == pl.Utf8:
        summary.update(
            {
                "avg_length": stats.get("avg_length"),
                "max_length": stats.get("max_length"),
            }
        )

    return summary


def build_summary_table_lazy(
    lf: pl.LazyFrame,
    max_rows: Optional[int],
) -> pl.DataFrame:
    """
    Build a full summary table for all columns in a LazyFrame.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame representing the dataset (possibly head-limited).
    max_rows : int or None
        If not ``None``, indicates that only the first ``max_rows`` rows
        were used for profiling.

    Returns
    -------
    pl.DataFrame
        DataFrame in which each row summarizes a column, with fields
        such as:

        - ``column``
        - ``dtype``
        - ``memory_bytes_approx``
        - ``null_pct``
        - ``unique_count``
        - ``unique_pct``
        - ``sample_values``
        - numeric/string stats as applicable.
    """
    # Use collect_schema() to avoid unnecessary computation
    schema = lf.collect_schema()

    summaries: List[Dict[str, Any]] = []
    for col, dtype in schema.items():
        logging.debug("Summarizing column: %s (%s)", col, dtype)
        info = summarize_column_lazy(lf, col, dtype, max_rows)
        summaries.append(info)

    return pl.DataFrame(summaries)


# ======================================================================
# Reports
# ======================================================================

def build_text_report(
    input_path: Path,
    summary_df: pl.DataFrame,
    max_rows: Optional[int],
) -> str:
    """
    Build a detailed text report from the summary table.

    Parameters
    ----------
    input_path : Path
        Path to the input dataset.
    summary_df : pl.DataFrame
        Column summary table.
    max_rows : int or None
        If not ``None``, indicates that only the first ``max_rows`` rows
        were profiled.

    Returns
    -------
    str
        Multi-line string containing the report.
    """
    lines: List[str] = []
    lines.append(f"FILE: {input_path}")
    lines.append(f"COLUMNS: {summary_df.height}")

    if max_rows is not None:
        lines.append(f"NOTE: Profiling limited to first {max_rows} rows.")
    lines.append("=" * 80)

    for row in summary_df.to_dicts():
        lines.append(f"\nCOLUMN: {row['column']}")
        lines.append("-" * 80)
        for key, value in row.items():
            if key == "column":
                continue
            lines.append(f"  {key:20s}: {value}")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT\n")

    return "\n".join(lines)


# ======================================================================
# CLI
# ======================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Lazy streaming data profiler for Parquet/CSV files."
    )

    parser.add_argument(
        "file",
        type=Path,
        nargs="?",
        default=INPUT_FILE,
        help="Input Parquet or CSV file (default set in script config).",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory to write reports. If omitted, the directory of the "
            "input file is used."
        ),
    )

    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help=(
            "Optional base name for report files. "
            "Default: 'profile_report_{input_stem}'."
        ),
    )

    parser.add_argument(
        "--report-format",
        type=str,
        choices=["txt", "csv", "both"],
        default=DEFAULT_REPORT_FORMAT,
        help="Which report formats to generate (default: both).",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=(
            "If set, profile only the first N rows (lazy head). "
            "Useful for very large datasets."
        ),
    )

    parser.add_argument(
        "--log",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Logging level for the profiler.",
    )

    return parser


def main() -> None:
    """
    Run the CLI entry point.

    Steps
    -----
    1. Parse CLI arguments.
    2. Configure logging.
    3. Build a lazy scan of the input file.
    4. Optionally apply a row limit via ``head()``.
    5. Build a column summary table lazily.
    6. Print a tabular summary to the console.
    7. Optionally write text/CSV reports to disk.
    """

    args = build_arg_parser().parse_args()
    setup_logging(args.log)

    input_path: Path = args.file
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # Output directory defaults to the input file's directory
    out_dir: Path = args.out_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine base report name
    if args.report_name:
        base_name = args.report_name
    else:
        base_name = f"profile_report_{input_path.stem}"

    # Lazily scan file
    if input_path.suffix.lower() == ".parquet":
        lf = pl.scan_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        lf = pl.scan_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    # Optionally limit rows (still lazily)
    if args.max_rows is not None:
        logging.info("Applying lazy head(%d) to limit rows.", args.max_rows)
        lf = lf.head(args.max_rows)

    # Build summary table lazily
    summary_df = build_summary_table_lazy(lf, args.max_rows)

    summary_df = summary_df.sort("column")

    # Console table (always)
    print("\n=== COLUMN SUMMARY TABLE (LAZY STREAMING) ===")
    # Full printing configuration
    pl.Config.set_tbl_rows(None)                # unlimited rows
    pl.Config.set_tbl_cols(None)                # unlimited columns
    pl.Config.set_tbl_width_chars(2000)         # wide output
    pl.Config.set_fmt_str_lengths(2000)         # no string truncation
    pl.Config.set_fmt_table_cell_list_len(2000) # long lists fully visible
    # You can tweak display with pl.Config, but plain print is usually fine
    print(summary_df)
    print("\n")

    # Text report
    if args.report_format in ("txt", "both"):
        txt_report = build_text_report(input_path, summary_df, args.max_rows)
        txt_path = out_dir / f"{base_name}.txt"
        txt_path.write_text(txt_report, encoding="utf-8")
        logging.info("Text report saved → %s", txt_path)

    if args.report_format in ("csv", "both"):
        csv_path = out_dir / f"{base_name}.csv"

        # convert list column to string
        summary_df_csv = summary_df.with_columns(
            pl.col("sample_values")
                .map_elements(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else str(x))
                .cast(pl.String)        # important!
                .alias("sample_values")
        )

        summary_df_csv.write_csv(csv_path)
        logging.info("CSV summary saved → %s", csv_path)


if __name__ == "__main__":
    main()
