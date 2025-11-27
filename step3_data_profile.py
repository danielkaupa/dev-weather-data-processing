#!/usr/bin/env python3
"""
data_profile.py — Lightweight schema & memory profiler for Parquet/CSV files.

Usage:
    python data_profile.py input.parquet
    python data_profile.py input.csv --out report.txt
"""

import argparse
from pathlib import Path
import polars as pl

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def safe_sample(df: pl.DataFrame, col: str, n=3):
    """Return a few sample values from a column."""
    try:
        return df[col].drop_nulls().head(n).to_list()
    except Exception:
        return ["<error sampling>"]


def compute_memory_usage(df: pl.DataFrame, col: str):
    """Approx column memory usage using Polars' internal byte size."""
    try:
        return df[col].estimated_size()
    except Exception:
        return 0


def summarize_column(df: pl.DataFrame, col: str):
    s = df[col]
    dtype = s.dtype
    n_rows = len(df)

    # ----- Unique values -----
    try:
        unique_count = s.n_unique()
    except Exception:
        unique_count = None

    unique_pct = (unique_count / n_rows * 100) if (unique_count is not None and n_rows > 0) else None

    summary = {
        "column": col,
        "dtype": str(dtype),
        "memory_bytes": compute_memory_usage(df, col),
        "null_pct": float(s.null_count() / n_rows * 100) if n_rows else 0,
        "sample_values": safe_sample(df, col),
        "unique_count": unique_count,
        "unique_pct": unique_pct,
    }

    # Numeric summary
    if dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ):
        summary["min"] = s.min()
        summary["max"] = s.max()
        summary["mean"] = s.mean()

    # String summary
    if dtype == pl.Utf8:
        lengths = s.str.len_chars()
        summary["avg_length"] = lengths.mean()
        summary["max_length"] = lengths.max()

    return summary


# ------------------------------------------------------------
# Main profiling logic
# ------------------------------------------------------------

def profile_file(path: Path) -> str:
    # Load file
    if path.suffix.lower() == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pl.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    report_lines = []
    report_lines.append(f"FILE: {path}")
    report_lines.append(f"ROWS: {df.height}")
    report_lines.append(f"COLUMNS: {df.width}")
    report_lines.append("=" * 80)

    for col in df.columns:
        info = summarize_column(df, col)

        report_lines.append(f"\nCOLUMN: {info['column']}")
        report_lines.append("-" * 80)
        report_lines.append(f"  dtype           : {info['dtype']}")
        report_lines.append(f"  memory (bytes)  : {info['memory_bytes']:,}")
        report_lines.append(f"  null %          : {info['null_pct']:.2f}%")
        report_lines.append(f"  unique values   : {info['unique_count']}")
        if info["unique_pct"] is not None:
            report_lines.append(f"  unique %        : {info['unique_pct']:.2f}%")
        report_lines.append(f"  sample values   : {info['sample_values']}")

        # Numeric stats
        if "min" in info:
            report_lines.append(f"  min / max       : {info['min']} / {info['max']}")
            report_lines.append(f"  mean            : {info['mean']}")

        # String stats
        if "avg_length" in info:
            report_lines.append(f"  avg length      : {info['avg_length']:.2f}")
            report_lines.append(f"  max length      : {info['max_length']}")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT\n")

    return "\n".join(report_lines)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quick data profiler for Parquet/CSV.")
    parser.add_argument("file", type=Path, help="Input Parquet or CSV file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional output report text file")

    args = parser.parse_args()

    report = profile_file(args.file)

    # Always print to terminal
    print(report)

    # Optionally save to file
    if args.out:
        args.out.write_text(report)
        print(f"\n[OK] Report saved to {args.out}")


if __name__ == "__main__":
    main()
