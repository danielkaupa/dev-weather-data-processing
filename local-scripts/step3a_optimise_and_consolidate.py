#!/usr/bin/env python3
"""
step3a_optimise_and_consolidate.py
==================================

End-to-end ERA5 Parquet pipeline in three stages:

1. **Optimize monthly files** (drop columns, cast dtypes) to a cleaned temp dir.
2. **Consolidate cleaned monthly files** into annual / biannual / quarterly Parquet.
3. **Rename columns** using metadata (datasetProcessingName) into final processed files.

Key features
------------
- Uses a *sample file* to infer the download group (prefix + UID) and processes
  only files belonging to that group.
- Parallelized with :class:`concurrent.futures.ProcessPoolExecutor`
  (configurable ``MAX_WORKERS``).
- Uses Polars ``scan_parquet`` + ``sink_parquet`` for lazy/streaming I/O where possible.
- Robust logging with timing and per-stage summaries.
- CLI with sensible defaults but also supports manual constants in the script.

Examples
--------
Default directories and modes, using a sample file::

    python step3a_optimise_and_consolidate.py \
        --sample-file ../data/interim/era5-world_INDIA_d514a3a3c256_2025_06.parquet

Specify output directories and modes explicitly::

    python step3a_optimise_and_consolidate.py \
        --sample-file ../data/interim/era5-world_INDIA_d514a3a3c256_2025_06.parquet \
        --input-dir ../data/interim \
        --clean-dir ../data/temp_clean \
        --agg-dir ../data/temp_agg \
        --output-dir ../data/processed \
        --metadata-json ../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json \
        --modes annual quarterly \
        --max-workers 8 \
        --overwrite \
        --cleanup-temp
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import polars as pl


# ---------------------------------------------------------------------
# DEFAULT CONFIG (can be overridden via CLI)
# ---------------------------------------------------------------------

DEFAULT_INPUT_DIR = Path("../data/interim")
SAMPLE_FILE = DEFAULT_INPUT_DIR / "era5-world_INDIA_d514a3a3c256_2025_06.parquet"
DEFAULT_CLEAN_DIR = Path("../data/temp/temp_clean")
DEFAULT_AGG_DIR = Path("../data/temp/temp_agg")
DEFAULT_OUTPUT_DIR = Path("../data/processed")

DEFAULT_METADATA_JSON = Path(
    "../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json"
)

DEFAULT_MODES = ["annual", "biannual", "quarterly"]
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_OVERWRITE = True
DEFAULT_MAX_WORKERS = max(int(multiprocessing.cpu_count() * 2 / 3), 2)

# Columns to drop in optimisation stage
DROP_COLS = [
    "centroid_in_region",
    "cell_area_m2",
    "number",
    "step",
    "surface",
]

# Manual rename map for non-ERA5-short-name columns (time, coords, etc.)
MANUAL_RENAME_MAP: Dict[str, str] = {
    # e.g. "time": "timestamp", "latitude": "lat", "longitude": "lon"
}

# Target dtypes for ERA5 variable shortNames (small, updated map)
CAST_MAP: Dict[str, pl.DataType] = {
    "longitude": pl.Float32,
    "latitude": pl.Float32,
}


# ---------------------------------------------------------------------
# REGEX / FILENAME PARSING
# ---------------------------------------------------------------------

# Matches: era5-world_INDIA_d514a3a3c256_2025_06.parquet
FNAME_RE = re.compile(
    r"^(?P<prefix>[^_]+_[^_]+)_"      # era5-world_INDIA or era5-world_N37W68S6E98
    r"(?P<uid>[A-Za-z0-9]+)_"         # d514a3a3c256
    r"(?P<year>\d{4})_"               # 2025
    r"(?P<month>\d{2})\.parquet$"     # 06.parquet
)


def parse_prefix_uid_year_month(path: Path) -> Tuple[str, str, int, int]:
    """
    Parse filenames of the form ``prefix_uid_YYYY_MM.parquet``.

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    prefix : str
        Dataset prefix (e.g. ``"era5-world_INDIA"``).
    uid : str
        Unique download/group identifier.
    year : int
        Four-digit year.
    month : int
        Two-digit month, as integer.

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.
    """
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename not recognized: {path.name}")
    return (
        m.group("prefix"),
        m.group("uid"),
        int(m.group("year")),
        int(m.group("month")),
    )


# ---------------------------------------------------------------------
# DATA CLASSES FOR SIMPLE STATS
# ---------------------------------------------------------------------

@dataclass
class FileStats:
    """
    Lightweight structure for per-file processing statistics.

    Parameters
    ----------
    stage : str
        Name of the pipeline stage, e.g. ``"stage1_optimise"``.
    name : str
        Output file name.
    cpu_time_sec : float
        CPU time spent on this file (in seconds).
    wall_time_sec : float
        Wall-clock time spent on this file (in seconds).
    row_count : int or None, optional
        Number of rows in the resulting file, if known.
    year : int or None, optional
        Year associated with this output (used for Stage 2 summaries).
    """
    stage: str
    name: str
    cpu_time_sec: float
    wall_time_sec: float
    row_count: int | None = None
    year: int | None = None


# ---------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------

def setup_logging(level: str) -> None:
    """
    Configure the root logger.

    Parameters
    ----------
    level : {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        Logging level.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def log_banner(title: str) -> None:
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"{title:^70}")  # centered
    logging.info("=" * 70)
    logging.info("")

# ---------------------------------------------------------------------
# METADATA / RENAME MAP
# ---------------------------------------------------------------------

def load_metadata_rename_map(metadata_json: Path) -> Dict[str, str]:
    """
    Load ERA5 variable metadata and build a rename map.

    The mapping goes from ERA5 *shortName* (e.g. ``"2t"``) to a
    processing-friendly name, ideally ``datasetProcessingName``.
    If that field is missing, it falls back to a naive conversion
    of ``fullName`` to snake_case.

    Parameters
    ----------
    metadata_json : Path
        Path to the JSON metadata file.

    Returns
    -------
    dict
        Mapping from existing column names (shortName) to their new
        processing-friendly names.
    """
    with metadata_json.open() as f:
        meta = json.load(f)

    rename_map: Dict[str, str] = {}
    for short, entry in meta.items():
        if "datasetProcessingName" in entry:
            rename_map[short] = entry["datasetProcessingName"]
        else:
            full = entry.get("fullName", "") or short
            rename_map[short] = full.lower().replace(" ", "_")

    return rename_map


# ---------------------------------------------------------------------
# UTILS: DISCOVER FILES IN GROUP, GROUP BY YEAR, CLEANUP, TIME
# ---------------------------------------------------------------------

def discover_monthly_files(
    input_dir: Path,
    prefix: str,
    uid: str,
) -> List[Path]:
    """
    Find all monthly Parquet files belonging to a given prefix+uid group.

    Filenames must match ``prefix_uid_YYYY_MM.parquet``. Any other files
    matching the glob pattern are logged and skipped.

    Parameters
    ----------
    input_dir : Path
        Directory where the monthly Parquet files are stored.
    prefix : str
        Dataset prefix (e.g., ``"era5-world_INDIA"``).
    uid : str
        Unique download UID.

    Returns
    -------
    list of Path
        Sorted list of monthly Parquet files for the group.
    """
    pattern = f"{prefix}_{uid}_*.parquet"
    files = sorted(input_dir.glob(pattern))

    valid_files: List[Path] = []
    years: List[int] = []

    for f in files:
        try:
            _, _, year, _ = parse_prefix_uid_year_month(f)
            valid_files.append(f)
            years.append(year)
        except ValueError:
            logging.warning("Skipping unrecognized file: %s", f.name)

    if valid_files:
        min_year = min(years)
        max_year = max(years)
        logging.info(
            "Discovered %d monthly files for group %s_%s spanning years %d → %d",
            len(valid_files),
            prefix,
            uid,
            min_year,
            max_year,
        )
    else:
        logging.info(
            "Discovered 0 monthly files for group %s_%s",
            prefix,
            uid,
        )

    return valid_files


def group_by_year(files: List[Path]) -> Dict[int, List[Path]]:
    """
    Group monthly files by year.

    Parameters
    ----------
    files : list of Path
        Monthly Parquet files.

    Returns
    -------
    dict
        Mapping ``{year: [Path, Path, ...]}``, where the file list
        for each year is sorted by month.
    """
    by_year: Dict[int, List[Tuple[int, Path]]] = {}
    for f in files:
        _, _, year, month = parse_prefix_uid_year_month(f)
        by_year.setdefault(year, []).append((month, f))

    out: Dict[int, List[Path]] = {}
    for year, month_files in by_year.items():
        out[year] = [p for (m, p) in sorted(month_files, key=lambda x: x[0])]
    return out


def cleanup_temp_dirs(clean_dir: Path, agg_dir: Path) -> Dict[str, int]:
    """
    Delete the entire clean_dir and agg_dir directories,
    but DO NOT delete their parent directories.
    """
    import shutil

    counts = {}

    # Delete clean_dir
    if clean_dir.exists():
        file_count = sum(1 for _ in clean_dir.rglob("*"))
        shutil.rmtree(clean_dir)
        counts[clean_dir] = file_count
    else:
        counts[clean_dir] = 0

    # Delete agg_dir
    if agg_dir.exists():
        file_count = sum(1 for _ in agg_dir.rglob("*"))
        shutil.rmtree(agg_dir)
        counts[agg_dir] = file_count
    else:
        counts[agg_dir] = 0

    return counts

def fmt_hms(seconds: float) -> str:
    """Format seconds as HHh:MMm:SSs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


def infer_cast_and_drop_columns(
    monthly_files: List[Path],
    cast_map: Dict[str, pl.DataType],
    drop_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Infer which columns are actually cast and dropped across all monthly files.

    Parameters
    ----------
    monthly_files : list of Path
        Original monthly Parquet files.
    cast_map : dict
        Mapping from column name to target dtype.
    drop_cols : list of str
        Columns configured for dropping.

    Returns
    -------
    cast_columns : list of str
        Alphabetically sorted list of columns that exist in at least one file
        and appear in ``cast_map``.
    dropped_columns : list of str
        Alphabetically sorted list of columns that exist in at least one file
        and appear in ``drop_cols``.
    """
    cast_found: set[str] = set()
    drop_found: set[str] = set()

    for f in monthly_files:
        try:
            schema = pl.scan_parquet(f).collect_schema()
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not inspect schema for %s: %s", f.name, e)
            continue

        names = set(schema.names())
        cast_found.update({c for c in cast_map if c in names})
        drop_found.update({c for c in drop_cols if c in names})

    return sorted(cast_found), sorted(drop_found)


# ---------------------------------------------------------------------
# STAGE 1: OPTIMISE MONTHLY FILES (DROP + CAST) → CLEAN_DIR
# ---------------------------------------------------------------------

def build_cast_expr(col: str, schema: Dict[str, pl.DataType], target_dtype: pl.DataType) -> pl.Expr:
    """
    Build a safe casting expression for a column.

    If the column is ``Utf8``, attempt a numeric parse by stripping
    non-numeric characters, then cast. If already numeric, cast directly
    with ``strict=False``.

    Parameters
    ----------
    col : str
        Column name.
    schema : dict
        Mapping of column name to Polars dtype for the source file.
    target_dtype : polars.DataType
        Target dtype for the column.

    Returns
    -------
    polars.Expr
        Expression that performs a best-effort cast into ``target_dtype``.
    """
    current_dtype = schema.get(col)

    if current_dtype == pl.Utf8:
        return (
            pl.col(col)
            .str.replace_all(r"[^\d\.\-eE]", "")
            .str.replace_all(r"^\s*$", None)
            .cast(target_dtype, strict=False)
            .alias(col)
        )

    if current_dtype and current_dtype.is_numeric():
        return pl.col(col).cast(target_dtype, strict=False).alias(col)

    # Fallback: attempt a generic cast
    return pl.col(col).cast(target_dtype, strict=False).alias(col)


def optimise_monthly_file(
    path: Path,
    clean_dir: Path,
    overwrite: bool,
) -> Tuple[Path | None, FileStats | None]:
    """
    Optimise a single monthly Parquet file.

    Operations
    ----------
    - Drop configured columns.
    - Cast selected columns to target dtypes (e.g., latitude/longitude).

    The function uses lazy ``scan_parquet`` and ``sink_parquet`` to
    minimize memory usage.

    Parameters
    ----------
    path : Path
        Input monthly Parquet file.
    clean_dir : Path
        Directory to store the optimised file.
    overwrite : bool
        If ``True``, overwrite existing cleaned files. Otherwise, skip.

    Returns
    -------
    clean_path : Path or None
        Path to the cleaned file, or ``None`` if processing failed.
    stats : FileStats or None
        Statistics for the processed file, or ``None`` if skipped/failure.
    """
    clean_path = clean_dir / path.name
    if clean_path.exists() and not overwrite:
        logging.info("[STAGE1] SKIP (exists): %s", clean_path.name)
        return clean_path, None

    start_wall = time.time()
    start_cpu = time.process_time()

    try:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
        names = schema.names()

        # drop columns
        drop_cols = [c for c in DROP_COLS if c in names]
        if drop_cols:
            lf = lf.drop(drop_cols)

        # cast expressions
        cast_exprs = [
            build_cast_expr(col, schema, dtype)
            for col, dtype in CAST_MAP.items()
            if col in names
        ]

        if cast_exprs:
            lf = lf.with_columns(cast_exprs)

        lf.sink_parquet(clean_path, compression="snappy", statistics=True)

        # estimate row count lazily from output
        row_count = (
            pl.scan_parquet(clean_path)
            .select(pl.len().alias("n"))
            .collect()
            .item()
        )

        cpu = time.process_time() - start_cpu
        wall = time.time() - start_wall

        stats = FileStats(
            stage="stage1_optimise",
            name=clean_path.name,
            cpu_time_sec=cpu,
            wall_time_sec=wall,
            row_count=row_count,
            year=None,
        )

        logging.info(
            "[STAGE1] OK: %-40s | rows=%d | CPU=%.2fs | WALL=%.2fs",
            clean_path.name,
            row_count,
            cpu,
            wall,
        )

        return clean_path, stats

    except Exception as e:  # noqa: BLE001
        logging.error("[STAGE1] FAIL %s: %s", path.name, e)
        return None, None


def run_stage1_parallel(
    monthly_files: List[Path],
    clean_dir: Path,
    max_workers: int,
    overwrite: bool,
) -> Tuple[List[Path], List[FileStats]]:
    """
    Run Stage 1 (optimisation) in parallel over all monthly files.

    Parameters
    ----------
    monthly_files : list of Path
        Monthly input Parquet files.
    clean_dir : Path
        Directory for cleaned outputs.
    max_workers : int
        Maximum number of worker processes.
    overwrite : bool
        Whether to overwrite any existing cleaned files.

    Returns
    -------
    cleaned_files : list of Path
        Paths to all successfully cleaned files.
    all_stats : list of FileStats
        Per-file statistics for successfully processed files.
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    cleaned_files: List[Path] = []
    all_stats: List[FileStats] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(optimise_monthly_file, f, clean_dir, overwrite): f
            for f in monthly_files
        }
        for fut in as_completed(futures):
            clean_path, stats = fut.result()
            if clean_path is not None:
                cleaned_files.append(clean_path)
            if stats is not None:
                all_stats.append(stats)

    cast_cols, drop_cols = infer_cast_and_drop_columns(
        monthly_files,
        CAST_MAP,
        DROP_COLS,
    )
    log_banner("STAGE 1 — File Optimisation Complete")
    logging.info(f"\tFiles processed: [{len(cleaned_files)}]")
    logging.info(
        f"\tCast [{len(cast_cols)}] columns: {', '.join(cast_cols) if cast_cols else ''}")
    logging.info(
        f"\tDropped [{len(drop_cols)}] columns: {', '.join(drop_cols) if drop_cols else ''}"
    )

    return cleaned_files, all_stats


# ---------------------------------------------------------------------
# STAGE 2: CONSOLIDATE CLEANED FILES → AGG_DIR
# ---------------------------------------------------------------------

def get_all_columns(files: List[Path]) -> Dict[str, pl.DataType]:
    """
    Compute the union schema (column → dtype) across a list of Parquet files.

    Float32 is promoted to Float64 to avoid inconsistencies.

    Parameters
    ----------
    files : list of Path
        Parquet files to inspect.

    Returns
    -------
    dict
        Mapping of column name to a unified Polars dtype.
    """
    all_cols: Dict[str, pl.DataType] = {}
    for f in files:
        schema = pl.scan_parquet(f).collect_schema()
        for name, dtype in schema.items():
            if dtype == pl.Float32:
                dtype = pl.Float64
            all_cols[name] = dtype
    return all_cols


def normalize_lazy(lf: pl.LazyFrame, all_cols: Dict[str, pl.DataType]) -> pl.LazyFrame:
    """
    Ensure a LazyFrame matches a global schema.

    Operations
    ----------
    - Add missing columns as nulls cast to target dtype.
    - Cast existing columns to target dtype with ``strict=False``.
    - Select columns in sorted order.

    Parameters
    ----------
    lf : polars.LazyFrame
        Input lazy frame.
    all_cols : dict
        Global schema mapping from column name to dtype.

    Returns
    -------
    polars.LazyFrame
        Normalised lazy frame.
    """
    existing = set(lf.collect_schema().names())
    missing = set(all_cols) - existing

    if missing:
        lf = lf.with_columns(
            [
                pl.lit(None).cast(all_cols[col]).alias(col)
                for col in missing
            ]
        )

    cast_exprs: List[pl.Expr] = []
    for col, dtype in all_cols.items():
        if col in existing:
            cast_exprs.append(pl.col(col).cast(dtype, strict=False).alias(col))

    if cast_exprs:
        lf = lf.with_columns(cast_exprs)

    return lf.select(sorted(all_cols.keys()))


def consolidate_year(
    year: int,
    files_for_year: List[Path],
    prefix: str,
    uid: str,
    modes: List[str],
    agg_dir: Path,
    overwrite: bool,
) -> Tuple[List[Path], List[FileStats]]:
    """
    Consolidate one year's cleaned monthly files into annual / biannual / quarterly files.

    Parameters
    ----------
    year : int
        Year being consolidated.
    files_for_year : list of Path
        Cleaned monthly files for this year.
    prefix : str
        Dataset prefix (e.g., ``"era5-world_INDIA"``).
    uid : str
        Unique download UID.
    modes : list of {"annual", "biannual", "quarterly"}
        Consolidation modes to run.
    agg_dir : Path
        Directory to write consolidated Parquet files.
    overwrite : bool
        Whether to overwrite existing consolidated files.

    Returns
    -------
    outputs : list of Path
        Paths to consolidated files created for this year.
    stats : list of FileStats
        Per-output statistics for this year.
    """
    agg_dir.mkdir(parents=True, exist_ok=True)
    all_outputs: List[Path] = []
    stats: List[FileStats] = []

    if not files_for_year:
        return all_outputs, stats

    all_cols = get_all_columns(files_for_year)

    def group_files_mode(files: List[Path], mode: str) -> List[List[Path]]:
        if mode == "annual":
            return [files]
        elif mode == "biannual":
            g1 = [f for f in files if parse_prefix_uid_year_month(f)[3] <= 6]
            g2 = [f for f in files if parse_prefix_uid_year_month(f)[3] > 6]
            return [g1, g2]
        elif mode == "quarterly":
            groups: List[List[Path]] = []
            for i in range(4):
                low = 1 + 3 * i
                high = low + 2
                groups.append(
                    [f for f in files if low <= parse_prefix_uid_year_month(f)[3] <= high]
                )
            return groups
        else:
            raise ValueError(f"Unknown consolidation mode: {mode}")

    for mode in modes:
        for idx, group in enumerate(group_files_mode(files_for_year, mode), start=1):
            if not group:
                continue

            if mode == "annual":
                out_name = f"{prefix}_{uid}_{year}.parquet"
            elif mode == "biannual":
                half = "H1" if idx == 1 else "H2"
                out_name = f"{prefix}_{uid}_{year}_{half}.parquet"
            else:  # quarterly
                out_name = f"{prefix}_{uid}_{year}_Q{idx}.parquet"

            out_path = agg_dir / out_name
            if out_path.exists() and not overwrite:
                logging.info("[STAGE2] SKIP (exists): %s", out_path.name)
                all_outputs.append(out_path)
                continue

            start_wall = time.time()
            start_cpu = time.process_time()

            lfs = [
                normalize_lazy(pl.scan_parquet(f), all_cols)
                for f in group
            ]
            combined = pl.concat(lfs, how="vertical")

            combined.sink_parquet(out_path, compression="snappy", statistics=True)

            row_count = (
                pl.scan_parquet(out_path)
                .select(pl.len().alias("n"))
                .collect()
                .item()
            )

            cpu = time.process_time() - start_cpu
            wall = time.time() - start_wall

            stats.append(
                FileStats(
                    stage=f"stage2_consolidate_{mode}",
                    name=out_path.name,
                    cpu_time_sec=cpu,
                    wall_time_sec=wall,
                    row_count=row_count,
                    year=year,
                )
            )

            logging.info(
                "[STAGE2] OK (%s): %-40s | files=%2d | rows=%d | CPU=%.2fs | WALL=%.2fs",
                mode,
                out_path.name,
                len(group),
                row_count,
                cpu,
                wall,
            )

            all_outputs.append(out_path)

    return all_outputs, stats


def run_stage2_parallel(
    cleaned_files: List[Path],
    prefix: str,
    uid: str,
    modes: List[str],
    agg_dir: Path,
    max_workers: int,
    overwrite: bool,
) -> Tuple[List[Path], List[FileStats]]:
    """
    Run Stage 2 (consolidation) in parallel by year.

    Parameters
    ----------
    cleaned_files : list of Path
        Cleaned monthly files from Stage 1.
    prefix : str
        Dataset prefix.
    uid : str
        Unique download UID.
    modes : list of {"annual", "biannual", "quarterly"}
        Consolidation modes to run.
    agg_dir : Path
        Directory to write consolidated Parquet files.
    max_workers : int
        Maximum number of worker processes.
    overwrite : bool
        Whether to overwrite any existing consolidated files.

    Returns
    -------
    agg_files : list of Path
        Paths to all consolidated output files.
    all_stats : list of FileStats
        Per-output statistics for all years.
    """
    year_map = group_by_year(cleaned_files)
    all_outputs: List[Path] = []
    all_stats: List[FileStats] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                consolidate_year,
                year,
                files_for_year,
                prefix,
                uid,
                modes,
                agg_dir,
                overwrite,
            ): year
            for year, files_for_year in year_map.items()
        }

        for fut in as_completed(futures):
            outputs, stats = fut.result()
            all_outputs.extend(outputs)
            all_stats.extend(stats)

    # Detailed mode/year summary (Option B)
    by_mode_year: Dict[str, Dict[int, List[str]]] = {}
    for s in all_stats:
        mode = s.stage.replace("stage2_consolidate_", "").upper()
        if s.year is None:
            continue
        by_mode_year.setdefault(mode, {}).setdefault(s.year, []).append(s.name)

    log_banner("STAGE 2 — Completed File Consolidation")
    for mode in sorted(by_mode_year.keys()):
        logging.info(f"{mode}")
        for year in sorted(by_mode_year[mode].keys()):
            files_for_year = sorted(by_mode_year[mode][year])
            logging.info(f"\t[{year}] → {', '.join(files_for_year)}")

    return all_outputs, all_stats


# ---------------------------------------------------------------------
# STAGE 3: RENAME COLUMNS → OUTPUT_DIR
# ---------------------------------------------------------------------

def rename_columns_file(
    agg_path: Path,
    output_dir: Path,
    meta_rename: Dict[str, str],
    manual_rename: Dict[str, str],
    overwrite: bool,
) -> Tuple[Path | None, FileStats | None]:
    """
    Rename columns in an aggregated file using a metadata-based rename map.

    Parameters
    ----------
    agg_path : Path
        Input aggregated Parquet file.
    output_dir : Path
        Directory where the renamed Parquet file will be written.
    meta_rename : dict
        Mapping from existing column names to new names, derived from metadata.
    manual_rename : dict
        Additional manual mapping for non-ERA5 columns (e.g., coordinates).
    overwrite : bool
        Whether to overwrite existing outputs.

    Returns
    -------
    out_path : Path or None
        Path to the renamed file, or ``None`` if skipped/failure.
    stats : FileStats or None
        Statistics for the file, or ``None`` if skipped/failure.
    """

    out_path = output_dir / agg_path.name
    if out_path.exists() and not overwrite:
        logging.info(f"[STAGE3] SKIP (exists): {out_path.name}")
        return out_path, None

    start_wall = time.time()
    start_cpu = time.process_time()

    try:
        lf = pl.scan_parquet(agg_path)
        schema = lf.collect_schema()
        names = schema.names()

        rename_map: Dict[str, str] = {}
        for col in names:
            if col in meta_rename:
                rename_map[col] = meta_rename[col]
            elif col in manual_rename:
                rename_map[col] = manual_rename[col]

        if rename_map:
            lf = lf.rename(rename_map)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        lf.sink_parquet(out_path, compression="snappy", statistics=True)

        row_count = (
            pl.scan_parquet(out_path)
            .select(pl.len().alias("n"))
            .collect()
            .item()
        )

        cpu = time.process_time() - start_cpu
        wall = time.time() - start_wall

        stats = FileStats(
            stage="stage3_rename",
            name=out_path.name,
            cpu_time_sec=cpu,
            wall_time_sec=wall,
            row_count=row_count,
            year=None,
        )

        logging.info(
            f"[STAGE3] OK: {out_path.name:40} | rows={row_count} | CPU={cpu:.2f}s | WALL={wall:.2f}s",
        )

        return out_path, stats

    except Exception as e:  # noqa: BLE001
        logging.error("[STAGE3] FAIL %s: %s", agg_path.name, e)
        return None, None


def run_stage3_parallel(
    agg_files: List[Path],
    output_dir: Path,
    meta_rename: Dict[str, str],
    manual_rename: Dict[str, str],
    max_workers: int,
    overwrite: bool,
    ) -> List[FileStats]:
    """
    Run Stage 3 (rename columns) in parallel over consolidated files.

    Parameters
    ----------
    agg_files : list of Path
        Aggregated Parquet files from Stage 2.
    output_dir : Path
        Directory where final renamed outputs are written.
    meta_rename : dict
        Mapping from existing column names to new names.
    manual_rename : dict
        Additional manual renames.
    max_workers : int
        Maximum number of worker processes.
    overwrite : bool
        Whether to overwrite any existing outputs.

    Returns
    -------
    all_stats : list of FileStats
        Per-output statistics for Stage 3.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stats: List[FileStats] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                rename_columns_file,
                f,
                output_dir,
                meta_rename,
                manual_rename,
                overwrite,
            ): f
            for f in agg_files
        }

        for fut in as_completed(futures):
            _, stats = fut.result()
            if stats is not None:
                all_stats.append(stats)

    return all_stats


# ---------------------------------------------------------------------
# ARGPARSE / CONTROLLER
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "ERA5 pipeline: optimise monthly files, consolidate, and rename columns "
            "for a single download group inferred from a sample file."
        )
    )

    parser.add_argument(
        "--sample-file",
        type=Path,
        default=SAMPLE_FILE,
        help="Path to a sample Parquet file in the download group to process.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing monthly input Parquet files "
            f"(default: {DEFAULT_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help=(
            "Directory for cleaned monthly outputs (Stage 1) "
            f"(default: {DEFAULT_CLEAN_DIR})."
        ),
    )
    parser.add_argument(
        "--agg-dir",
        type=Path,
        default=DEFAULT_AGG_DIR,
        help=(
            "Directory for consolidated outputs (Stage 2) "
            f"(default: {DEFAULT_AGG_DIR})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for final renamed outputs (Stage 3) "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=DEFAULT_METADATA_JSON,
        help=(
            "Metadata JSON with datasetProcessingName field "
            f"(default: {DEFAULT_METADATA_JSON})."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["annual", "biannual", "quarterly"],
        default=DEFAULT_MODES,
        help="Consolidation modes to run (default: annual biannual quarterly).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Max worker processes (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=DEFAULT_OVERWRITE,
        help="Overwrite existing outputs instead of skipping.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=DEFAULT_LOG_LEVEL,
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Delete intermediate cleaned and aggregated files after success.",
    )

    return parser


def main() -> None:
    """
    Execute the full three-stage ERA5 processing pipeline.

    Steps
    -----
    1. Parse CLI arguments and configure logging.
    2. Infer prefix/UID/year/month from the sample file.
    3. Discover all monthly files in the same download group.
    4. Stage 1: optimise monthly files (drop + cast) → cleaned dir.
    5. Stage 2: consolidate cleaned files (annual/biannual/quarterly) → agg dir.
    6. Stage 3: rename columns according to metadata → output dir.
    7. Optionally clean up intermediate directories.
    8. Log global timing and per-stage summaries.
    """
    args = build_arg_parser().parse_args()
    setup_logging(args.log_level)

    overall_wall_start = time.time()
    overall_cpu_start = time.process_time()

    log_banner("STARTING [step3a_optimise_and_consolidate.py]")

    logging.info("Running script with following configuration:")
    for arg, value in vars(args).items():
        logging.info(f"  -- {arg}: {value}")

    sample = args.sample_file
    if not sample.exists():
        raise FileNotFoundError(sample)

    # logging.info("")")
    prefix, uid, year, month = parse_prefix_uid_year_month(sample)
    # logging.info(
    #     "Sample file: %s (prefix=%s, uid=%s, year=%d, month=%02d)",
    #     sample.name,
    #     prefix,
    #     uid,
    #     year,
    #     month,
    # )

    # Ensure dirs exist
    for d in (args.input_dir, args.clean_dir, args.agg_dir, args.output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load metadata rename map
    meta_rename = load_metadata_rename_map(args.metadata_json)
    logging.info(
        "Loaded metadata rename map from %s (%d entries)",
        args.metadata_json,
        len(meta_rename),
    )
    logging.info("Starting processing....")
    # -----------------------------------------------------------------
    # STAGE 1: optimise monthly → clean_dir
    # -----------------------------------------------------------------
    monthly_files = discover_monthly_files(args.input_dir, prefix, uid)
    if not monthly_files:
        logging.error("No monthly files found for group %s_%s", prefix, uid)
        return

    # ----- TRUE STAGE 1 WALL TIME -----
    stage1_wall_start = time.time()

    cleaned_files, stats1 = run_stage1_parallel(
        monthly_files,
        args.clean_dir,
        args.max_workers,
        args.overwrite,
    )

    stage1_wall_true = time.time() - stage1_wall_start


    # -----------------------------------------------------------------
    # STAGE 2: consolidate cleaned → agg_dir
    # -----------------------------------------------------------------
    stage2_wall_start = time.time()

    agg_files, stats2 = run_stage2_parallel(
        cleaned_files,
        prefix,
        uid,
        args.modes,
        args.agg_dir,
        args.max_workers,
        args.overwrite,
    )

    stage2_wall_true = time.time() - stage2_wall_start
    # -----------------------------------------------------------------
    # STAGE 3: rename columns → output_dir
    # -----------------------------------------------------------------
    stage3_wall_start = time.time()

    stats3 = run_stage3_parallel(
        agg_files,
        args.output_dir,
        meta_rename,
        MANUAL_RENAME_MAP,
        args.max_workers,
        args.overwrite,
    )

    stage3_wall_true = time.time() - stage3_wall_start

    # -----------------------------------------------------------------
    # STAGE 3 SUMMARY
    # -----------------------------------------------------------------
    log_banner("STAGE 3 — Column Renaming Complete")

    logging.info(f"Files renamed: {len(stats3)}")
    logging.info(f"Output directory: {args.output_dir}")

    logging.info(
        "Mapping based on `datasetProcessingName` from metadata: %s",
        args.metadata_json,
    )

    # optional: list renamed files
    renamed_files = [s.name for s in stats3]
    for name in sorted(renamed_files):
        logging.info(f"  - {name}")

    # -----------------------------------------------------------------
    # STAGE 4 (optional): cleanup
    # -----------------------------------------------------------------
    if args.cleanup_temp:
        log_banner("STAGE 4 — Cleanup of Temporary Directories")

        counts = cleanup_temp_dirs(args.clean_dir, args.agg_dir)

        total_deleted = sum(counts.values())
        logging.info(f"Total files deleted: {total_deleted}")

        for d, n in counts.items():
            logging.info(f"  • {d} → deleted {n} file(s)")

        logging.info("")

    # -----------------------------------------------------------------
    # PIPELINE TIMINGS
    # -----------------------------------------------------------------
    wall_total = time.time() - overall_wall_start
    cpu_total = time.process_time() - overall_cpu_start

    # -----------------------------------------------------------------
    # PIPELINE COMPLETE — Unified Summary
    # -----------------------------------------------------------------
    log_banner("PIPELINE COMPLETE — SUMMARY")

    group_key = f"{prefix}_{uid}"
    total_files = len(monthly_files)
    total_outputs = len(stats3)

    logging.info(f"  Grouping key(s)          : {group_key}")
    logging.info(f"  Monthly files processed   : {total_files}")
    logging.info(f"  Final output files        : {total_outputs}")
    logging.info(f"  Consolidation modes       : {', '.join(args.modes)}")
    logging.info(f"  Parallel backend          : processpool")
    logging.info(f"  Parallel workers          : {args.max_workers}")
    logging.info("")

    logging.info("  TIMINGS")
    logging.info("  --------------------------------------------------")
    avg_per_file = wall_total / max(total_files, 1)
    logging.info(f"  Average per file          : {fmt_hms(avg_per_file)}")
    logging.info(f"  CPU total time            : {fmt_hms(cpu_total)}")
    logging.info(f"  TOTAL WALL TIME           : {fmt_hms(wall_total)}")
    logging.info("  --------------------------------------------------")
    logging.info("")

    logging.info("  PER-STAGE BREAKDOWN")
    logging.info("  --------------------------------------------------")
    def stage_summary(label, true_wall, stats):
        worker_wall = sum(s.wall_time_sec for s in stats)
        worker_cpu  = sum(s.cpu_time_sec for s in stats)

        logging.info(
            f"{label:<10} | TRUE_WALL={fmt_hms(true_wall)} | "
            f"WORKER_WALL_SUM={fmt_hms(worker_wall)} | WORKER_CPU_SUM={fmt_hms(worker_cpu)} | "
            f"files={len(stats):3d}"
        )

    stage_summary("STAGE1", stage1_wall_true, stats1)
    stage_summary("STAGE2", stage2_wall_true, stats2)
    stage_summary("STAGE3", stage3_wall_true, stats3)

    logging.info("  --------------------------------------------------")
    logging.info("  --------------------------------------------------")
    logging.info("  --------------------------------------------------")
    logging.info(" Processing completed")


if __name__ == "__main__":
    main()
