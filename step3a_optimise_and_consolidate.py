"""
step3a_optimise_and_consolidate.py (MPI version)
===============================================

End-to-end ERA5 Parquet pipeline in three stages, parallelised over YEARS
using MPI ranks:

1. Optimise monthly files (filter timestamps, drop columns, cast dtypes) → cleaned temp dir.
2. Consolidate cleaned monthly files → annual / biannual / quarterly Parquet.
3. Rename columns using metadata (datasetProcessingName) → final processed files.

Design
------
- Rank 0:
    * Parses CLI
    * Infers download group from sample file (prefix + UID)
    * Discovers all monthly files for that group
    * Groups files by year
    * Loads metadata JSON and builds rename map
    * Loads reference Parquet schema and builds a GLOBAL_DTYPE_MAP:
        GLOBAL_DTYPE_MAP[col] = CAST_MAP[col] (if override)
                             = ref_schema[col] (otherwise)
      Columns in DROP_COLS are excluded from GLOBAL_DTYPE_MAP.
    * Broadcasts the work plan and GLOBAL_DTYPE_MAP to all ranks

- All ranks:
    * Receive the same year→file map, rename map, and GLOBAL_DTYPE_MAP
    * Are assigned a subset of YEARS: year[i] goes to rank (i % size)

- Stage 1:
    * Each rank:
        - For each monthly file in its years:
            * Validates schema against GLOBAL_DTYPE_MAP (allowing DROP_COLS)
            * Filters out-of-bounds timestamps using valid_time/time
            * Drops DROP_COLS
            * Validates schema AGAIN:
                columns must exactly equal GLOBAL_DTYPE_MAP keys
            * Casts columns where dtype != GLOBAL_DTYPE_MAP[col]
            * Writes cleaned file
    * STOP & GATHER (MPI Barrier + gather stats & outputs).

- Stage 2:
    * Each rank:
        - For each year assigned:
            * Reads cleaned monthly files
            * (Optional safety: validate schema against GLOBAL_DTYPE_MAP)
            * Selects columns in sorted(GLOBAL_DTYPE_MAP.keys())
            * Concatenates vertically
            * Writes annual / biannual / quarterly Parquet
    * STOP & GATHER (Barrier + gather).

- Stage 3:
    * Each rank:
        * Renames consolidated files for its years using metadata.
    * STOP & GATHER (Barrier + gather).

- Rank 0:
    * Prints global per-stage + pipeline summary.
    * Optionally cleans up temp directories.

If run without mpiexec, it still works with a single MPI rank (size=1).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl
from mpi4py import MPI


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

# Columns to drop in optimisation stage (removed BEFORE global schema is enforced)
DROP_COLS = [
    "centroid_in_region",
    "cell_area_m2",
    "number",
    "step",
    "surface",
    "valid_time",
]

# Manual rename map for non-ERA5-short-name columns (time, coords, etc.)
MANUAL_RENAME_MAP: Dict[str, str] = {
    # e.g. "time": "timestamp", "latitude": "lat", "longitude": "lon"
}

# Target dtypes for ERA5 variable shortNames (explicit overrides)
CAST_MAP: Dict[str, pl.DataType] = {
    "longitude": pl.Float32,
    "latitude": pl.Float32,
    "2t": pl.Float32,
    "tp": pl.Float32,
    "10u": pl.Float32,
    "10v": pl.Float32,
    "100u": pl.Float32,
    "100v": pl.Float32,
    "cdir": pl.Float32,
    "uvb": pl.Float32,
    "ssr": pl.Float32,
    "ssrc": pl.Float32,
    "str": pl.Float32,
    "strc": pl.Float32,
    "ssrdc": pl.Float32,
    "ssrd": pl.Float32,
    "strdc": pl.Float32,
    "strd": pl.Float32,
    "tsr": pl.Float32,
    "tsrc": pl.Float32,
    "ttr": pl.Float32,
    "ttrc": pl.Float32,
    "fdir": pl.Float32,
    "hcc": pl.Float32,
    "lcc": pl.Float32,
    "mcc": pl.Float32,
    "tcc": pl.Float32,
    "cvh": pl.Float32,
    "lai_hv": pl.Float32,
    "lai_lv": pl.Float32,
    "cvl": pl.Float32,
    "kx": pl.Float32,
    "frac_in_region": pl.Float32,
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

def setup_logging(level: str, rank: int) -> None:
    """
    Configure the root logger; prefix messages with MPI rank.
    """
    fmt = f"%(asctime)s [RANK {rank:03d}] [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
    )


def log_banner(title: str, rank: int, root: int = 0) -> None:
    """
    Pretty ASCII banner. Only printed by root rank.
    """
    if rank != root:
        return
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"{title:^70}")  # centered
    logging.info("=" * 70)
    logging.info("")


def fmt_hms(seconds: float) -> str:
    """Format seconds as HHh:MMm:SSs."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h:{m:02d}m:{s:02d}s"


# ---------------------------------------------------------------------
# METADATA / RENAME MAP
# ---------------------------------------------------------------------

def load_metadata_rename_map(metadata_json: Path) -> Dict[str, str]:
    """
    Load ERA5 variable metadata and build a rename map.

    Mapping goes from ERA5 *shortName* (e.g. "2t") to a processing-friendly name,
    ideally ``datasetProcessingName``; fall back to `fullName` → snake_case.
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
# UTILS: DISCOVER FILES, GROUP BY YEAR, CLEANUP, CAST/DROP SUMMARY
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# DIAGNOSTIC UTILITIES
# ---------------------------------------------------------------------

def print_global_dtype_map(global_dtype_map: Dict[str, pl.DataType]) -> None:
    """
    Pretty-print the GLOBAL_DTYPE_MAP sorted by column name.
    """
    print("\n========== GLOBAL_DTYPE_MAP ==========")
    for col, dtype in sorted(global_dtype_map.items()):
        print(f"{col:<30}  {dtype}")
    print("======================================\n")


def print_schema(file_path: Path, label: str = "SCHEMA") -> None:
    """
    Pretty-print the schema of a Parquet file using polars.
    """
    print(f"\n========== {label}: {file_path.name} ==========")
    try:
        schema = pl.scan_parquet(file_path).collect_schema()
        for col, dtype in schema.items():
            print(f"{col:<30}  {dtype}")
    except Exception as e:
        print(f"[ERROR] Could not read schema: {e}")
    print("==============================================\n")


def diff_schema(
    file_path: Path,
    global_dtype_map: Dict[str, pl.DataType],
    label: str = "SCHEMA DIFF"
) -> None:
    """
    Compare a file's schema to GLOBAL_DTYPE_MAP and show diffs:
        - Missing columns
        - Extra columns
        - Mismatched dtypes
    """
    print(f"\n========== {label}: {file_path.name} ==========")

    try:
        schema = pl.scan_parquet(file_path).collect_schema()
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        print("==============================================\n")
        return

    file_cols = set(schema.names())
    global_cols = set(global_dtype_map.keys())

    missing = global_cols - file_cols
    extra = file_cols - global_cols

    print("\n-- Missing columns (in file but required by global map) --")
    if missing:
        for col in sorted(missing):
            print(f"  {col}")
    else:
        print("  None")

    print("\n-- Extra columns (in file but NOT in global map) --")
    if extra:
        for col in sorted(extra):
            print(f"  {col}")
    else:
        print("  None")

    print("\n-- Dtype mismatches ---------------------------------------")
    mismatch = False
    for col in sorted(global_cols & file_cols):
        gdt = global_dtype_map[col]
        fdt = schema[col]
        if gdt != fdt:
            mismatch = True
            print(f"  {col:<30}  file={fdt:<12}  expected={gdt}")
    if not mismatch:
        print("  None")

    print("============================================================\n")



def discover_monthly_files(
    input_dir: Path,
    prefix: str,
    uid: str,
) -> List[Path]:
    """
    Find all monthly Parquet files belonging to a given prefix+uid group.
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
            len(valid_files), prefix, uid, min_year, max_year,
        )
    else:
        logging.info(
            "Discovered 0 monthly files for group %s_%s",
            prefix, uid,
        )

    return valid_files


def group_by_year(files: List[Path]) -> Dict[int, List[Path]]:
    """
    Group monthly files by year; files within each year sorted by month.
    """
    by_year: Dict[int, List[Tuple[int, Path]]] = {}
    for f in files:
        _, _, year, month = parse_prefix_uid_year_month(f)
        by_year.setdefault(year, []).append((month, f))

    out: Dict[int, List[Path]] = {}
    for year, month_files in by_year.items():
        out[year] = [p for (m, p) in sorted(month_files, key=lambda x: x[0])]
    return out


def cleanup_temp_dirs(clean_dir: Path, agg_dir: Path) -> Dict[Path, int]:
    """
    Delete the entire clean_dir and agg_dir directories,
    but DO NOT delete their parent directories.
    """
    import shutil

    counts: Dict[Path, int] = {}

    if clean_dir.exists():
        file_count = sum(1 for _ in clean_dir.rglob("*"))
        shutil.rmtree(clean_dir)
        counts[clean_dir] = file_count
    else:
        counts[clean_dir] = 0

    if agg_dir.exists():
        file_count = sum(1 for _ in agg_dir.rglob("*"))
        shutil.rmtree(agg_dir)
        counts[agg_dir] = file_count
    else:
        counts[agg_dir] = 0

    return counts


def infer_cast_and_drop_columns(
    monthly_files: List[Path],
    cast_map: Dict[str, pl.DataType],
    drop_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Infer which columns are actually cast and dropped across all monthly files.
    Only used for logging summary.
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
# GLOBAL DTYPE MAP FROM REFERENCE FILE
# ---------------------------------------------------------------------

def build_global_dtype_map(
    sample_file: Path,
    cast_map: Dict[str, pl.DataType],
) -> Dict[str, pl.DataType]:
    """
    Build a global dtype map based on the reference sample file schema,
    with CAST_MAP overriding any selected columns.

    Columns listed in DROP_COLS are excluded from the global map.

    GLOBAL_DTYPE_MAP[col] = CAST_MAP[col]        if col in CAST_MAP
                          = ref_schema[col]      otherwise
    """
    logging.info("Building global dtype map from reference file: %s", sample_file)

    ref_schema = pl.scan_parquet(sample_file).collect_schema()
    drop_set = set(DROP_COLS)

    global_map: Dict[str, pl.DataType] = {}

    for col, dtype in ref_schema.items():
        if col in drop_set:
            continue
        if col in cast_map:
            global_map[col] = cast_map[col]
        else:
            global_map[col] = dtype

    logging.info(
        "Global dtype map constructed with %d columns (%d overrides from CAST_MAP)",
        len(global_map),
        sum(1 for c in ref_schema if c in cast_map),
    )
    return global_map


def validate_schema_against_global(
    schema: pl.Schema,
    global_dtype_map: Dict[str, pl.DataType],
    allow_drop_cols: bool,
    context: str,
) -> None:
    """
    Validate that a file schema matches GLOBAL_DTYPE_MAP.

    If allow_drop_cols=True:
        - DROP_COLS are allowed to appear in the file but are ignored.
        - We require that all GLOBAL_DTYPE_MAP keys are present.
        - Extra columns not in GLOBAL_DTYPE_MAP must be subset of DROP_COLS.

    If allow_drop_cols=False:
        - Columns must exactly equal GLOBAL_DTYPE_MAP keys.
        - No extras, no missing.
    """
    file_cols = set(schema.names())
    global_cols = set(global_dtype_map.keys())
    drop_set = set(DROP_COLS) if allow_drop_cols else set()

    core_cols = file_cols - drop_set

    missing = global_cols - core_cols
    extra = core_cols - global_cols

    if missing or extra:
        raise ValueError(
            f"[{context}] Schema mismatch.\n"
            f"  Missing columns: {sorted(missing) if missing else '[]'}\n"
            f"  Extra columns  : {sorted(extra) if extra else '[]'}"
        )

    # Check dtype compatibility for all global columns
    for col in global_cols:
        if col not in core_cols:
            continue
        file_dtype = schema[col]
        target_dtype = global_dtype_map[col]
        if file_dtype != target_dtype:
            # We allow Stage 1 to correct mismatches via casting;
            # Stage 2 should see them already fixed.
            logging.debug(
                "[%s] Dtype mismatch for column %s: file=%s, global=%s",
                context,
                col,
                file_dtype,
                target_dtype,
            )


# ---------------------------------------------------------------------
# STAGE 1: OPTIMISE MONTHLY FILES → CLEAN_DIR
# ---------------------------------------------------------------------

def build_cast_expr(col: str, current_dtype: pl.DataType, target_dtype: pl.DataType) -> pl.Expr:
    """
    Robust cast expression that handles numeric, Utf8, null, and mixed types.
    Only applies string cleanup if needed.
    """
    if current_dtype == target_dtype:
        return pl.col(col)

    if current_dtype is not None and current_dtype.is_numeric():
        return pl.col(col).cast(target_dtype, strict=False).alias(col)

    if current_dtype == pl.Utf8:
        return (
            pl.col(col)
            .str.replace_all(r"[^\d\.\-eE]", "")   # strip junk
            .str.replace_all(r"^\s*$", None)       # blank -> null
            .cast(target_dtype, strict=False)
            .alias(col)
        )

    # Fallback: best-effort cast
    return pl.col(col).cast(target_dtype, strict=False).alias(col)


def optimise_monthly_file(
    path: Path,
    clean_dir: Path,
    overwrite: bool,
    global_dtype_map: Dict[str, pl.DataType],
) -> Tuple[Path | None, FileStats | None]:
    """
    Optimise a single monthly Parquet file:

      - Validate raw schema against GLOBAL_DTYPE_MAP (allowing DROP_COLS).
      - Filter timestamps to match filename year+month.
      - Drop DROP_COLS.
      - Validate resulting schema matches GLOBAL_DTYPE_MAP exactly.
      - Cast columns where dtype != GLOBAL_DTYPE_MAP[col].
      - Write to clean_dir.

    On schema mismatch (missing/extra columns), a ValueError is raised
    (as per user's choice A: fail fast instead of silently patching).
    """
    clean_path = clean_dir / path.name
    if clean_path.exists() and not overwrite:
        logging.info("[STAGE1] SKIP (exists): %s", clean_path.name)
        return clean_path, None

    start_wall = time.time()
    start_cpu = time.process_time()

    lf = pl.scan_parquet(path)
    raw_schema = lf.collect_schema()

    # Validate schema before any dropping (allow DROP_COLS to exist)
    validate_schema_against_global(
        raw_schema,
        global_dtype_map,
        allow_drop_cols=True,
        context=f"STAGE1_PRE_DROP:{path.name}",
    )

    # Determine expected year/month from filename
    _, _, file_year, file_month = parse_prefix_uid_year_month(path)

    names = raw_schema.names()
    if "valid_time" in names:
        time_col = "valid_time"
    elif "time" in names:
        time_col = "time"
    else:
        time_col = None

    # Filter OUT-OF-BOUNDS timestamps
    if time_col:
        logging.info(
            f"[STAGE1] Filtering by year={file_year}, month={file_month:02d} "
            f"for {path.name} using {time_col}"
        )
        lf = lf.filter(
            (pl.col(time_col).dt.year() == file_year)
            & (pl.col(time_col).dt.month() == file_month)
        )

    # Drop unwanted columns
    drop_cols = [c for c in DROP_COLS if c in names]
    if drop_cols:
        logging.info(f"[STAGE1] Dropping columns: {drop_cols}")
        lf = lf.drop(drop_cols)

    # Validate schema after dropping: MUST match global map exactly
    schema_after = lf.collect_schema()
    validate_schema_against_global(
        schema_after,
        global_dtype_map,
        allow_drop_cols=False,
        context=f"STAGE1_POST_DROP:{path.name}",
    )

    # Cast columns to GLOBAL_DTYPE_MAP where needed
    cast_exprs: List[pl.Expr] = []
    for col, target_dtype in global_dtype_map.items():
        current_dtype = schema_after[col]
        if current_dtype != target_dtype:
            logging.info(
                f"[STAGE1] Casting {col}: {current_dtype} → {target_dtype}"
            )
            cast_exprs.append(build_cast_expr(col, current_dtype, target_dtype))

    if cast_exprs:
        lf = lf.with_columns(cast_exprs)

    # Write cleaned output
    clean_dir.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(clean_path, compression="snappy", statistics=True)

    # Row count
    row_count = (
        pl.scan_parquet(clean_path)
        .select(pl.len().alias("n"))
        .collect()
        .item()
    )

    cpu = time.process_time() - start_cpu
    wall = time.time() - start_wall

    stat = FileStats(
        stage="stage1_optimise",
        name=clean_path.name,
        cpu_time_sec=cpu,
        wall_time_sec=wall,
        row_count=row_count,
        year=file_year,
    )

    logging.info(
        "[STAGE1] OK: %-40s | rows=%d | CPU=%.2fs | WALL=%.2fs",
        clean_path.name,
        row_count,
        cpu,
        wall,
    )

    return clean_path, stat


# ---------------------------------------------------------------------
# STAGE 2: CONSOLIDATE CLEANED FILES → AGG_DIR
# ---------------------------------------------------------------------

def consolidate_year(
    year: int,
    cleaned_files: List[Path],
    prefix: str,
    uid: str,
    modes: List[str],
    agg_dir: Path,
    overwrite: bool,
    global_dtype_map: Dict[str, pl.DataType],
) -> Tuple[List[Path], List[FileStats]]:
    """
    Consolidate one year's cleaned monthly files into annual / biannual / quarterly files.

    Assumes that all cleaned files already:
      - Have identical columns = GLOBAL_DTYPE_MAP.keys()
      - Have correct dtypes per GLOBAL_DTYPE_MAP

    Stage 2 therefore:
      - (Optionally) validates schema vs GLOBAL_DTYPE_MAP
      - Selects columns in sorted(GLOBAL_DTYPE_MAP.keys())
      - Concatenates vertically
      - Writes aggregated files
    """
    outputs: List[Path] = []
    stats: List[FileStats] = []

    if not cleaned_files:
        logging.warning(f"[STAGE2] Year {year}: no cleaned files available.")
        return outputs, stats

    agg_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"[STAGE2] Year {year}: consolidating {len(cleaned_files)} cleaned monthly files "
        f"using global schema with {len(global_dtype_map)} columns."
    )

    def mode_groups(mode: str) -> List[List[Path]]:
        if mode == "annual":
            return [cleaned_files]

        elif mode == "biannual":
            h1 = [f for f in cleaned_files if parse_prefix_uid_year_month(f)[3] <= 6]
            h2 = [f for f in cleaned_files if parse_prefix_uid_year_month(f)[3] > 6]
            return [h1, h2]

        elif mode == "quarterly":
            groups: List[List[Path]] = []
            for q in range(4):
                low = 1 + 3 * q
                high = low + 2
                groups.append(
                    [
                        f
                        for f in cleaned_files
                        if low <= parse_prefix_uid_year_month(f)[3] <= high
                    ]
                )
            return groups

        else:
            raise ValueError(f"Unknown consolidation mode: {mode}")

    col_order = sorted(global_dtype_map.keys())

    for mode in modes:
        groups = mode_groups(mode)

        for idx, file_list in enumerate(groups, start=1):
            if not file_list:
                logging.info(
                    f"[STAGE2] Year {year}, mode {mode}, part {idx}: empty group (skipped)."
                )
                continue

            # Output filename
            if mode == "annual":
                out_name = f"{prefix}_{uid}_{year}.parquet"
            elif mode == "biannual":
                part = "H1" if idx == 1 else "H2"
                out_name = f"{prefix}_{uid}_{year}_{part}.parquet"
            else:  # quarterly
                out_name = f"{prefix}_{uid}_{year}_Q{idx}.parquet"

            out_path = agg_dir / out_name

            if out_path.exists() and not overwrite:
                logging.info(f"[STAGE2] SKIP (exists): {out_name}")
                outputs.append(out_path)
                continue

            logging.info(
                f"[STAGE2] Year {year}, mode {mode}, part {idx}: "
                f"merging {len(file_list)} files → {out_name}"
            )

            start_wall = time.time()
            start_cpu = time.process_time()

            lfs: List[pl.LazyFrame] = []
            for f in file_list:
                lf = pl.scan_parquet(f)
                # Optional safety check
                schema = lf.collect_schema()
                validate_schema_against_global(
                    schema,
                    global_dtype_map,
                    allow_drop_cols=False,
                    context=f"STAGE2:{f.name}",
                )
                lf = lf.select(col_order)
                lfs.append(lf)

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
                    name=out_name,
                    cpu_time_sec=cpu,
                    wall_time_sec=wall,
                    row_count=row_count,
                    year=year,
                )
            )

            logging.info(
                f"[STAGE2] OK {out_name:40s} | files={len(file_list):2d} "
                f"| rows={row_count} | CPU={cpu:.2f}s | WALL={wall:.2f}s"
            )

            outputs.append(out_path)

    return outputs, stats


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
            "[STAGE3] OK: %-40s | rows=%d | CPU=%.2fs | WALL=%.2fs",
            out_path.name,
            row_count,
            cpu,
            wall,
        )

        return out_path, stats

    except Exception as e:  # noqa: BLE001
        logging.error("[STAGE3] FAIL %s: %s", agg_path.name, e)
        return None, None


# ---------------------------------------------------------------------
# ARGPARSE / CONTROLLER
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "ERA5 MPI pipeline: optimise monthly files, consolidate, and rename columns "
            "for a single download group inferred from a sample file. "
            "Parallelised over YEARS using MPI ranks."
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
        help="Delete intermediate cleaned and aggregated directories after success.",
    )

    return parser


# ---------------------------------------------------------------------
# MAIN (MPI ORCHESTRATION)
# ---------------------------------------------------------------------

def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = build_arg_parser().parse_args()
    setup_logging(args.log_level, rank)

    overall_wall_start = time.time()
    overall_cpu_start = time.process_time()

    # --------------------------------------------------------------
    # PHASE 0 — DISCOVERY & BROADCAST (Rank 0 only does work)
    # --------------------------------------------------------------
    prefix: str | None = None
    uid: str | None = None
    monthly_files: List[Path] | None = None
    year_map_monthly: Dict[int, List[Path]] | None = None
    years_sorted: List[int] | None = None
    meta_rename: Dict[str, str] | None = None
    global_dtype_map: Dict[str, pl.DataType] | None = None

    # Ensure dirs exist on all ranks
    for d in (args.input_dir, args.clean_dir, args.agg_dir, args.output_dir):
        d.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        log_banner("STARTING [step3a_optimise_and_consolidate.py]", rank)

        logging.info("Running script with configuration:")
        for arg, value in vars(args).items():
            logging.info("  %s = %r", arg, value)

        sample = args.sample_file
        if not sample.exists():
            logging.error("Sample file does not exist: %s", sample)
            comm.Abort(1)

        prefix, uid, _, _ = parse_prefix_uid_year_month(sample)

        monthly_files = discover_monthly_files(args.input_dir, prefix, uid)
        if not monthly_files:
            logging.error("No monthly files found for group %s_%s", prefix, uid)
            comm.Abort(1)

        year_map_monthly = group_by_year(monthly_files)
        years_sorted = sorted(year_map_monthly.keys())

        meta_rename = load_metadata_rename_map(args.metadata_json)
        logging.info(
            "Loaded metadata rename map from %s (%d entries)",
            args.metadata_json,
            len(meta_rename),
        )

        # Log year → file layout
        logging.info("Monthly files by year:")
        for y in years_sorted:
            files_y = [p.name for p in year_map_monthly[y]]
            logging.info("  %d → %s", y, ", ".join(files_y))

        # Build GLOBAL_DTYPE_MAP from reference file schema + CAST_MAP overrides
        global_dtype_map = build_global_dtype_map(sample, CAST_MAP)
        print_global_dtype_map(global_dtype_map)

    # Broadcast plan + GLOBAL_DTYPE_MAP to all ranks
    prefix = comm.bcast(prefix, root=0)
    uid = comm.bcast(uid, root=0)
    monthly_files = comm.bcast(monthly_files, root=0)
    year_map_monthly = comm.bcast(year_map_monthly, root=0)
    years_sorted = comm.bcast(years_sorted, root=0)
    meta_rename = comm.bcast(meta_rename, root=0)
    global_dtype_map = comm.bcast(global_dtype_map, root=0)

    # Safety: type asserts
    assert prefix is not None and uid is not None
    assert monthly_files is not None
    assert year_map_monthly is not None
    assert years_sorted is not None
    assert meta_rename is not None
    assert global_dtype_map is not None

    total_monthly_files = len(monthly_files)

    # Compute year assignment for each rank
    def years_for_rank(r: int) -> List[int]:
        return [y for i, y in enumerate(years_sorted) if i % size == r]

    my_years = years_for_rank(rank)

    if rank == 0:
        logging.info("Year → rank assignment:")
        for r in range(size):
            ys = years_for_rank(r)
            ys_str = ", ".join(str(y) for y in ys) if ys else "(none)"
            logging.info("  Rank %d → %s", r, ys_str)

    comm.Barrier()  # End of PHASE 0

    # --------------------------------------------------------------
    # STAGE 1 — Optimise monthly files (per-year, per-rank)
    # --------------------------------------------------------------
    stage1_wall_start = time.time()
    stage1_cpu_start = time.process_time()

    stats1_local: List[FileStats] = []
    cleaned_by_year_local: Dict[int, List[Path]] = {}

    try:
        for year in my_years:
            files_for_year = year_map_monthly.get(year, [])
            for path in files_for_year:
                clean_path, stat = optimise_monthly_file(
                    path=path,
                    clean_dir=args.clean_dir,
                    overwrite=args.overwrite,
                    global_dtype_map=global_dtype_map,
                )
                if clean_path is not None:
                    cleaned_by_year_local.setdefault(year, []).append(clean_path)
                if stat is not None:
                    stat.year = year
                    stats1_local.append(stat)
    except Exception as e:
        logging.error("Fatal error in STAGE 1 on rank %d: %s", rank, e)
        comm.Abort(1)

    stage1_wall_local = time.time() - stage1_wall_start
    stage1_cpu_local = time.process_time() - stage1_cpu_start

    # Gather Stage 1 results
    all_stats1 = comm.gather(stats1_local, root=0)
    all_cleaned_maps = comm.gather(cleaned_by_year_local, root=0)

    # Reduce timing across ranks
    stage1_wall_max = comm.reduce(stage1_wall_local, op=MPI.MAX, root=0)
    stage1_cpu_sum = comm.reduce(stage1_cpu_local, op=MPI.SUM, root=0)

    cleaned_by_year_global: Dict[int, List[Path]] = {}

    if rank == 0:
        # Merge cleaned maps
        for mp in all_cleaned_maps:
            for y, flist in mp.items():
                cleaned_by_year_global.setdefault(y, []).extend(flist)

        # Flatten stats
        stats1_flat = [s for lst in all_stats1 for s in lst]

        # Column cast/drop summary (based on original monthly files)
        cast_cols, drop_cols = infer_cast_and_drop_columns(
            monthly_files,
            CAST_MAP,
            DROP_COLS,
        )

        log_banner("STAGE 1 — File Optimisation Complete", rank)
        logging.info("  Files processed        : %d", len(stats1_flat))
        logging.info(
            "  Cast [%d] columns       : %s",
            len(cast_cols),
            ", ".join(cast_cols) if cast_cols else "(none)",
        )
        logging.info(
            "  Dropped [%d] columns    : %s",
            len(drop_cols),
            ", ".join(drop_cols) if drop_cols else "(none)",
        )
        logging.info(
            "  TRUE_WALL (max over ranks): %s", fmt_hms(stage1_wall_max)
        )
        logging.info(
            "  CPU_SUM (sum over ranks)  : %s", fmt_hms(stage1_cpu_sum)
        )
    else:
        stats1_flat = None

    cleaned_by_year_global = comm.bcast(cleaned_by_year_global, root=0)

    comm.Barrier()  # End of STAGE 1

    # --------------------------------------------------------------
    # STAGE 2 — Consolidate cleaned files
    # --------------------------------------------------------------
    stage2_wall_start = time.time()
    stage2_cpu_start = time.process_time()

    stats2_local: List[FileStats] = []
    agg_by_year_local: Dict[int, List[Path]] = {}

    try:
        for year in my_years:
            files_for_year = cleaned_by_year_global.get(year, [])
            out_files, year_stats = consolidate_year(
                year=year,
                cleaned_files=files_for_year,
                prefix=prefix,
                uid=uid,
                modes=args.modes,
                agg_dir=args.agg_dir,
                overwrite=args.overwrite,
                global_dtype_map=global_dtype_map,
            )
            agg_by_year_local[year] = out_files
            stats2_local.extend(year_stats)
    except Exception as e:
        logging.error("Fatal error in STAGE 2 on rank %d: %s", rank, e)
        comm.Abort(1)

    stage2_wall_local = time.time() - stage2_wall_start
    stage2_cpu_local = time.process_time() - stage2_cpu_start

    # Gather Stage 2 results
    all_stats2 = comm.gather(stats2_local, root=0)
    all_agg_maps = comm.gather(agg_by_year_local, root=0)

    # Reduce timings
    stage2_wall_max = comm.reduce(stage2_wall_local, op=MPI.MAX, root=0)
    stage2_cpu_sum = comm.reduce(stage2_cpu_local, op=MPI.SUM, root=0)

    agg_by_year_global: Dict[int, List[Path]] = {}

    if rank == 0:
        for mp in all_agg_maps:
            for y, flist in mp.items():
                agg_by_year_global.setdefault(y, []).extend(flist)

        stats2_flat = [s for lst in all_stats2 for s in lst]

        # Build mode/year summary
        by_mode_year: Dict[str, Dict[int, List[str]]] = {}
        for s in stats2_flat:
            mode = s.stage.replace("stage2_consolidate_", "").upper()
            if s.year is None:
                continue
            by_mode_year.setdefault(mode, {}).setdefault(s.year, []).append(s.name)

        log_banner("STAGE 2 — Completed File Consolidation", rank)
        for mode in sorted(by_mode_year.keys()):
            logging.info("  %s", mode)
            for y in sorted(by_mode_year[mode].keys()):
                files_for_year = sorted(by_mode_year[mode][y])
                logging.info("    [%d] → %s", y, ", ".join(files_for_year))
        logging.info(
            "  TRUE_WALL (max over ranks): %s", fmt_hms(stage2_wall_max)
        )
        logging.info(
            "  CPU_SUM (sum over ranks)  : %s", fmt_hms(stage2_cpu_sum)
        )
    else:
        stats2_flat = None

    agg_by_year_global = comm.bcast(agg_by_year_global, root=0)

    comm.Barrier()  # End of STAGE 2

    # --------------------------------------------------------------
    # STAGE 3 — Rename columns
    # --------------------------------------------------------------
    stage3_wall_start = time.time()
    stage3_cpu_start = time.process_time()

    stats3_local: List[FileStats] = []

    for year in my_years:
        agg_files = agg_by_year_global.get(year, [])
        for agg_path in agg_files:
            _, stat = rename_columns_file(
                agg_path=agg_path,
                output_dir=args.output_dir,
                meta_rename=meta_rename,
                manual_rename=MANUAL_RENAME_MAP,
                overwrite=args.overwrite,
            )
            if stat is not None:
                stat.year = year
                stats3_local.append(stat)

    stage3_wall_local = time.time() - stage3_wall_start
    stage3_cpu_local = time.process_time() - time.process_time() + stage3_cpu_start

    all_stats3 = comm.gather(stats3_local, root=0)
    stage3_wall_max = comm.reduce(stage3_wall_local, op=MPI.MAX, root=0)
    stage3_cpu_sum = comm.reduce(stage3_cpu_local, op=MPI.SUM, root=0)

    if rank == 0:
        if rank == 0 and agg_path.name.endswith("2025.parquet"):
            print_schema(agg_path, label="AGG FILE SCHEMA")
            diff_schema(agg_path, global_dtype_map)

        stats3_flat = [s for lst in all_stats3 for s in lst]

        log_banner("STAGE 3 — Column Renaming Complete", rank)
        logging.info("  Files renamed   : %d", len(stats3_flat))
        logging.info("  Output directory: %s", args.output_dir)
        logging.info(
            "  Mapping based on `datasetProcessingName` from metadata: %s",
            args.metadata_json,
        )

        renamed_files = [s.name for s in stats3_flat]
        for name in sorted(renamed_files):
            logging.info("    - %s", name)

        logging.info(
            "  TRUE_WALL (max over ranks): %s", fmt_hms(stage3_wall_max)
        )
        logging.info(
            "  CPU_SUM (sum over ranks)  : %s", fmt_hms(stage3_cpu_sum)
        )

    comm.Barrier()  # End of STAGE 3

    # --------------------------------------------------------------
    # STAGE 4 — Cleanup (optional, rank 0 only)
    # --------------------------------------------------------------
    if args.cleanup_temp and rank == 0:
        log_banner("STAGE 4 — Cleanup of Temporary Directories", rank)
        counts = cleanup_temp_dirs(args.clean_dir, args.agg_dir)
        total_deleted = sum(counts.values())
        logging.info("  Total files deleted: %d", total_deleted)
        for d, n in counts.items():
            logging.info("    • %s → deleted %d file(s)", d, n)

    comm.Barrier()  # End of STAGE 4

    # --------------------------------------------------------------
    # GLOBAL PIPELINE SUMMARY (Rank 0)
    # --------------------------------------------------------------
    wall_total_local = time.time() - overall_wall_start
    cpu_total_local = time.process_time() - overall_cpu_start

    wall_total_max = comm.reduce(wall_total_local, op=MPI.MAX, root=0)
    cpu_total_sum = comm.reduce(cpu_total_local, op=MPI.SUM, root=0)

    if rank == 0:
        stats1_flat = [s for lst in all_stats1 for s in lst]
        stats2_flat = [s for lst in all_stats2 for s in lst]
        stats3_flat = [s for lst in all_stats3 for s in lst]

        log_banner("PIPELINE COMPLETE — SUMMARY", rank)

        group_key = f"{prefix}_{uid}"
        total_outputs = len(stats3_flat)

        logging.info("  Grouping key(s)           : %s", group_key)
        logging.info("  Monthly files processed   : %d", total_monthly_files)
        logging.info("  Final output files        : %d", total_outputs)
        logging.info("  Consolidation modes       : %s", ", ".join(args.modes))
        logging.info("  MPI ranks (processes)     : %d", size)
        logging.info("")

        logging.info("  TIMINGS")
        logging.info("  --------------------------------------------------")
        avg_per_file = wall_total_max / max(total_monthly_files, 1)
        logging.info("  Average per file          : %s", fmt_hms(avg_per_file))
        logging.info("  CPU total time (sum ranks): %s", fmt_hms(cpu_total_sum))
        logging.info("  TOTAL WALL TIME (max rank): %s", fmt_hms(wall_total_max))
        logging.info("  --------------------------------------------------")
        logging.info("")

        def stage_summary(
            label: str,
            true_wall: float,
            true_cpu: float,
            stats: List[FileStats],
        ) -> None:
            worker_wall = sum(s.wall_time_sec for s in stats)
            worker_cpu = sum(s.cpu_time_sec for s in stats)
            logging.info(
                f"  {label:<8} | TRUE_WALL={fmt_hms(true_wall)} | "
                f"TRUE_CPU_SUM={fmt_hms(true_cpu)} | "
                f"WORKER_WALL_SUM={fmt_hms(worker_wall)} | "
                f"WORKER_CPU_SUM={fmt_hms(worker_cpu)} | "
                f"files={len(stats):3d}"
            )

        logging.info("  PER-STAGE BREAKDOWN")
        logging.info("  --------------------------------------------------")
        stage_summary("STAGE1", stage1_wall_max, stage1_cpu_sum, stats1_flat)
        stage_summary("STAGE2", stage2_wall_max, stage2_cpu_sum, stats2_flat)
        stage_summary("STAGE3", stage3_wall_max, stage3_cpu_sum, stats3_flat)
        logging.info("  --------------------------------------------------")
        logging.info("  Processing completed.")
        logging.info("")


if __name__ == "__main__":
    main()
