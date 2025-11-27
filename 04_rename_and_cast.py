#!/usr/bin/env python3
import json
from pathlib import Path
import polars as pl
import logging

# ============================================================================
# CONFIG
# ============================================================================

INPUT_DIR   = Path("../data/interim")
TEMP_DIR    = Path("../data/temp_clean")
OUTPUT_DIR  = Path("../data/processed")

TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

METADATA_JSON = Path("era5-world_variable_metadata.json")

DROP_COLS = []

MANUAL_RENAME_MAP = {}

CAST_MAP = {
    "2t": pl.Float32,
    "2d": pl.Float32,
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
}

# Load metadata → rename mapping
with open(METADATA_JSON) as f:
    META = json.load(f)

META_RENAME = {
    short: META[short]["fullName"].lower().replace(" ", "_")
    for short in META
}

# ============================================================================
# SAFE CASTING BUILDER
# ============================================================================

def build_cast_expr(col: str, schema: dict, target_dtype: pl.DataType):
    current_dtype = schema.get(col)

    # Utf8 → strip junk → cast
    if current_dtype == pl.Utf8:
        return (
            pl.col(col)
            .str.replace_all(r"[^\d\.\-eE]", "")
            .str.replace_all(r"^\s*$", None)
            .cast(target_dtype, strict=False)
            .alias(col)
        )

    # Numeric → direct cast
    if current_dtype and current_dtype.is_numeric():
        return pl.col(col).cast(target_dtype, strict=False).alias(col)

    # Fallback
    return pl.col(col).cast(target_dtype, strict=False).alias(col)

# ============================================================================
# PASS 1 — CAST ONLY
# ============================================================================

def pass1_cast_only(path: Path):
    tmp_path = TEMP_DIR / f"{path.stem}.tmp.parquet"

    try:
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()  # dict: column → dtype
        names = schema.names()

        # drop columns
        drop_cols = [c for c in DROP_COLS if c in names]
        lf = lf.drop(drop_cols)

        # build cast expressions
        exprs = []
        for col, dtype in CAST_MAP.items():
            if col in names:
                exprs.append(build_cast_expr(col, schema, dtype))

        if exprs:
            lf = lf.with_columns(exprs)

        lf.sink_parquet(tmp_path)
        logging.info(f"PASS1 OK: {path.name}")
        return tmp_path

    except Exception as e:
        logging.error(f"PASS1 FAIL: {path.name}: {e}")
        return None

# ============================================================================
# PASS 2 — RENAME ONLY
# ============================================================================

def pass2_rename_only(tmp_path: Path):
    out_path = OUTPUT_DIR / tmp_path.name.replace(".tmp", "")

    try:
        lf = pl.scan_parquet(tmp_path)
        schema = lf.collect_schema()
        names = schema.names()

        rename_map = {}

        for old in names:
            if old in META_RENAME:
                rename_map[old] = META_RENAME[old]
            elif old in MANUAL_RENAME_MAP:
                rename_map[old] = MANUAL_RENAME_MAP[old]

        if rename_map:
            lf = lf.rename(rename_map)

        lf.sink_parquet(out_path)
        tmp_path.unlink(missing_ok=True)

        logging.info(f"PASS2 OK: {out_path.name}")
        return out_path

    except Exception as e:
        logging.error(f"PASS2 FAIL: {tmp_path.name}: {e}")
        return None

# ============================================================================
# MAIN DRIVER
# ============================================================================

def run_cleaning():
    files = sorted(INPUT_DIR.glob("*.parquet"))
    logging.info(f"Found {len(files)} files")

    tmp_files = [t for f in files if (t := pass1_cast_only(f))]
    for t in tmp_files:
        pass2_rename_only(t)

    logging.info("Cleaning pipeline complete.")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cleaning()
