
"""
step2c_update_metadata_with_processing_names.py
===============================================

Add datasetProcessingName to ECMWF-style variable metadata JSON.

This script reads a JSON file containing ECMWF variable metadata
(e.g., "fullName", "shortName", "paramId"), derives a new
``datasetProcessingName`` for each entry using deterministic rules,
and writes the updated metadata to a new JSON file.

Default behavior
----------------
If no CLI arguments are given, the script uses:

    INPUT_JSON  = Path("era5_variable_metadata.json")
    OUTPUT_JSON = Path("era5_variable_metadata_with_processing_names.json")

You may override these with:

    --input  path/to/input.json
    --output path/to/output.json

Transformation rules for ``datasetProcessingName``
--------------------------------------------------
1. Convert the string to lowercase.
2. Replace the word "metre" with "m".
3. Replace commas (",") with hyphens ("-").
4. Replace spaces:
   - If preceded by a digit → removed completely.
   - Otherwise → replaced with underscore "_".

Examples
--------
>>> make_processing_name("2 metre temperature")
'2m_temperature'

>>> make_processing_name("100 metre U wind component")
'100m_u_wind_component'

Usage
-----
Default input/output:

    python add_processing_names.py

Explicit:

    python add_processing_names.py --input meta.json --output meta_out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


# ---------------------------------------------------------------------
# Default file paths (manual specification retained)
# ---------------------------------------------------------------------

INPUT_JSON = Path("../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json")
OUTPUT_JSON = Path("../data/interim/era5-world_N37W68S6E98_d514a3a3c256_metadata.json")

# ---------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------

def make_processing_name(full_name: str) -> str:
    """
    Convert a variable's ``fullName`` into a datasetProcessingName.

    Parameters
    ----------
    full_name : str
        The raw "fullName" text from the metadata.

    Returns
    -------
    str
        Sanitized machine-friendly name following the rules:

        - lowercase
        - "metre" → "m"
        - commas → hyphens
        - spaces → underscores, unless previous character is numeric

    Examples
    --------
    >>> make_processing_name("2 metre temperature")
    '2m_temperature'

    >>> make_processing_name("100 metre U wind component")
    '100m_u_wind_component'
    """
    if not full_name:
        return ""

    name = full_name.lower().strip()
    name = name.replace("metre", "m")
    name = name.replace(",", "-")

    out_chars = []
    for ch in name:
        if ch == " ":
            prev = out_chars[-1] if out_chars else ""
            if prev.isdigit():
                # Remove space when preceded by a digit
                continue
            out_chars.append("_")
        else:
            out_chars.append(ch)

    return "".join(out_chars)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.
    """
    return json.loads(path.read_text())


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Save a JSON-serializable object to disk with indentation.

    Parameters
    ----------
    path : Path
        Output JSON file path.
    data : dict
        The JSON content to write.
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def process_metadata(input_path: Path, output_path: Path) -> None:
    """
    Add ``datasetProcessingName`` to each variable in the JSON metadata.

    Parameters
    ----------
    input_path : Path
        Path to the source metadata JSON file.

    output_path : Path
        Path where the transformed JSON will be written.

    Returns
    -------
    None
    """
    meta = load_json(input_path)
    updated: Dict[str, Any] = {}

    for key, entry in meta.items():
        full_name = entry.get("fullName", "")
        proc_name = make_processing_name(full_name)

        new_entry = dict(entry)
        new_entry["datasetProcessingName"] = proc_name

        updated[key] = new_entry

    save_json(output_path, updated)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Add datasetProcessingName fields to ECMWF metadata JSON."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_JSON,
        help=f"Input metadata JSON file (default: {INPUT_JSON})",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_JSON,
        help=f"Output JSON file (default: {OUTPUT_JSON})",
    )

    return parser


def main() -> None:
    """
    Execute the program.

    Steps
    -----
    1. Parse CLI arguments or use default paths.
    2. Process the JSON metadata.
    3. Write a new output file containing datasetProcessingName fields.
    """
    args = build_arg_parser().parse_args()
    process_metadata(args.input, args.output)
    print(f"[OK] Created datasetProcessingName → {args.output}")


if __name__ == "__main__":
    main()
