"""
step1c_extract_and_validate_variable_metadata.py
============================================

Automatically scan GRIB files in a directory, group them by unique download
signature (dataset, coordinates, uid), validate variable consistency within
each group, and extract enriched metadata (Selenium from ECMWF parameter DB).

User Input
----------
The user only needs to specify:
    --grib-dir /path/to/grib/files

Optionally:
    --dataset <value>
    --coordinates <value>
    --uid <value>

If optional filters are provided, only matching download groups are processed.

Filename Format Assumed
-----------------------
GRIB filenames must match the pattern:

    <dataset>_<coordinates>_<uid>_<year>_<month>.grib

Example:
    era5-world_N37W68S6E98_d514a3a3c256_2025_04.grib

Grouping Logic
--------------
All GRIB files sharing the same triple:
    (dataset, coordinates, uid)
belong to the same download batch and should be processed together.

Output
------
For each group, a JSON metadata file is produced:

    <dataset>_<coordinates>_<uid>_metadata.json

Containing:
    shortName → {
        paramId,
        shortName,
        fullName,
        units,
        description,
        url
    }

Requirements
------------
eccodes
selenium
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

URL_BASE = "https://codes.ecmwf.int/grib/param-db"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging with timestamps.

    Parameters
    ----------
    level : int, optional
        Logging level to use (default is ``logging.INFO``).

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"""
    ^(?P<dataset>[A-Za-z0-9\-]+)_         # dataset or dataset-region
    (?P<coordinates>[A-Za-z0-9]+)_        # coordinates like N37W68S6E98
    (?P<uid>[A-Za-z0-9]+)_                # UID (hash-like)
    (?P<year>\d{4})_                       # YYYY
    (?P<month>\d{2})\.grib$               # MM
    """,
    re.VERBOSE,
)


def parse_grib_filename(path: Path) -> Dict[str, str]:
    """
    Parse ERA5-style GRIB filename.

    Parameters
    ----------
    path : Path
        GRIB filename.

    Returns
    -------
    dict
        Contains ``dataset``, ``coordinates``, ``uid``, ``year``, ``month``.

    Raises
    ------
    ValueError
        If the filename pattern does not match.
    """
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        raise ValueError(
            f"Filename '{path.name}' does not match expected pattern: "
            "<dataset>_<coordinates>_<uid>_<year>_<month>.grib"
        )
    return m.groupdict()


def group_files_by_download(files: List[Path]) -> Dict[Tuple[str, str, str], List[Path]]:
    """
    Group GRIB files by (dataset, coordinates, uid).

    Parameters
    ----------
    files : list of Path

    Returns
    -------
    dict
        Mapping:
            (dataset, coordinates, uid) → [list of GRIB files]
    """
    groups = defaultdict(list)
    for f in files:
        tokens = parse_grib_filename(f)
        key = (tokens["dataset"], tokens["coordinates"], tokens["uid"])
        groups[key].append(f)
    return groups


def make_metadata_filename(key: Tuple[str, str, str]) -> str:
    """
    Build metadata filename based on download key.

    Parameters
    ----------
    key : tuple
        (dataset, coordinates, uid)

    Returns
    -------
    str
        Example:
        era5-world_N37W68S6E98_d514a3a3c256_metadata.json
    """
    dataset, coordinates, uid = key
    return f"{dataset}_{coordinates}_{uid}_metadata.json"


# ---------------------------------------------------------------------
# GRIB Variable Scanning
# ---------------------------------------------------------------------

def scan_variables_in_file(path: Path) -> Set[str]:
    """
    Extract unique GRIB ``shortName`` values from a file.
    """
    variables = set()

    try:
        with open(path, "rb") as f:
            while True:
                try:
                    gid = codes_grib_new_from_file(f)
                except CodesInternalError:
                    break
                if gid is None:
                    break

                try:
                    short = codes_get(gid, "shortName")
                    if isinstance(short, bytes):
                        short = short.decode("utf-8")
                    variables.add(str(short))
                finally:
                    codes_release(gid)
    except Exception as e:
        logging.error("Error scanning variables in %s: %s", path, e)

    return variables


def scan_parameters_in_file(path: Path) -> List[Dict]:
    """
    Extract unique {paramId, shortName} entries from a GRIB file.
    """
    params = {}

    try:
        with open(path, "rb") as f:
            while True:
                try:
                    gid = codes_grib_new_from_file(f)
                except CodesInternalError:
                    break
                if gid is None:
                    break

                try:
                    short = codes_get(gid, "shortName")
                    if isinstance(short, bytes):
                        short = short.decode("utf-8")
                    pid = int(codes_get(gid, "paramId"))
                    params[pid] = {"paramId": pid, "shortName": str(short)}
                finally:
                    codes_release(gid)
    except Exception as e:
        logging.error("Error scanning parameters in %s: %s", path, e)

    return list(params.values())


# ---------------------------------------------------------------------
# Consistency Checking
# ---------------------------------------------------------------------

def verify_variables_across_files(
        files: List[Path],
        reference_vars: Set[str]
        ) -> None:
    """
    Check that the variable set is consistent within a GRIB download group.

    Parameters
    ----------
    files : list of Path
        GRIB files in the download group.
    reference_vars : set of str
        Expected set of variable shortNames.

    Returns
    -------
    None
    """
    mismatches = []
    for path in files:
        vars_this = scan_variables_in_file(path)
        if vars_this != reference_vars:
            mismatches.append((path, vars_this))

    if not mismatches:
        logging.info("All files in this download share the same variable set.")
        return

    logging.warning("Variable inconsistencies detected within this download:")
    for path, vars_this in mismatches:
        missing = reference_vars - vars_this
        extra = vars_this - reference_vars
        logging.warning("File: %s", path.name)
        if missing:
            logging.warning("  Missing: %s", sorted(missing))
        if extra:
            logging.warning("  Extra: %s", sorted(extra))


# ---------------------------------------------------------------------
# Selenium Metadata Scraping
# ---------------------------------------------------------------------

def setup_driver() -> webdriver.Chrome:
    """
    Create a headless Chrome Selenium driver.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_parameter_details_selenium(
        param_id: int,
        driver: webdriver.Chrome
        ) -> Dict:
    """
    Scrape ECMWF param database for a parameter's metadata.

    Parameters
    ----------
    param_id : int
        The GRIB parameter ID.
    driver : webdriver.Chrome
        Selenium Chrome driver.

    Returns
    -------
    dict
        Contains 'name', 'unit', 'description' fields.

    """
    url = f"{URL_BASE}/{param_id}/"
    details = {"name": "", "unit": "", "description": ""}

    try:
        driver.get(url)

        # Wait for metadata table
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//td[contains(., 'Name')]")
            )
        )

        # Extract fields (each inside <td><p>...</p></td>)
        def safe_extract(xpath):
            try:
                cell = driver.find_element(By.XPATH, xpath)
                return cell.find_element(By.TAG_NAME, "p").text
            except Exception:
                return ""

        details["name"] = safe_extract("//td[p[contains(., 'Name')]]/following-sibling::td")
        details["unit"] = safe_extract("//td[p[contains(., 'Unit')]]/following-sibling::td")
        details["description"] = safe_extract("//td[p[contains(., 'Description')]]/following-sibling::td")

    except Exception as e:
        logging.error("Failed Selenium fetch for paramId %d: %s", param_id, e)

    return details


def enrich_variables_with_selenium(
        param_list: List[Dict],
        delay: float
        ) -> List[Dict]:
    """
    Add metadata fields via Selenium for each parameter in param_list.

    Parameters
    ----------
    param_list : list of dict
        Each dict contains 'paramId' and 'shortName'.
    delay : float
        Seconds to wait between requests.

    Returns
    -------
    list of dict
        Each dict enriched with 'fullName', 'units', 'description', 'url'.
    """
    logging.info("Enriching %d parameters...", len(param_list))

    driver = setup_driver()
    enriched = []

    try:
        for i, p in enumerate(param_list):
            pid = p["paramId"]
            short = p["shortName"]

            logging.info("[%d/%d] %s (ID %d)", i + 1, len(param_list), short, pid)

            details = get_parameter_details_selenium(pid, driver)

            enriched.append(
                {
                    "paramId": pid,
                    "shortName": short,
                    "fullName": details.get("name", ""),
                    "units": details.get("unit", ""),
                    "description": details.get("description", ""),
                    "url": f"{URL_BASE}/{pid}/",
                }
            )

            time.sleep(delay)

    finally:
        driver.quit()

    return enriched


# ---------------------------------------------------------------------
# CLI and Main
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and validate GRIB variable metadata by download batch."
    )

    parser.add_argument("--grib-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--coordinates", type=str, default=None)
    parser.add_argument("--uid", type=str, default=None)

    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--process-all-groups",
        action="store_true",
        help="Process all detected GRIB groups when no filters are provided. "
            "Default behavior: only process the most recent group.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(getattr(logging, args.log.upper()))

    grib_dir = args.grib_dir
    files = sorted(grib_dir.glob("*.grib"))

    if not files:
        raise FileNotFoundError(f"No GRIB files found in {grib_dir}")

    logging.info("Found %d GRIB files.", len(files))

    # Group by download batch
    groups = group_files_by_download(files)

    # Optional filtering
    if args.dataset or args.coordinates or args.uid:
        groups = {
            key: fs
            for key, fs in groups.items()
            if (args.dataset is None or key[0] == args.dataset)
            and (args.coordinates is None or key[1] == args.coordinates)
            and (args.uid is None or key[2] == args.uid)
        }

    if not groups:
        logging.error("No GRIB groups match the provided filters.")
        return

    # ------------------------------------------------------------------
    # If multiple groups found and user did not specify filters:
    # - default: pick most recent group (by mtime)
    # - optional: process all groups if --process-all-groups is set
    # ------------------------------------------------------------------
    multiple_groups = (
        len(groups) > 1
        and args.dataset is None
        and args.coordinates is None
        and args.uid is None
    )

    if multiple_groups and not args.process_all_groups:
        logging.warning(
            "Multiple GRIB download groups detected (%d groups). "
            "No filters were provided. "
            "Default behavior: selecting the most recent group. "
            "Use --process-all-groups to process all groups.",
            len(groups),
        )

        def group_latest_mtime(key_and_files):
            key, files = key_and_files
            mtimes = [f.stat().st_mtime for f in files]
            return max(mtimes)

        # Pick the most recent group
        most_recent_group = max(groups.items(), key=group_latest_mtime)

        # Keep only the selected group
        groups = {most_recent_group[0]: most_recent_group[1]}

        dataset, coordinates, uid = most_recent_group[0]
        logging.info(
            "Selected most recent group (by file mtime) → dataset=%s, coordinates=%s, uid=%s",
            dataset, coordinates, uid
        )
    elif multiple_groups and args.process_all_groups:
        logging.info(
            "Multiple groups detected (%d groups), and --process-all-groups was provided. "
            "Processing ALL groups.",
            len(groups),
        )

    # Process each download batch
    for key, group_files in groups.items():
        dataset, coordinates, uid = key
        logging.info(
            "\nProcessing download group: %s, %s, %s (%d files)",
            dataset, coordinates, uid, len(group_files)
        )

        # 1. Scan reference variables from first file
        param_list = scan_parameters_in_file(group_files[0])
        ref_vars = {p["shortName"] for p in param_list}

        # 2. Validate consistency
        verify_variables_across_files(group_files, ref_vars)

        # 3. Enrich via Selenium
        enriched = enrich_variables_with_selenium(param_list, delay=args.delay)

        # 4. Save metadata
        metadata_filename = make_metadata_filename(key)
        output_path = args.output_dir / metadata_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata_dict = {e["shortName"]: e for e in enriched}
        output_path.write_text(json.dumps(metadata_dict, indent=2))

        logging.info("Metadata saved → %s", output_path.resolve())


if __name__ == "__main__":
    main()
