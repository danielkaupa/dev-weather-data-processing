#!/usr/bin/env python3
"""
step1c_extract_and_validate_variable_metadata.py
===============================================

Automatically scan GRIB files in a directory, group them by unique download
signature (dataset, coordinates, uid), validate variable consistency within
each group, and extract enriched metadata (via Selenium from ECMWF parameter DB).

User Input
----------
The user only needs to specify:
    --grib-dir /path/to/grib/files

Optionally:
    --dataset <value>
    --coordinates <value>
    --uid <value>
    --delay <seconds>          # delay between Selenium requests
    --process-all-groups       # process all detected groups instead of most recent

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

Parallelism
-----------
- **Variable consistency checks** across files in a group are parallelised with MPI:
  each rank gets a unique subset of monthly files to compare.
- **Selenium metadata enrichment** is performed **only on rank 0** (once per group).
- Script still works without MPI (SIZE=1) and behaves like a normal serial script.

Requirements
------------
eccodes
selenium
mpi4py           (for MPI parallelism; script still runs without it)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

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
# MPI SETUP (safe if mpi4py or MPI isn't available)
# ---------------------------------------------------------------------

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    MPI_ENABLED = True
except Exception:  # noqa: BLE001
    COMM = None
    RANK = 0
    SIZE = 1
    MPI_ENABLED = False


def detect_default_processpool_workers() -> int:
    """
    Choose a sensible worker count: 2/3 of available logical CPUs.
    """
    try:
        n = os.cpu_count()
        if not n:
            return 1
        return max(1, int(n * 2 / 3))
    except Exception:
        return 1


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
    """
    # We include the MPI rank in every log line.
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [RANK {RANK}] [%(levelname)s] %(message)s",
    )


# ---------------------------------------------------------------------
# Filename Parsing
# ---------------------------------------------------------------------

FILENAME_PATTERN = re.compile(
    r"""
    ^(?P<dataset>[A-Za-z0-9\-]+)_      # dataset or dataset-region
    (?P<coordinates>[A-Za-z0-9]+)_     # coordinates like N37W68S6E98
    (?P<uid>[A-Za-z0-9]+)_             # UID (hash-like)
    (?P<year>\d{4})_                   # YYYY
    (?P<month>\d{2})\.grib$            # MM
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
    groups: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
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

    Parameters
    ----------
    path : Path
        GRIB file path.

    Returns
    -------
    set of str
        Unique shortName values present in the file.
    """
    variables: Set[str] = set()

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
    except Exception as e:  # noqa: BLE001
        logging.error("Error scanning variables in %s: %s", path, e)

    return variables


def scan_parameters_in_file(path: Path) -> List[Dict]:
    """
    Extract unique {paramId, shortName} entries from a GRIB file.

    Parameters
    ----------
    path : Path
        GRIB file path.

    Returns
    -------
    list of dict
        Each dict contains 'paramId' and 'shortName'.
    """
    params: Dict[int, Dict[str, object]] = {}

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
    except Exception as e:  # noqa: BLE001
        logging.error("Error scanning parameters in %s: %s", path, e)

    return list(params.values())


# ---------------------------------------------------------------------
# Consistency Checking (MPI-based)
# ---------------------------------------------------------------------

def _check_single_file_variables(
    path: Path,
    reference_vars: Set[str],
) -> Tuple[Path, Set[str], Set[str], bool, float]:
    """
    Compare one file's variables to the reference set, with timing.

    Parameters
    ----------
    path : Path
        GRIB file to check.
    reference_vars : set of str
        Expected variable set (shortNames).

    Returns
    -------
    path : Path
        The input file path.
    missing : set of str
        Variables present in reference but missing in this file.
    extra : set of str
        Variables present in this file but not in the reference.
    ok : bool
        True if variable sets match exactly.
    elapsed : float
        Time taken (seconds).
    """
    start = time.time()
    vars_this = scan_variables_in_file(path)

    missing = reference_vars - vars_this
    extra = vars_this - reference_vars
    ok = (len(missing) == 0 and len(extra) == 0)

    elapsed = time.time() - start
    return path, missing, extra, ok, elapsed


def verify_variables_across_files(
    files: List[Path],
    reference_vars: Set[str],
    processing_mode: str = "mpi",
    pool_workers: Optional[int] = None,
) -> None:
    """
    Validate that all GRIB files in a group share the same variable set.

    Supported modes
    ---------------
    - mpi:
        Distribute files across MPI ranks evenly (i % SIZE == RANK).
    - processpool:
        Only RANK 0 uses ProcessPoolExecutor with `pool_workers`;
        other ranks idle.
    - sequential:
        Only RANK 0 processes files in a loop.

    All modes print per-file status messages:
        PROCESS COMPLETE — <filename>
            RESULT: SUCCESS/FAILURE
            TIME: HH:MM:SS
            PROGRESS: X / N
    """
    # --------------------------
    # Input validation
    # --------------------------
    if not files:
        if RANK == 0:
            logging.warning("No files provided for variable verification.")
        return

    reference_file = files[0]
    other_files = files[1:]
    total_global = len(other_files)

    if total_global == 0:
        if RANK == 0:
            logging.info(
                "Only one file (%s) in group; variable consistency trivially satisfied.",
                reference_file.name,
            )
        return

    if RANK == 0:
        logging.info(
            "Validating variable consistency across %d files (reference: %s)...",
            total_global,
            reference_file.name,
        )

    # --------------------------
    # Work assignment
    # --------------------------
    # Build list of (global index, file)
    file_pairs = list(enumerate(other_files))

    # -------- MPI DISTRIBUTION --------
    if processing_mode == "mpi" and MPI_ENABLED and SIZE > 1:
        local_pairs = file_pairs[RANK::SIZE]
        if RANK == 0:
            logging.info("MPI mode: %d ranks dividing %d files", SIZE, total_global)

    # -------- PROCESSPOOL MODE --------
    elif processing_mode == "processpool":
        if RANK == 0:
            # Rank 0 processes all files using a local pool
            local_pairs = file_pairs
        else:
            local_pairs = []  # all other ranks idle

        if RANK == 0:
            logging.info(
                "ProcessPool mode: using %s worker(s) on rank 0",
                pool_workers,
            )

    # -------- SEQUENTIAL MODE --------
    elif processing_mode == "sequential":
        if RANK == 0:
            local_pairs = file_pairs
            logging.info("Sequential mode: rank 0 processing all files")
        else:
            local_pairs = []

    else:
        raise ValueError(f"Unknown processing_mode={processing_mode}")

    # --------------------------
    # Containers for mismatches
    # --------------------------
    local_mismatches: List[Tuple[Path, Set[str], Set[str]]] = []

    # =========================================================================
    # CASE 1 — ProcessPoolExecutor (only RANK 0)
    # =========================================================================
    if processing_mode == "processpool" and RANK == 0:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=pool_workers) as ex:
            futures = {
                ex.submit(_check_single_file_variables, path, reference_vars): (idx, path)
                for idx, path in local_pairs
            }

            for fut in as_completed(futures):
                idx, path = futures[fut]
                path, missing, extra, ok, elapsed = fut.result()

                # Pretty time formatting
                h = int(elapsed // 3600)
                m = int((elapsed % 3600) // 60)
                s = int(elapsed % 60)
                t_str = f"{h:02d}:{m:02d}:{s:02d}"

                status = "SUCCESS (variable names match)" if ok else "FAILURE (mismatch)"
                progress_str = f"{idx + 1} / {total_global}"

                print(
                    f"[Rank 0] PROCESS COMPLETE — {path.name}\n"
                    f"        RESULT: {status}\n"
                    f"        TIME: {t_str}\n"
                    f"        PROGRESS: {progress_str}",
                    flush=True,
                )

                if not ok:
                    local_mismatches.append((path, missing, extra))

        # No MPI to gather; rank 0 alone has full results
        combined = local_mismatches

        # Summary
        if not combined:
            logging.info(
                "Variable consistency check complete — all %d files share "
                "the same shortName set.",
                total_global + 1,
            )
        else:
            logging.warning("Variable inconsistencies detected:")
            for path, missing, extra in combined:
                logging.warning("File: %s", path.name)
                if missing:
                    logging.warning("  Missing: %s", sorted(missing))
                if extra:
                    logging.warning("  Extra:   %s", sorted(extra))
        return

    # =========================================================================
    # CASE 2 — MPI MODE or SEQUENTIAL ON RANK 0
    # =========================================================================
    for global_idx, path in local_pairs:
        path, missing, extra, ok, elapsed = _check_single_file_variables(path, reference_vars)

        # Format timestamp
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        t_str = f"{h:02d}:{m:02d}:{s:02d}"

        status = "SUCCESS (variable names match)" if ok else "FAILURE (mismatch)"
        progress_str = f"{global_idx + 1} / {total_global}"

        print(
            f"[Rank {RANK}] PROCESS COMPLETE — {path.name}\n"
            f"        RESULT: {status}\n"
            f"        TIME: {t_str}\n"
            f"        PROGRESS: {progress_str}",
            flush=True,
        )

        if not ok:
            local_mismatches.append((path, missing, extra))

    # --------------------------
    # Combine results across MPI
    # --------------------------
    if processing_mode == "mpi" and MPI_ENABLED and SIZE > 1:
        all_mismatches = COMM.gather(local_mismatches, root=0)
        COMM.Barrier()
    else:
        all_mismatches = [local_mismatches]

    # --------------------------
    # Rank 0: Print summary
    # --------------------------
    if RANK == 0:
        combined = [x for sub in all_mismatches for x in sub]

        if not combined:
            logging.info(
                "Variable consistency check complete — all %d files share "
                "the same shortName set.",
                total_global + 1,
            )
        else:
            logging.warning("Variable inconsistencies detected:")
            for path, missing, extra in combined:
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

    Returns
    -------
    webdriver.Chrome
        Configured headless Chrome driver instance.
    """
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_parameter_details_selenium(
    param_id: int,
    driver: webdriver.Chrome,
) -> Dict[str, str]:
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
        def safe_extract(xpath: str) -> str:
            try:
                cell = driver.find_element(By.XPATH, xpath)
                return cell.find_element(By.TAG_NAME, "p").text
            except Exception:  # noqa: BLE001
                return ""

        details["name"] = safe_extract(
            "//td[p[contains(., 'Name')]]/following-sibling::td"
        )
        details["unit"] = safe_extract(
            "//td[p[contains(., 'Unit')]]/following-sibling::td"
        )
        details["description"] = safe_extract(
            "//td[p[contains(., 'Description')]]/following-sibling::td"
        )

    except Exception as e:  # noqa: BLE001
        logging.error("Failed Selenium fetch for paramId %d: %s", param_id, e)

    return details


def enrich_variables_with_selenium(
    param_list: List[Dict],
    delay: float,
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
    logging.info("Enriching %d parameters via Selenium...", len(param_list))

    driver = setup_driver()
    enriched: List[Dict] = []

    try:
        for i, p in enumerate(param_list):
            pid = p["paramId"]
            short = p["shortName"]

            logging.info("[%d/%d] %s (paramId=%d)", i + 1, len(param_list), short, pid)

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
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Extract and validate GRIB variable metadata by download batch."
    )

    parser.add_argument(
        "--grib-dir",
        type=Path,
        required=True,
        help="Directory containing GRIB files (*.grib).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write output metadata JSON files.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional filter: dataset name.",
    )
    parser.add_argument(
        "--coordinates",
        type=str,
        default=None,
        help="Optional filter: coordinates token.",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Optional filter: download UID.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay (seconds) between Selenium requests (default: 1.0).",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    parser.add_argument(
        "--process-all-groups",
        action="store_true",
        help=(
            "Process all detected GRIB groups when no filters are provided. "
            "Default: only process the most recent group."
        ),
    )

    parser.add_argument(
        "--processing-mode",
        type=str,
        default="processpool",
        choices=["processpool", "mpi", "sequential"],
        help="Processing mode for variable consistency checks. "
            "'processpool' uses local CPU cores, 'mpi' distributes across MPI ranks, "
            "and 'sequential' runs single-threaded (default: processpool).",
    )
    parser.add_argument(
        "--use-selenium",
        action="store_true",
        help="Enable Selenium metadata scraping (default: off; use only on local machine).",
    )



    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(getattr(logging, args.log.upper()))

    use_selenium = args.use_selenium

    grib_dir = args.grib_dir
    files = sorted(grib_dir.glob("*.grib"))

    if not files:
        if RANK == 0:
            raise FileNotFoundError(f"No GRIB files found in {grib_dir}")
        # Other ranks just return
        return

    if RANK == 0:
        logging.info("Found %d GRIB files in %s", len(files), grib_dir)

    # Group by download batch (same on all ranks)
    groups = group_files_by_download(files)

    if RANK == 0:
        logging.info("Detected %d GRIB download group(s).", len(groups))

    # Optional filtering by dataset/coordinates/uid
    if args.dataset or args.coordinates or args.uid:
        groups = {
            key: fs
            for key, fs in groups.items()
            if (args.dataset is None or key[0] == args.dataset)
            and (args.coordinates is None or key[1] == args.coordinates)
            and (args.uid is None or key[2] == args.uid)
        }

        if not groups:
            if RANK == 0:
                logging.error("No GRIB groups match the provided filters.")
            return

        if RANK == 0:
            logging.info(
                "After applying filters, %d group(s) remain.",
                len(groups),
            )

    # ------------------------------------------------------------------
    # Handle multiple groups when no explicit filters are given.
    # ------------------------------------------------------------------
    multiple_groups = (
        len(groups) > 1
        and args.dataset is None
        and args.coordinates is None
        and args.uid is None
    )

    if multiple_groups and not args.process_all_groups:
        if RANK == 0:
            logging.warning(
                "Multiple GRIB download groups detected (%d groups). "
                "No filters were provided. "
                "Default behavior: selecting the most recent group. "
                "Use --process-all-groups to process all groups.",
                len(groups),
            )

        def group_latest_mtime(key_and_files):
            key, files_for_group = key_and_files
            mtimes = [f.stat().st_mtime for f in files_for_group]
            return max(mtimes)

        # Pick the most recent group (same decision on all ranks)
        most_recent_group = max(groups.items(), key=group_latest_mtime)
        groups = {most_recent_group[0]: most_recent_group[1]}

        if RANK == 0:
            dataset, coordinates, uid = most_recent_group[0]
            logging.info(
                "Selected most recent group (by file mtime) → dataset=%s, coordinates=%s, uid=%s",
                dataset,
                coordinates,
                uid,
            )
    elif multiple_groups and args.process_all_groups and RANK == 0:
        logging.info(
            "Multiple groups detected (%d groups), and --process-all-groups was provided. "
            "Processing ALL groups.",
            len(groups),
        )

    # ------------------------------------------------------------------
    # Process each download batch
    # ------------------------------------------------------------------
    for key, group_files in groups.items():
        dataset, coordinates, uid = key
        group_files = sorted(group_files)

        if RANK == 0:
            logging.info(
                "Processing download group: dataset=%s, coordinates=%s, uid=%s (%d files)",
                dataset,
                coordinates,
                uid,
                len(group_files),
            )

        # 1. Scan reference parameters from first file (same file on all ranks)
        reference_file = group_files[0]
        if RANK == 0:
            logging.info(
                "Using reference file for parameters: %s",
                reference_file.name,
            )

        # Only rank 0 actually needs to build param_list / ref_vars;
        # but doing it everywhere is cheap and keeps logic simple.
        param_list = scan_parameters_in_file(reference_file)
        if not param_list:
            if RANK == 0:
                logging.error(
                    "No parameters found in reference file %s. Skipping group.",
                    reference_file.name,
                )
            continue

        ref_vars = {p["shortName"] for p in param_list}
        if RANK == 0:
            logging.info("Reference variable set size: %d shortNames", len(ref_vars))

        # 2. Validate consistency across group (MPI distributes files)
        # -------------------------------
        # 2. Variable consistency checking
        # -------------------------------
        mode = args.processing_mode

        if mode == "sequential":
            if RANK == 0:
                logging.info("Processing mode: sequential")
            # Override MPI so only rank 0 does work
            verify_variables_across_files(
                group_files,
                ref_vars,
                processing_mode="sequential",
                pool_workers=1,
            )

        elif mode == "mpi":
            if RANK == 0:
                logging.info("Processing mode: MPI — using %d ranks", SIZE)
            verify_variables_across_files(
                group_files,
                ref_vars,
                processing_mode="mpi",
                pool_workers=None,
            )

        else:  # processpool
            # Only rank 0 uses the processpool; others idle
            if RANK == 0:
                workers = detect_default_processpool_workers()
                logging.info(
                    "Processing mode: processpool — using %d workers (2/3 of CPU count)",
                    workers,
                )
                verify_variables_across_files(
                    group_files,
                    ref_vars,
                    processing_mode="processpool",
                    pool_workers=workers,
                )
            else:
                # Other ranks wait
                pass

        # Synchronise ranks before moving on
        if MPI_ENABLED and SIZE > 1:
            COMM.Barrier()

        # 3. Enrich or load metadata
        if RANK == 0:
            metadata_filename = make_metadata_filename(key)
            output_path = args.output_dir / metadata_filename

            if use_selenium:
                logging.info("Selenium enabled → scraping live ECMWF parameter metadata.")
                enriched = enrich_variables_with_selenium(param_list, delay=args.delay)
                metadata_dict = {e["shortName"]: e for e in enriched}
                output_path.write_text(json.dumps(metadata_dict, indent=2))
                logging.info("Metadata saved → %s", output_path.resolve())

            else:
                logging.info("Selenium disabled → loading existing metadata from disk.")
                if not output_path.exists():
                    raise FileNotFoundError(
                        f"No metadata file found: {output_path}\n"
                        f"Run once locally with --use-selenium to generate it."
                    )
                metadata_dict = json.loads(output_path.read_text())
                if not metadata_dict:
                    raise RuntimeError("Loaded metadata was empty – this indicates a damaged file.")


        # Synchronise ranks before moving to next group
        if MPI_ENABLED and SIZE > 1:
            COMM.Barrier()


if __name__ == "__main__":
    main()
