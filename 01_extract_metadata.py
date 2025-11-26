"""
03_extract_and_validate_variable_metadata.py

Extract GRIB variable lists from ERA5 files, validate consistency across files,
and enrich metadata using Selenium (ECMWF param database).

Output:
- era5-world_variable_metadata.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Set

from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_release,
    CodesInternalError,
)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

RAW_GRIB_DIR = Path("../data/raw")
OUTPUT_METADATA_JSON = Path("era5-world_variable_metadata.json")

# polite delay so ECMWF doesn't ban your IP
SCRAPE_DELAY = 1.0


# ---------------------------------------------------------------------
# GRIB VARIABLE SCANNING
# ---------------------------------------------------------------------

def scan_variables_in_file(path: Path) -> Set[str]:
    """
    Scan GRIB file for variable shortNames using ecCodes low-level API.
    Returns a set of shortName strings.
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
                    short_name = codes_get(gid, "shortName")
                    if isinstance(short_name, bytes):
                        short_name = short_name.decode("utf-8")
                    variables.add(str(short_name))

                finally:
                    codes_release(gid)
    except Exception as e:
        print(f"❌ Error scanning variables in {path}: {e}")

    return variables


def scan_parameters_in_file(path: Path) -> List[Dict]:
    """
    Returns a list of:
        { "paramId": int, "shortName": str }
    with deduplication by paramId.
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
                    short_name = codes_get(gid, "shortName")
                    if isinstance(short_name, bytes):
                        short_name = short_name.decode("utf-8")

                    param_id = int(codes_get(gid, "paramId"))

                    params[param_id] = {
                        "paramId": param_id,
                        "shortName": str(short_name),
                    }

                finally:
                    codes_release(gid)
    except Exception as e:
        print(f"❌ Error scanning paramIds in {path}: {e}")

    return list(params.values())


# ---------------------------------------------------------------------
# VARIABLE CONSISTENCY CHECK
# ---------------------------------------------------------------------

def verify_variables_across_files(files: List[Path], reference_vars: Set[str]) -> None:
    """
    Check that all GRIB files contain exactly the same variable set.
    Warn if any file differs.
    """
    print("\n🔍 Checking variable consistency across files...\n")
    mismatches = []

    for path in files:
        vars_this = scan_variables_in_file(path)
        if vars_this != reference_vars:
            diff_ref = reference_vars - vars_this
            diff_this = vars_this - reference_vars
            mismatches.append((path, diff_ref, diff_this))

    if not mismatches:
        print("✅ All GRIB files share the same variable set.\n")
    else:
        print("⚠ Variable inconsistencies detected:\n")
        for path, missing, extra in mismatches:
            print(f"File: {path.name}")
            if missing:
                print("  Missing (expected but absent):")
                for v in sorted(missing):
                    print(f"    - {v}")
            if extra:
                print("  Extra (unexpected):")
                for v in sorted(extra):
                    print(f"    - {v}")
            print()


# ---------------------------------------------------------------------
# SELENIUM SCRAPING
# ---------------------------------------------------------------------

def setup_driver():
    """
    Create headless Chrome Selenium driver.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def get_parameter_details_selenium(param_id: int, driver) -> Dict:
    """
    Scrape ECMWF param DB via Selenium for name, unit, description.
    """
    url = f"https://codes.ecmwf.int/grib/param-db/{param_id}/"
    try:
        driver.get(url)

        # Wait for metadata table
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//td[contains(., 'Name')]"))
        )

        details = {}

        # Name
        try:
            elem = driver.find_element(By.XPATH,
                "//td[p[contains(., 'Name')]]/following-sibling::td")
            details["name"] = elem.find_element(By.TAG_NAME, "p").text
        except:
            details["name"] = ""

        # Unit
        try:
            elem = driver.find_element(By.XPATH,
                "//td[p[contains(., 'Unit')]]/following-sibling::td")
            details["unit"] = elem.find_element(By.TAG_NAME, "p").text
        except:
            details["unit"] = ""

        # Description
        try:
            elem = driver.find_element(By.XPATH,
                "//td[p[contains(., 'Description')]]/following-sibling::td")
            details["description"] = elem.find_element(By.TAG_NAME, "p").text
        except:
            details["description"] = ""

        return details

    except Exception as e:
        print(f"❌ Failed Selenium fetch for paramId {param_id}: {e}")
        return {"name": "", "unit": "", "description": ""}


def enrich_variables_with_selenium(param_list: List[Dict]) -> List[Dict]:
    """
    Add fullName, units, description to each paramId / shortName entry.
    """
    print("\n🔎 Enriching parameter metadata with Selenium...\n")

    driver = setup_driver()
    enriched = []

    try:
        for i, p in enumerate(param_list):
            param_id = p["paramId"]
            short_name = p["shortName"]

            print(f"  → {short_name} (ID {param_id}) {i+1}/{len(param_list)}")

            details = get_parameter_details_selenium(param_id, driver)

            enriched.append({
                "paramId": param_id,
                "shortName": short_name,
                "fullName": details.get("name", ""),
                "units": details.get("unit", ""),
                "description": details.get("description", ""),
                "url": f"https://codes.ecmwf.int/grib/param-db/{param_id}/"
            })

            time.sleep(SCRAPE_DELAY)

    finally:
        driver.quit()

    return enriched


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("📂 Scanning GRIB directory:", RAW_GRIB_DIR)

    grib_files = sorted(RAW_GRIB_DIR.glob("*.grib"))
    if not grib_files:
        raise FileNotFoundError(f"No GRIB files found in {RAW_GRIB_DIR}")

    # 1. Get reference variable set
    print("🔍 Scanning first GRIB file for variables...")
    param_list = scan_parameters_in_file(grib_files[0])
    reference_vars = {p["shortName"] for p in param_list}
    print(f"Found {len(reference_vars)} variables.")

    # 2. Validate consistency
    verify_variables_across_files(grib_files, reference_vars)

    # 3. Enrich with selenium
    enriched = enrich_variables_with_selenium(param_list)

    # 4. Convert to dict keyed by shortName
    metadata_dict = {item["shortName"]: item for item in enriched}

    # 5. Save
    OUTPUT_METADATA_JSON.write_text(
        json.dumps(metadata_dict, indent=2)
    )

    print(f"\n💾 Saved metadata to {OUTPUT_METADATA_JSON.absolute()}")


if __name__ == "__main__":
    main()
