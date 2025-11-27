import json
from pathlib import Path

# Input and output JSON
INPUT_JSON = Path("era5_variable_metadata.json")
OUTPUT_JSON = Path("era5_variable_metadata_with_processing_names.json")

def make_processing_name(full_name: str) -> str:
    """
    Convert fullName → data_processing_name:
      - lower case
      - spaces → underscores
      - optional: remove parentheses, commas, etc.
    """
    if not full_name:
        return ""

    name = full_name.lower().strip()
    name = name.replace(" ", "_")

    # optionally clean punctuation (keep if you prefer)
    for ch in ",()/%":
        name = name.replace(ch, "")

    return name

def main():
    # load metadata
    meta = json.loads(INPUT_JSON.read_text())

    updated = {}

    for short, entry in meta.items():
        full_name = entry.get("fullName", "")

        processing_name = make_processing_name(full_name)

        # copy existing metadata
        new_entry = dict(entry)

        # add new field
        new_entry["data_processing_name"] = processing_name

        updated[short] = new_entry

    # save new JSON
    OUTPUT_JSON.write_text(
        json.dumps(updated, indent=2, ensure_ascii=False)
    )

    print(f"[OK] Added data_processing_name → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
