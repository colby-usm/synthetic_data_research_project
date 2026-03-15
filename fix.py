# pretty_refcoco_json.py

import json
import os

# ==========================================================
# CONFIG
# ==========================================================

INPUT_FILE = "data/RefCOCO/refcoco+/instances.json"
OUTPUT_FILE = os.path.join(os.path.dirname(INPUT_FILE), "annotations.json")

# ==========================================================
# LOAD AND PRETTY-PRINT
# ==========================================================

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=2)

print(f"Pretty JSON saved to: {OUTPUT_FILE}")
