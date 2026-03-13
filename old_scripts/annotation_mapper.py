"""
Remap category IDs based on actor_label patterns.

Rules:
  - actor_label contains 'mtvr'  →  category_id = 3
  - actor_label contains 'glb'   →  category_id = 2
"""

import json
import os

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ANNOTATIONS_IN  = "./bulk_output_annotations.json"   # Path to input annotations JSON
ANNOTATIONS_OUT = "./remapped_annotations.json"   # Path to write output (set same as input to overwrite)

# Rules: (substring_to_match, new_category_id)
REMAP_RULES = [
    ("mtvr", 3),
    ("glb",  2),
]

# ──────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(data: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)
    print(f"[INFO] Saved → {path}")

# ──────────────────────────────────────────────

coco = load_json(ANNOTATIONS_IN)
annotations = coco.get("annotations", [])

counters = {label: 0 for label, _ in REMAP_RULES}
no_match = 0

for ann in annotations:
    actor_label = ann.get("actor_label", "").lower()
    matched = False
    for substring, new_cat_id in REMAP_RULES:
        if substring in actor_label:
            if ann.get("category_id") != new_cat_id:
                ann["category_id"] = new_cat_id
            counters[substring] += 1
            matched = True
            break   # first matching rule wins
    if not matched:
        no_match += 1

print("\n" + "=" * 45)
print("  Remap Results")
print("=" * 45)
for substring, new_cat_id in REMAP_RULES:
    print(f"  *{substring}*  →  category_id={new_cat_id}  :  {counters[substring]} annotation(s)")
print(f"  No match (unchanged)             :  {no_match} annotation(s)")
print(f"  Total                            :  {len(annotations)} annotation(s)")
print("=" * 45 + "\n")

save_json(coco, ANNOTATIONS_OUT)
