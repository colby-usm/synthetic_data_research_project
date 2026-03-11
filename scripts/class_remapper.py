import json

# ==========================================================
# CONFIGURATION
# ==========================================================

ANNOTATIONS_PATH = "synthetic_data_v1/annotations.json"
OUTPUT_PATH = "synthetic_data_v1/annotations_fixed.json"

# The labels we want to remap
GLB_LABELS = {"glb", "glb1", "glb2"}
NEW_CATEGORY_ID = 6

# ==========================================================
# LOAD ANNOTATIONS
# ==========================================================

with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

annotations = data["annotations"]

# ==========================================================
# REMAP CATEGORY IDS
# ==========================================================

count_fixed = 0

for ann in annotations:
    if ann.get("actor_label") in GLB_LABELS:
        if ann.get("category_id") != NEW_CATEGORY_ID:
            ann["category_id"] = NEW_CATEGORY_ID
            count_fixed += 1

print(f"Fixed category_id for {count_fixed} annotations.")

# ==========================================================
# SAVE FIXED FILE
# ==========================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved fixed annotations to {OUTPUT_PATH}")
