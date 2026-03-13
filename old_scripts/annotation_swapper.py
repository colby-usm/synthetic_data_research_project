import json
ANNOTATIONS_PATH = "synthetic_data_v1/annotations_fixed.json"
OUTPUT_PATH = "synthetic_data_v1/annotations_swapped.json"


# Define the mapping: old_category_id -> new_category_id
CATEGORY_SWAP = {
    6: 2,
    5: 3
}

# ==========================================================
# LOAD ANNOTATIONS
# ==========================================================

with open(ANNOTATIONS_PATH, "r") as f:
    data = json.load(f)

annotations = data["annotations"]

# ==========================================================
# APPLY CATEGORY SWAP
# ==========================================================

count_swapped = 0

for ann in annotations:
    old_id = ann.get("category_id")
    if old_id in CATEGORY_SWAP:
        ann["category_id"] = CATEGORY_SWAP[old_id]
        count_swapped += 1

print(f"Swapped category_id for {count_swapped} annotations.")

# ==========================================================
# SAVE FIXED FILE
# ==========================================================

with open(OUTPUT_PATH, "w") as f:
    json.dump(data, f, indent=2)

print(f"Saved swapped annotations to {OUTPUT_PATH}")
