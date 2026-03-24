import json
from pathlib import Path

# ----------------------
# Paths
# ----------------------
file1_path = Path("refexps_colby.json")  # first JSON file
file2_path = Path("refexps_lucas.json")  # second JSON file
output_path = Path("refexps.json")

# ----------------------
# Load JSON files
# ----------------------
with open(file1_path, "r") as f:
    data1 = json.load(f)

with open(file2_path, "r") as f:
    data2 = json.load(f)

# ----------------------
# Build lookup by image_id for second file
# ----------------------
image2_map = {}
for ref in data2:
    image_id = ref["image_id"]
    image2_map.setdefault(image_id, []).append(ref)

# ----------------------
# Combine refs
# ----------------------
combined_refs = []

for ref1 in data1:
    image_id = ref1["image_id"]
    combined_refs.append(ref1)  # always include ref1

    # If image_id exists in file2, add all refs from file2
    if image_id in image2_map:
        for ref2 in image2_map[image_id]:
            # Make a copy and adjust ref_id to avoid collisions
            new_ref = ref2.copy()
            new_ref["ref_id"] = None  # optionally reassign ref_id later
            combined_refs.append(new_ref)

# ----------------------
# Write output
# ----------------------
with open(output_path, "w") as f:
    json.dump(combined_refs, f, indent=4)

print(f"Combined references written to {output_path}")
