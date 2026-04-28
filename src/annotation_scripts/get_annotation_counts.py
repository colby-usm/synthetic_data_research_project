import json
import os
from collections import Counter

cwd = os.getcwd()
print(cwd)

with open('data/real_data/custom_subset/annotations.json', 'r') as f:
    data = json.load(f)

anns = data['annotations']
total_count = len(anns)

category_counts = Counter(ann['category_id'] for ann in anns)
cats = data['categories']
id_to_name_dict = {cat['id']: cat['name'] for cat in cats}

final_counts = {
    cid: {
        "name": id_to_name_dict.get(cid, "unknown"),
        "count": count
    }
    for cid, count in category_counts.items()
}

print(f"final counts: {final_counts}")
print(f"total count: {total_count}")
