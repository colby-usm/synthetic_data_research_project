import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Config
# ------------------------
SYN_ANNOTATIONS = Path("data/synthetic_data/annotations.json")
REAL_ANNOTATIONS = Path("data/real_data/custom_subset/annotations.json")
SYN_ROOT = Path("data/synthetic_data/images")
REAL_ROOT = Path("data/real_data/custom_subset/images")
SEED = 42

# ------------------------
# Helper: get all images
# ------------------------
def get_all_images(ann_path, root_path):
    with open(ann_path, "r") as f:
        data = json.load(f)
    images = {img["id"]: img["file_name"] for img in data["images"]}
    return [root_path / fname for fname in images.values()]

# ------------------------
# Load images
# ------------------------
syn_images = get_all_images(SYN_ANNOTATIONS, SYN_ROOT)
real_images = get_all_images(REAL_ANNOTATIONS, REAL_ROOT)
print(f"Synthetic images: {len(syn_images)}")
print(f"Real images: {len(real_images)}")

# Warn if duplicates are present (would cause off-diagonal 1.0s)
if len(syn_images) != len(set(syn_images)):
    print(f"[WARNING] Duplicate paths in syn_images: {len(syn_images) - len(set(syn_images))}")
if len(real_images) != len(set(real_images)):
    print(f"[WARNING] Duplicate paths in real_images: {len(real_images) - len(set(real_images))}")

# ------------------------
# Embedding extraction
# ------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove classifier
resnet.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_embeddings(image_paths, batch_size=16):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = []
        for img_path in image_paths[i:i+batch_size]:
            img = Image.open(img_path).convert("RGB")
            img = preprocess(img)
            batch_imgs.append(img)
        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            batch_emb = resnet(batch_tensor).squeeze(-1).squeeze(-1)
            embeddings.append(batch_emb.cpu().numpy())
    return np.vstack(embeddings)

print("[INFO] Computing embeddings...")
syn_emb = get_embeddings(syn_images)
real_emb = get_embeddings(real_images)

# ------------------------
# Combine embeddings
# ------------------------
all_emb = np.vstack([syn_emb, real_emb])
labels = ["Synthetic"] * len(syn_emb) + ["Real"] * len(real_emb)
colors = {"Synthetic": "red", "Real": "blue"}

# ------------------------
# Compute similarity statistics
# ------------------------
print("[INFO] Computing similarity statistics...")

# Centroids
centroid_syn = np.mean(syn_emb, axis=0, keepdims=True)
centroid_real = np.mean(real_emb, axis=0, keepdims=True)

# Cosine similarity between centroids
centroid_cos_sim = cosine_similarity(centroid_syn, centroid_real)[0][0]

# Intra-set similarities — explicitly zero out the diagonal to exclude
# self-similarities (always 1.0) before flattening.
intra_syn = cosine_similarity(syn_emb)
intra_real = cosine_similarity(real_emb)

np.fill_diagonal(intra_syn, np.nan)
np.fill_diagonal(intra_real, np.nan)

intra_syn_flat = intra_syn[~np.isnan(intra_syn)]
intra_real_flat = intra_real[~np.isnan(intra_real)]

mean_intra_syn = np.mean(intra_syn_flat)
mean_intra_real = np.mean(intra_real_flat)

# Cross-set pairwise similarities
cross_sim = cosine_similarity(syn_emb, real_emb)
cross_sim_flat = cross_sim.flatten()
mean_cross = np.mean(cross_sim_flat)
std_cross = np.std(cross_sim_flat)
min_cross = np.min(cross_sim_flat)
max_cross = np.max(cross_sim_flat)
q1_cross = np.percentile(cross_sim_flat, 25)
median_cross = np.percentile(cross_sim_flat, 50)
q3_cross = np.percentile(cross_sim_flat, 75)

print(f"Centroid cosine similarity (Synthetic vs Real): {centroid_cos_sim:.4f}")
print(f"Mean intra-set similarity: Synthetic={mean_intra_syn:.4f}, Real={mean_intra_real:.4f}")
print(f"Cross-set similarity: mean={mean_cross:.4f}, std={std_cross:.4f}, min={min_cross:.4f}, max={max_cross:.4f}")
print(f"5-number summary (cross-set similarities):")
print(f"  Min: {min_cross:.4f}")
print(f"  25th percentile (Q1): {q1_cross:.4f}")
print(f"  Median: {median_cross:.4f}")
print(f"  75th percentile (Q3): {q3_cross:.4f}")
print(f"  Max: {max_cross:.4f}")
print("=> Half of cross-set similarities fall between Q1 and Q3, i.e., roughly 0.56–0.64.")

# ------------------------
# 2D PCA
# ------------------------
print("[INFO] Reducing embeddings to 2D via PCA...")
pca2 = PCA(n_components=2)
all_emb_2d = pca2.fit_transform(all_emb)

plt.figure(figsize=(8, 6))
for lbl in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == lbl]
    plt.scatter(all_emb_2d[idxs, 0], all_emb_2d[idxs, 1],
                c=colors[lbl], label=f"{lbl} ({len(idxs)})", alpha=0.6, s=50)
plt.title("Synthetic vs Real Image Embeddings (2D PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------
# 3D PCA
# ------------------------
print("[INFO] Reducing embeddings to 3D via PCA...")
pca3 = PCA(n_components=3)
all_emb_3d = pca3.fit_transform(all_emb)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
for lbl in set(labels):
    idxs = [i for i, l in enumerate(labels) if l == lbl]
    ax.scatter(all_emb_3d[idxs, 0], all_emb_3d[idxs, 1], all_emb_3d[idxs, 2],
               c=colors[lbl], label=f"{lbl} ({len(idxs)})", alpha=0.6, s=50)
ax.set_title("Synthetic vs Real Image Embeddings (3D PCA)")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()
plt.tight_layout()
plt.show()

# ------------------------
# Cross-set similarity CDF
# ------------------------
print("[INFO] Plotting cross-set similarity CDF...")
cross_sim_sorted = np.sort(cross_sim_flat)
cdf = np.arange(1, len(cross_sim_sorted) + 1) / len(cross_sim_sorted)

plt.figure(figsize=(8, 6))
plt.plot(cross_sim_sorted, cdf, marker='.', linestyle='none')
plt.title("CDF of Cross-Set Cosine Similarities")
plt.xlabel("Cosine similarity (Synthetic vs Real)")
plt.ylabel("CDF")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# Combined boxplots (Intra + Cross)
# ------------------------
print("[INFO] Plotting combined similarity boxplots...")

box_data = [intra_real_flat, intra_syn_flat, cross_sim_flat]
labels_box = ["Real-Real", "Synthetic-Synthetic", "Real-Synthetic"]
colors_box = ["blue", "red", "purple"]

plt.figure(figsize=(10, 6))

box = plt.boxplot(box_data, vert=True, patch_artist=True)

for patch, color in zip(box['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

plt.xticks([1, 2, 3], labels_box)
plt.ylabel("Cosine similarity")
plt.title("Cosine Similarity Distributions: Intra vs Cross Domain")

# ------------------------
# Annotate 5-number summaries
# ------------------------
def five_num_summary(data):
    return [
        np.min(data),
        np.percentile(data, 25),
        np.median(data),
        np.percentile(data, 75),
        np.max(data)
    ]

summaries = [
    five_num_summary(intra_real_flat),
    five_num_summary(intra_syn_flat),
    five_num_summary(cross_sim_flat)
]

x_offset = 1.25

for i, summary in enumerate(summaries):
    for y, lbl in zip(summary, ["Min", "Q1", "Median", "Q3", "Max"]):
        plt.text(i + x_offset, y, f"{lbl}: {y:.2f}",
                 va='center', fontsize=8)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("similarity_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
