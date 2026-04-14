"""
compare_embeddings.py

Compare token embeddings between synthetic and real corpora to visualize
semantic coverage and domain shift.
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ------------------------
# Configuration
# ------------------------
CFG_PATH = "cfg.json"

with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# Model
MODEL_ID = cfg.get("directories", {}).get("padt_7b_rec", "PaDT-MLLM/PaDT_REC_7B")

# Synthetic (reference)
SYN_PATH = Path(cfg.get("directories", {}).get("synthetic_data", "synthetic_data")) / "refexps.json"

# Real (target)
REAL_PATH = Path(cfg.get("eval", {}).get("real_data", "real_data")) / "refexps.json"

# ------------------------
# Helpers
# ------------------------
def load_sentences(path):
    with open(path, "r") as f:
        data = json.load(f)
    sentences = []
    for item in data:
        for s in item.get("sentences", []):
            if s.get("sent"):
                sentences.append(s["sent"])
    return sentences


def get_corpus_embeddings(texts, tokenizer, embeddings):
    token_ids = set()
    for text in texts:
        toks = tokenizer.tokenize(text)
        ids = tokenizer(toks, add_special_tokens=False)["input_ids"]
        token_ids.update(ids)
    token_ids = list(token_ids)
    token_embeds = embeddings[token_ids]  # [num_tokens, hidden_size]
    return token_ids, token_embeds


def pairwise_cosine_similarity(emb1, emb2):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    return emb1 @ emb2.T


# ------------------------
# Load tokenizer + embeddings
# ------------------------
print("[INFO] Loading tokenizer and model embeddings (low memory)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#model = AutoModel.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
model = AutoModel.from_pretrained("PaDT-MLLM/PaDT_REC_7B")
embeddings = model.get_input_embeddings().weight  # [vocab_size, hidden_size]

# ------------------------
# Load corpora
# ------------------------
print("[INFO] Loading corpora...")
syn_texts = load_sentences(SYN_PATH)
real_texts = load_sentences(REAL_PATH)
print(f"[INFO] Synthetic sentences: {len(syn_texts)}")
print(f"[INFO] Real sentences: {len(real_texts)}")

# ------------------------
# Map tokens → embeddings
# ------------------------
print("[INFO] Mapping synthetic tokens to embeddings...")
syn_ids, syn_emb = get_corpus_embeddings(syn_texts, tokenizer, embeddings)
print("[INFO] Mapping real tokens to embeddings...")
real_ids, real_emb = get_corpus_embeddings(real_texts, tokenizer, embeddings)

# ------------------------
# Compute max cosine similarity (synthetic → real)
# ------------------------
print("[INFO] Computing pairwise cosine similarities...")
sim_matrix = pairwise_cosine_similarity(syn_emb, real_emb)
max_sim_per_token = sim_matrix.max(dim=1).values.cpu().numpy()

# ------------------------
# Visualize histogram / CDF
# ------------------------
plt.figure(figsize=(8,5))
plt.hist(max_sim_per_token, bins=30, density=True, alpha=0.7, color='skyblue')
plt.xlabel("Max Cosine Similarity to Real Tokens")
plt.ylabel("Density")
plt.title("Semantic Coverage of Synthetic Tokens")
plt.grid(True)
plt.tight_layout()
plt.show()

# CDF
sorted_sim = sorted(max_sim_per_token)
plt.figure(figsize=(8,5))
plt.plot(sorted_sim, [i/len(sorted_sim) for i in range(len(sorted_sim))], marker='.', linestyle='none')
plt.xlabel("Max Cosine Similarity to Real Tokens")
plt.ylabel("CDF")
plt.title("CDF of Synthetic Token Coverage")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# PCA embedding visualization
# ------------------------
print("[INFO] PCA visualization of embeddings...")
all_embeds = torch.cat([syn_emb, real_emb], dim=0).cpu().numpy()
pca = PCA(n_components=2).fit_transform(all_embeds)

plt.figure(figsize=(8,6))
plt.scatter(pca[:len(syn_emb),0], pca[:len(syn_emb),1], label="Synthetic", alpha=0.5)
plt.scatter(pca[len(syn_emb):,0], pca[len(syn_emb):,1], label="Real", alpha=0.5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Embedding Space Overlap")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("[INFO] Done.")
