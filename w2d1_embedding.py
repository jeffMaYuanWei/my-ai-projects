#!/usr/bin/env python3
"""Week 2 Day 1: Embedding & Similarity"""

from sentence_transformers import SentenceTransformer, util

# Load pre-trained embedding model
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Three sentences to compare
sentences = [
    "蛋白質結構預測",
    "protein structure prediction",
    "今天天氣很好",
]

# Convert sentences to embeddings (vectors)
embeddings = model.encode(sentences, convert_to_tensor=True)

# Calculate cosine similarity between each pair
pairs = [
    (0, 1, "蛋白質結構預測  vs  protein structure prediction"),
    (0, 2, "蛋白質結構預測  vs  今天天氣很好"),
    (1, 2, "protein structure prediction  vs  今天天氣很好"),
]

print("\n=== Similarity Scores ===\n")
for i, j, label in pairs:
    score = util.cos_sim(embeddings[i], embeddings[j]).item()
    print(f"{label}")
    print(f"Score: {score:.4f}\n")
