"""
H6 — Data Efficiency (Embedding Stability Under Dataset Subsampling)

Claim:
JEPA maintains more stable representations under reduced data availability.

Measures:
- kNN neighborhood stability
- Representation variance
- Collapse resistance
- Embedding drift under subsampling
"""

import os, sys, json
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h6")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load models

from utils.embeddings import embed_image  # JEPA
from torchvision.models import resnet50
import timm
from transformers import CLIPModel, CLIPProcessor

models = {}

models["JEPA"] = embed_image

resnet = resnet50(pretrained=True).to(device).eval()
models["ResNet"] = lambda x: resnet(x.to(device))

dino = timm.create_model("vit_base_patch16_224", pretrained=True).to(device).eval()
models["DINO"] = lambda x: dino(x.to(device))

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_clip(img_tensor):
    img = T.ToPILImage()(img_tensor.squeeze(0))
    inputs = clip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)

models["CLIP"] = embed_clip

## Transform

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

## Load images from dataset

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            image_files.append(os.path.join(root, f))

np.random.shuffle(image_files)
image_files = image_files[:400]  # cap for speed

SUBSAMPLE_RATIOS = [1.0, 0.5, 0.25, 0.1, 0.05]

results = {}

## Exp -6 : Data Efficiency

for model_name, embed_fn in models.items():
    print(f"\nRunning H6 for {model_name}...")
    results[model_name] = {}

    full_embeddings = []

    ## Compute full embeddings once
    for path in image_files:
        img = np.array(Image.open(path).convert("RGB"))
        tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)
        emb = embed_fn(tensor)
        emb_np = emb.detach().cpu().numpy().reshape(-1) ## flatten to (D ,)
        full_embeddings.append(emb_np) 

    full_embeddings = np.vstack(full_embeddings) ## shape = (N , D)

    for ratio in SUBSAMPLE_RATIOS:
        k = int(len(image_files) * ratio)
        subset = full_embeddings[:k]

        ## kNN stability
        nn = NearestNeighbors(n_neighbors=5).fit(full_embeddings)
        dists_full, idx_full = nn.kneighbors(full_embeddings)

        nn_sub = NearestNeighbors(n_neighbors=5).fit(subset)
        dists_sub, idx_sub = nn_sub.kneighbors(subset)

        overlap = np.mean([
            len(set(idx_full[i]).intersection(set(idx_sub[min(i, k-1)]))) / 5
            for i in range(min(len(idx_full), len(idx_sub)))
        ])

        ## Representation variance
        variance = np.mean(np.var(subset, axis=0))

        ## Collapse score
        collapse = np.linalg.norm(np.mean(subset, axis=0))

        ## Drift vs full
        drift = np.mean(np.linalg.norm(full_embeddings[:k] - subset[:k], axis=1))

        results[model_name][str(ratio)] = {
            "knn_overlap": float(overlap),
            "variance": float(variance),
            "collapse": float(collapse),
            "drift": float(drift)
        }

## Save results as json

json_path = os.path.join(RESULTS_DIR, "h6_data_efficiency.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)

## Plots

ratios = [str(r) for r in SUBSAMPLE_RATIOS]

## Plot 1 — kNN Stability
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][r]["knn_overlap"] for r in ratios]
    plt.plot(SUBSAMPLE_RATIOS, vals, label=model)

plt.title("H6 — kNN Stability vs Dataset Size")
plt.xlabel("Dataset Fraction")
plt.ylabel("Neighbor Overlap")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "knn_stability.png"))
plt.close()

## Plot 2 — Representation Variance
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][r]["variance"] for r in ratios]
    plt.plot(SUBSAMPLE_RATIOS, vals, label=model)

plt.title("H6 — Representation Diversity vs Dataset Size")
plt.xlabel("Dataset Fraction")
plt.ylabel("Embedding Variance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "representation_variance.png"))
plt.close()

## Plot 3 — Collapse Resistance
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][r]["collapse"] for r in ratios]
    plt.plot(SUBSAMPLE_RATIOS, vals, label=model)

plt.title("H6 — Collapse Resistance Curve")
plt.xlabel("Dataset Fraction")
plt.ylabel("Collapse Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "collapse_resistance.png"))
plt.close()

## Plot 4 — Representation Drift
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][r]["drift"] for r in ratios]
    plt.plot(SUBSAMPLE_RATIOS, vals, label=model)

plt.title("H6 — Embedding Drift Under Subsampling")
plt.xlabel("Dataset Fraction")
plt.ylabel("Embedding Drift")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "embedding_drift.png"))
plt.close()

print("H6 Complete — Results saved to plots/h6/")
