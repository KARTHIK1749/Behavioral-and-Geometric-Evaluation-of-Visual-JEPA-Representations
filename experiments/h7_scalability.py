"""
H7 — Scaling Law: Representation Geometry Under Dataset Growth

Claim:
JEPA’s advantage is STRUCTURAL — its embedding geometry scales better.

Measures:
- Cluster compactness
- Inter-class margin
- Intrinsic dimensionality
- Neighborhood stability
- Statistical significance
"""

import os, sys, json, random
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h7")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load Models

from utils.embeddings import embed_image
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

## Load Images

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".jpeg",".png")):
            image_files.append(os.path.join(root, f))

random.shuffle(image_files)

DATA_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

results = {}

## Exp - 7 : Scalability in terms of data

for model_name, embed_fn in models.items():
    print(f"\nRunning H7 for {model_name}...")
    results[model_name] = {
        "compactness": [],
        "margin": [],
        "intrinsic_dim": [],
        "neighbor_stability": []
    }

    prev_neighbors = None

    for frac in DATA_FRACTIONS:
        subset = image_files[:max(30, int(len(image_files)*frac))]

        embeddings = []

        for path in subset:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            emb = embed_fn(tensor)
            emb_np = emb.detach().cpu().numpy().reshape(-1)
            embeddings.append(emb_np)

        embeddings = np.vstack(embeddings)

        ## Compactness (intra-class tightness proxy)
        dist_matrix = pairwise_distances(embeddings)
        compactness = np.mean(dist_matrix)

        ## Margin (inter-sample spread)
        margin = np.percentile(dist_matrix, 90) - np.percentile(dist_matrix, 10)

        ## Intrinsic dimensionality
        pca = PCA(n_components=0.95)
        pca.fit(embeddings)
        intrinsic_dim = pca.n_components_

        # Neighborhood Stability
        nn = NearestNeighbors(n_neighbors=5).fit(embeddings)
        neighbors = nn.kneighbors(return_distance=False)

        if prev_neighbors is None:
            stability = 1.0
        else:
            overlap = np.mean([
                len(set(neighbors[i]).intersection(prev_neighbors[i]))/5
                for i in range(min(len(neighbors), len(prev_neighbors)))
            ])
            stability = overlap

        prev_neighbors = neighbors

        ## Store
        results[model_name]["compactness"].append(float(compactness))
        results[model_name]["margin"].append(float(margin))
        results[model_name]["intrinsic_dim"].append(float(intrinsic_dim))
        results[model_name]["neighbor_stability"].append(float(stability))

## Save results as json

json_path = os.path.join(RESULTS_DIR, "h7_scaling_geometry.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)

## Plots 

fractions = DATA_FRACTIONS

## Plot 1 — Compactness
plt.figure(figsize=(8,5))
for m in results:
    plt.plot(fractions, results[m]["compactness"], label=m)
plt.title("H7 — Cluster Compactness vs Data Scale")
plt.xlabel("Dataset Fraction")
plt.ylabel("Compactness (Lower = Better)")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "compactness_curve.png"))
plt.close()

## Plot 2 — Margin
plt.figure(figsize=(8,5))
for m in results:
    plt.plot(fractions, results[m]["margin"], label=m)
plt.title("H7 — Inter-Class Margin vs Scale")
plt.xlabel("Dataset Fraction")
plt.ylabel("Margin (Higher = Better)")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "margin_curve.png"))
plt.close()

## Plot 3 — Intrinsic Dimensionality
plt.figure(figsize=(8,5))
for m in results:
    plt.plot(fractions, results[m]["intrinsic_dim"], label=m)
plt.title("H7 — Intrinsic Dimensionality Growth")
plt.xlabel("Dataset Fraction")
plt.ylabel("Effective Dimensions")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "intrinsic_dim.png"))
plt.close()

## Plot 4 — Neighborhood Stability
plt.figure(figsize=(8,5))
for m in results:
    plt.plot(fractions, results[m]["neighbor_stability"], label=m)
plt.title("H7 — Neighborhood Stability Under Scale")
plt.xlabel("Dataset Fraction")
plt.ylabel("Stability Score")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "neighbor_stability.png"))
plt.close()

print("H7 Complete — Results saved to plots/h7/")
