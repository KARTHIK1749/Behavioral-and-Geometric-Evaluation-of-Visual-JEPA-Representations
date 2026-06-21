"""
H5 — Ambiguity & Multi-Interpretation Probing

Claim:
JEPA preserves multiple plausible interpretations instead of collapsing representation.

Tests:
- Occlusion ambiguity
- Figure-ground confusion
- Multi-object salience competition
- Illusion / perceptual conflict

Measures:
- Embedding clustering
- Bifurcation score
- Representation spread
- Interpretation stability
"""

import os, sys, json
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h5")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


from utils.embeddings import embed_image
from torchvision.models import resnet50
import timm
from transformers import CLIPModel, CLIPProcessor
import cv2

## Load models
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

## Ambiguity Generators

def occlusion(img):
    h, w = img.shape[:2]
    size = int(0.3 * w)
    x = np.random.randint(0, w-size)
    y = np.random.randint(0, h-size)
    img[y:y+size, x:x+size] = 0
    return img

def figure_ground(img):
    blurred = cv2.GaussianBlur(img, (25,25), 0)
    return np.where(np.random.rand(*img.shape[:2],1) > 0.5, img, blurred)

def multi_object(img):
    flip = np.fliplr(img)
    return (img * 0.5 + flip * 0.5).astype(np.uint8)

def illusion(img):
    shifted = np.roll(img, shift=30, axis=1)
    return (img * 0.6 + shifted * 0.4).astype(np.uint8)

AMBIGUITY_TYPES = {
    "Occlusion": occlusion,
    "FigureGround": figure_ground,
    "MultiObject": multi_object,
    "Illusion": illusion
}

## Exp - 5 : Ambiguity testing

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            image_files.append(os.path.join(root, f))

image_files = image_files[:150]  ## cap for speed

results = {}

for model_name, embed_fn in models.items():
    print(f"\nRunning H5 for {model_name}...")
    results[model_name] = {}

    for amb_name, amb_fn in AMBIGUITY_TYPES.items():
        embeddings = []

        for path in image_files:
            img = np.array(Image.open(path).convert("RGB"))
            amb_img = amb_fn(img.copy())

            tensor = transform(Image.fromarray(amb_img)).unsqueeze(0).to(device)
            emb = embed_fn(tensor).detach().cpu().numpy().flatten()
            embeddings.append(emb)

        embeddings = np.array(embeddings)

        ## Clustering
        kmeans = KMeans(n_clusters=2).fit(embeddings)
        cluster_labels = kmeans.labels_

        silhouette = silhouette_score(embeddings, cluster_labels)
        spread = float(np.mean(pdist(embeddings)))
        bifurcation = float(np.linalg.norm(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]))

        results[model_name][amb_name] = {
            "silhouette": float(silhouette),
            "spread": spread,
            "bifurcation": bifurcation,
            "cluster_labels": cluster_labels.tolist()
        }

## Save json 
json_path = os.path.join(RESULTS_DIR, "h5_ambiguity.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)


## Plots

## Plot 1 — Cluster Separability
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][a]["silhouette"] for a in AMBIGUITY_TYPES]
    plt.plot(AMBIGUITY_TYPES.keys(), vals, label=model)

plt.title("H5 — Cluster Separability Under Ambiguity")
plt.ylabel("Silhouette Score (Higher = Multi-Meaning)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cluster_separability.png"))
plt.close()

## Plot 2 — Representation Spread
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][a]["spread"] for a in AMBIGUITY_TYPES]
    plt.plot(AMBIGUITY_TYPES.keys(), vals, label=model)

plt.title("H5 — Embedding Spread (Uncertainty)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "embedding_spread.png"))
plt.close()

## Plot 3 — Bifurcation Strength
plt.figure(figsize=(8,5))
for model in results:
    vals = [results[model][a]["bifurcation"] for a in AMBIGUITY_TYPES]
    plt.plot(AMBIGUITY_TYPES.keys(), vals, label=model)

plt.title("H5 — Representation Bifurcation Strength")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "bifurcation_strength.png"))
plt.close()

## Plot 4 — Model Collapse Score
plt.figure(figsize=(7,5))

collapse_scores = []
labels = []

for model in results:
    mean_sil = np.mean([results[model][a]["silhouette"] for a in AMBIGUITY_TYPES])
    collapse_scores.append(mean_sil)
    labels.append(model)

plt.bar(labels, collapse_scores)
plt.title("H5 — Representation Collapse Score (Lower = Worse)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "collapse_score.png"))
plt.close()

print("H5 Complete — Results saved to plots/h5/")
