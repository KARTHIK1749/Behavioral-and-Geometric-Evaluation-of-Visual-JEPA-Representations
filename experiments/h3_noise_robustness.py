"""
H3 — Noise Robustness (ImageNet-C Style)

Core Claim:
JEPA degrades differently — not just slower, but structurally distinct.

Measures:
- Robustness curves
- AURC (Area Under Robustness Curve)
- Degradation slope + curvature
"""

import os, sys, json
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.stats import linregress

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load Embedding Project

from utils.embeddings import embed_image
from torchvision.models import resnet50
import timm
from transformers import CLIPModel, CLIPProcessor
from metrics.cosine import cosine_similarity

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

## Corruption Functions
def gaussian_noise(img, severity):
    noise = np.random.normal(0, severity * 20, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def blur(img, severity):
    k = 3 + severity * 2
    return cv2.GaussianBlur(img, (k,k), 0)

def jpeg(img, severity):
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90 - severity*15])
    return cv2.imdecode(enc, 1)

def occlusion(img, severity):
    h, w = img.shape[:2]

    ## min occlusion size
    min_size = max(5, int(0.05*w))
    max_size = int(0.4 * w)

    size = int(min_size + severity * (max_size - min_size) / 5)
    
    ## clamping the size to image dims
    size = min(size , w-2 , h-2)

    if size<=1 :
        return img
    
    x = np.random.randint(0, w-size)
    y = np.random.randint(0, h-size)

    img =img.copy()
    img[y:y+size, x:x+size] = 0
    return img

def motion_blur(img, severity):
    k = 3 + severity * 3
    kernel = np.zeros((k, k))
    kernel[k//2, :] = 1
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

CORRUPTIONS = {
    "Gaussian": gaussian_noise,
    "Blur": blur,
    "JPEG": jpeg,
    "Occlusion": occlusion,
    "MotionBlur": motion_blur,
}

SEVERITY_LEVELS = [0,1,2,3,4,5]

## Exp -3 : Noise Robustness

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".png",".jpeg")):
            image_files.append(os.path.join(root, f))

results = {}

for model_name, embed_fn in models.items():
    print(f"\nRunning H3 for {model_name}...")
    results[model_name] = {}

    for cname, corrupt_fn in CORRUPTIONS.items():
        sims_by_severity = []

        for severity in SEVERITY_LEVELS:
            sims = []

            for path in image_files[:200]:  # cap for speed
                img = np.array(Image.open(path).convert("RGB"))

                corrupted = img.copy()
                if severity > 0:
                    corrupted = corrupt_fn(corrupted, severity=4)
                    corrupted = np.clip(corrupted, 0, 255).astype(np.uint8)

                orig_tensor = transform(Image.fromarray(img)).unsqueeze(0).to(device)
                corr_tensor = transform(Image.fromarray(corrupted)).unsqueeze(0).to(device)

                orig_emb = embed_fn(orig_tensor)
                corr_emb = embed_fn(corr_tensor)

                sims.append(cosine_similarity(orig_emb, corr_emb))

            sims_by_severity.append(np.mean(sims))

        ## Slope and curvature
        x = np.array(SEVERITY_LEVELS)
        y = np.array(sims_by_severity)

        slope = linregress(x, y).slope
        curvature = np.mean(np.diff(np.diff(y)))  # 2nd derivative approx
        aurc = np.trapz(y, x)

        results[model_name][cname] = {
            "curve": sims_by_severity,
            "slope": float(slope),
            "curvature": float(curvature),
            "aurc": float(aurc)
        }

## Save results in json

json_path = os.path.join(RESULTS_DIR, "h3_noise_robustness.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)

## Plots

## Plot 1 — Robustness Curves
for cname in CORRUPTIONS.keys():
    plt.figure(figsize=(8,5))

    for model in results:
        plt.plot(SEVERITY_LEVELS, results[model][cname]["curve"], label=model)

    plt.title(f"H3 — Robustness Curve ({cname})")
    plt.xlabel("Severity")
    plt.ylabel("Similarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{cname}_curves.png"))
    plt.close()

## Plot 2 — AURC Comparison
plt.figure(figsize=(7,5))

aurc_means = []
labels = []

for model in results:
    aurcs = [results[model][c]["aurc"] for c in CORRUPTIONS.keys()]
    aurc_means.append(np.mean(aurcs))
    labels.append(model)

plt.bar(labels, aurc_means)
plt.title("H3 — Mean Robustness (AURC)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "aurc_comparison.png"))
plt.close()

## Plot 3 — Geometry Signature
plt.figure(figsize=(7,5))

for model in results:
    slopes = [results[model][c]["slope"] for c in CORRUPTIONS.keys()]
    curvs = [results[model][c]["curvature"] for c in CORRUPTIONS.keys()]

    plt.scatter(slopes, curvs, label=model)

plt.axhline(0, linestyle="--")
plt.axvline(0, linestyle="--")
plt.xlabel("Slope (Decay Rate)")
plt.ylabel("Curvature (Failure Shape)")
plt.title("H3 — Degradation Geometry")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "geometry_signature.png"))
plt.close()

## Plot 4 - Failur mode per model heatmap
import seaborn as sns

heatmap_data = []

model_names = list(results.keys())
corr_names = list(CORRUPTIONS.keys())

for model in model_names:
    row = []
    for cname in corr_names:
        row.append(results[model][cname]["aurc"])
    heatmap_data.append(row)

plt.figure(figsize=(9,6))
sns.heatmap(
    heatmap_data,
    xticklabels=corr_names,
    yticklabels=model_names,
    cmap="viridis",
    annot=True,
    fmt=".3f"
)

plt.title("H3 — Failure Mode Heatmap (Lower = Worse)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "failure_mode_heatmap.png"))
plt.close()

## Plot 5 - Example degradation curves
sample_img = np.array(Image.open(image_files[0]).convert("RGB"))

plt.figure(figsize=(14,5))

for i, (cname, corrupt_fn) in enumerate(CORRUPTIONS.items()):
    corrupted = corrupt_fn(sample_img.copy(), severity=4)

    plt.subplot(2, len(CORRUPTIONS), i+1)
    plt.imshow(sample_img)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(2, len(CORRUPTIONS), i+1+len(CORRUPTIONS))
    plt.imshow(corrupted)
    plt.axis("off")
    plt.title(cname)

plt.suptitle("H3 — Example Corruption Visualizations")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "example_degradations.png"))
plt.close()


print("H3 Complete — Results saved to plots/h3/")

