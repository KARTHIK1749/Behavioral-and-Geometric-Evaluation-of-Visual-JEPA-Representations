"""
H1 — Semantic Faithfulness (Comparative Diagnostic)

Does JEPA preserve meaning better or worse than other embeddings under corruption?

Models:
JEPA vs CLIP vs DINO vs ResNet

Metrics:
- Similarity decay curves
- Degradation slope
- Effect size (Cohen's d)
- Statistical significance (t-test)
"""

import sys, os, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from scipy.stats import ttest_rel

from utils.embeddings import embed_image
from metrics.cosine import cosine_similarity

import timm
from torchvision.models import resnet18
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the models
models = {}

## JEPA
models["JEPA"] = embed_image

## CLIP(hf)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_clip(img_tensor):
    img = T.ToPILImage()(img_tensor.squeeze(0))
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    return emb

models["CLIP"] = embed_clip

## DINO(small)
dino = timm.create_model("vit_small_patch16_224", pretrained=True).to(device).eval()

def embed_dino(img_tensor):
    with torch.no_grad():
        return dino(img_tensor.to(device))

models["DINO"] = embed_dino

## ResNet18 (light)
resnet = resnet18(pretrained=True).to(device).eval()

def embed_resnet(img_tensor):
    with torch.no_grad():
        return resnet(img_tensor.to(device))

models["ResNet"] = embed_resnet


# Load dataset(images from raw folder)
RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            image_files.append(os.path.join(root, f))

if len(image_files) < 10:
    raise RuntimeError("Too few images found in RAW_DIR")

print("Images found:", len(image_files))

## Transform 
transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])

blur_levels = [0, 2, 4, 6, 8]
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

results = {}

## Exp -1 : Semantic faithfulness
for model_name, embed_fn in models.items():
    blur_curves = []
    noise_curves = []

    print("Running:", model_name)

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        base_tensor = transform(img).unsqueeze(0).to(device)

        base_emb = embed_fn(base_tensor)

        blur_sims = []
        noise_sims = []

        ## Blur corruption
        for b in blur_levels:
            blur_img = img if b == 0 else T.GaussianBlur(kernel_size=7)(img)
            blur_tensor = transform(blur_img).unsqueeze(0).to(device)
            blur_emb = embed_fn(blur_tensor)
            blur_sims.append(cosine_similarity(base_emb, blur_emb))

        ## Noise corruption
        for n in noise_levels:
            noisy_tensor = base_tensor + torch.randn_like(base_tensor) * n
            noise_emb = embed_fn(noisy_tensor)
            noise_sims.append(cosine_similarity(base_emb, noise_emb))

        blur_curves.append(blur_sims)
        noise_curves.append(noise_sims)

    blur_curves = np.array(blur_curves)
    noise_curves = np.array(noise_curves)

    blur_mean = blur_curves.mean(axis=0)
    noise_mean = noise_curves.mean(axis=0)

    blur_slope = np.polyfit(blur_levels, blur_mean, 1)[0]
    noise_slope = np.polyfit(noise_levels, noise_mean, 1)[0]

    semantic_margin = blur_mean.mean() - noise_mean.mean()

    pooled_std = np.sqrt((blur_curves.std()**2 + noise_curves.std()**2) / 2)
    cohens_d = semantic_margin / pooled_std if pooled_std > 0 else 0

    t_stat, p_val = ttest_rel(blur_curves.flatten(), noise_curves.flatten())

    results[model_name] = {
        "blur_curve_mean": blur_mean.tolist(),
        "noise_curve_mean": noise_mean.tolist(),
        "blur_slope": float(blur_slope),
        "noise_slope": float(noise_slope),
        "semantic_margin": float(semantic_margin),
        "cohens_d": float(cohens_d),
        "p_value": float(p_val)
    }

## Save the results (values) in json format
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

json_path = os.path.join(RESULTS_DIR, "h1_semantic_faithfulness.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)

## Plots
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h1")
os.makedirs(PLOTS_DIR, exist_ok=True)

model_names = list(results.keys())

## Blur curves
plt.figure()
for m in model_names:
    plt.plot(blur_levels, results[m]["blur_curve_mean"], label=m)
plt.title("H1 — Blur Degradation Curves")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "blur_decay.png"))
plt.close()

## Noise curves
plt.figure()
for m in model_names:
    plt.plot(noise_levels, results[m]["noise_curve_mean"], label=m)
plt.title("H1 — Noise Degradation Curves")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "noise_decay.png"))
plt.close()

## Slopes
plt.figure()
plt.bar(model_names, [results[m]["noise_slope"] for m in model_names])
plt.title("H1 — Noise Degradation Slopes")
plt.savefig(os.path.join(PLOTS_DIR, "slopes.png"))
plt.close()

## Effect size
plt.figure()
plt.bar(model_names, [results[m]["cohens_d"] for m in model_names])
plt.title("H1 — Effect Size (Cohen's d)")
plt.savefig(os.path.join(PLOTS_DIR, "effect_size.png"))
plt.close()

print("H1 plots saved → plots/h1/")
print("H1 Comparative Scientific Diagnostic Complete.")
