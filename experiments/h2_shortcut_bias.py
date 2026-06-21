"""
H2 — Shortcut Reliance (Causal Foreground/Background Bias)

Core Claim:
JEPA is more object-centric and less background-biased than CNN / CLIP / DINO.

We use Segment Anything (FAST SamPredictor) to isolate subject vs background.

Outputs:
- SDI per model
- Effect size
- Statistical evidence
- 4 plots + Example visualization
"""

import sys, os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from scipy.stats import ttest_rel
import cv2

## Project paths

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

RAW_DIR = os.path.join(PROJECT_ROOT, "data/raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h2")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Laod segment anything model (we are using the small vit_b model here)

from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint=os.path.join(MODEL_DIR, "sam_vit_b.pth"))
sam = sam.to(device)
predictor = SamPredictor(sam)

## Load Embedding Models

from utils.embeddings import embed_image  # JEPA
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


## Load dataset of images (data/raw)

image_files = []
for root, _, files in os.walk(RAW_DIR):
    for f in files:
        if f.lower().endswith((".jpg",".jpeg",".png")):
            image_files.append(os.path.join(root, f))

print("Total images found:", len(image_files))
print("Sample files:", image_files[:3])

## Mask cache (critical speed fix)

mask_cache = {}

def get_mask(img_path):
    if img_path in mask_cache:
        return mask_cache[img_path]

    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_small = cv2.resize(img_np, (512, 512))
    predictor.set_image(img_small)

    h, w = img_small.shape[:2]
    input_point = np.array([[w//2, h//2]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    best_mask = masks[np.argmax(scores)]

    mask_resized = cv2.resize(
        best_mask.astype(np.uint8),
        (img_np.shape[1], img_np.shape[0])
    ) > 0

    mask_cache[img_path] = mask_resized
    return mask_resized

## Exp -2 : Shortcut Reliance

results = {}
example_saved = False

for model_name, embed_fn in models.items():
    print(f"\nRunning H2 for {model_name}...")

    fg_scores = []
    bg_scores = []
    valid_count = 0

    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"{model_name}: Processing {i}/{len(image_files)}")

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        mask = get_mask(img_path)

        fg_img = img_np.copy()
        bg_img = img_np.copy()

        fg_img[~mask] = [255,255,255]
        bg_img[mask] = [255,255,255]

        ## Save example visualization
        if not example_saved:
            vis = np.hstack([img_np, fg_img, bg_img])
            cv2.imwrite(os.path.join(PLOTS_DIR, "example_fg_bg.png"), vis[:,:,::-1])
            example_saved = True

        orig_tensor = transform(img).unsqueeze(0).to(device)
        fg_tensor = transform(Image.fromarray(fg_img)).unsqueeze(0).to(device)
        bg_tensor = transform(Image.fromarray(bg_img)).unsqueeze(0).to(device)

        orig_emb = embed_fn(orig_tensor)
        fg_emb = embed_fn(fg_tensor)
        bg_emb = embed_fn(bg_tensor)

        fg_sim = cosine_similarity(orig_emb, fg_emb)
        bg_sim = cosine_similarity(orig_emb, bg_emb)

        fg_scores.append(fg_sim)
        bg_scores.append(bg_sim)
        valid_count += 1

    fg_scores = np.array(fg_scores)
    bg_scores = np.array(bg_scores)

    if len(fg_scores) == 0:
        print(f"Skipping {model_name}: No valid samples.")
        continue

    print(f"{model_name}: Valid samples =", valid_count)

    ## Shortcut Dependence Index (SDI)
    sdi = fg_scores - bg_scores

    pooled_std = np.sqrt((fg_scores.std()**2 + bg_scores.std()**2) / 2)
    cohens_d = float(sdi.mean() / pooled_std) if pooled_std > 1e-8 else 0.0

    p_val = ttest_rel(bg_scores, fg_scores).pvalue if len(bg_scores) > 1 else 1.0

    results[model_name] = {
        "foreground_similarity": fg_scores.tolist(),
        "background_similarity": bg_scores.tolist(),
        "sdi": sdi.tolist(),
        "mean_sdi": float(sdi.mean()),
        "cohens_d": float(cohens_d),
        "p_value": float(p_val)
    }

## Save results as json

json_path = os.path.join(RESULTS_DIR, "h2_shortcut_reliance.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved →", json_path)

## Plots

models_list = list(results.keys())

## Plot 1 — SDI by Model
plt.figure(figsize=(7,5))
plt.bar(models_list, [results[m]["mean_sdi"] for m in models_list])
plt.title("H2 — Shortcut Dependence Index by Model")
plt.ylabel("SDI (Higher = More Object-Centric)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sdi_by_model.png"))
plt.close()

## Plot 2 — SDI Distribution
plt.figure(figsize=(8,5))
for m in models_list:
    plt.hist(results[m]["sdi"], alpha=0.4, bins=30, label=m)
plt.legend()
plt.title("H2 — SDI Distribution Across Images")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "sdi_distribution.png"))
plt.close()

## Plot 3 — FG vs BG Similarity
plt.figure(figsize=(7,5))
fg_means = [np.mean(results[m]["foreground_similarity"]) for m in models_list]
bg_means = [np.mean(results[m]["background_similarity"]) for m in models_list]

x = np.arange(len(models_list))
plt.bar(x-0.2, fg_means, width=0.4, label="Foreground")
plt.bar(x+0.2, bg_means, width=0.4, label="Background")

plt.xticks(x, models_list)
plt.legend()
plt.title("H2 — Subject vs Background Encoding")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fg_vs_bg_similarity.png"))
plt.close()

## Plot 4 — Effect Sizes
plt.figure(figsize=(7,5))
plt.bar(models_list, [results[m]["cohens_d"] for m in models_list])
plt.title("H2 — Effect Size (Cohen’s d)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "effect_sizes.png"))
plt.close()

print("H2 Complete — Results saved to plots/h2/")
