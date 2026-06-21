"""
H4 — Temporal Stability (Real Video Frames)

Claim:
JEPA maintains more temporally stable representations than vision & video baselines.

Measures:
- Frame-to-frame embedding drift
- Temporal Stability Score (TSS)
- Drift variance
- Failure spikes
- Temporal coherence plots
"""

import os, sys, json
import cv2
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

VIDEO_DIR = os.path.join(PROJECT_ROOT, "data/videos")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/h4")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Load Models

from utils.embeddings import embed_image  # JEPA
from torchvision.models import resnet50
import timm
from transformers import CLIPModel, CLIPProcessor
from transformers import VideoMAEModel, VideoMAEImageProcessor
from metrics.cosine import cosine_similarity as cosine_sim
models = {}

## JEPA
models["JEPA"] = embed_image

## ResNet18
resnet = resnet50(pretrained=True).to(device).eval()
models["ResNet"] = lambda x: resnet(x.to(device))

## DINO
dino = timm.create_model("vit_base_patch16_224", pretrained=True).to(device).eval()
models["DINO"] = lambda x: dino(x.to(device))

## CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_clip(img_tensor):
    img = T.ToPILImage()(img_tensor.squeeze(0))
    inputs = clip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        return clip_model.get_image_features(**inputs)

models["CLIP"] = embed_clip

## VideoMAE
videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
videomae_proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

def embed_videomae(frames):
    pil_frames = [Image.fromarray(f) for f in frames]

    inputs = videomae_proc(
        pil_frames,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = videomae(**inputs)

    ## pooled video embedding
    return outputs.last_hidden_state.mean(dim=1)


models["VideoMAE"] = embed_videomae

## Transforms

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor()
])


## Extract frames 

def extract_frames(video_path, target_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    idxs = np.linspace(0, total-1, target_frames).astype(int)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id in idxs:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

## Exp -4 : Temporal Stability

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]

results = {}

for model_name, embed_fn in models.items():
    print(f"\nRunning H4 for {model_name}...")
    drift_all = []

    for vid in video_files:
        frames = extract_frames(os.path.join(VIDEO_DIR, vid))

        if len(frames) < 5:
            continue

        embeddings = []

        for f in frames:
            tensor = transform(Image.fromarray(f)).unsqueeze(0).to(device)

            if model_name == "VideoMAE":
                embeddings.append(None)
            else:
                emb = embed_fn(tensor)
                embeddings.append(emb)

        ## VideoMAE batch embed
        if model_name == "VideoMAE":
            video_emb = embed_fn(frames)
            embeddings = [video_emb[:,i].unsqueeze(0) for i in range(video_emb.shape[1])]

        ## Drift calculation
        drifts = []
        for i in range(len(embeddings)-1):
            sim = cosine_sim(embeddings[i], embeddings[i+1])
            drifts.append(sim)

        drift_all.append(drifts)

    ## Flatten
    if len(drift_all) == 0:
        print(f"No valid videos for {model_name}, skipping it...")
        continue
    drift_flat = np.concatenate(drift_all)

    results[model_name] = {
        "mean_stability": float(np.mean(drift_flat)),
        "variance": float(np.var(drift_flat)),
        "curve": drift_flat.tolist()
    }

## Save the results as json 

json_path = os.path.join(RESULTS_DIR, "h4_temporal_stability.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("Saved →", json_path)

## Plots

## Plot 1 — Drift Curve
plt.figure(figsize=(9,5))
for model in results:
    plt.plot(results[model]["curve"], label=model)

plt.title("H4 — Temporal Drift Curve")
plt.xlabel("Frame Index")
plt.ylabel("Embedding Similarity")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "temporal_drift_curve.png"))
plt.close()

## Plot 2 — Mean Stability Bar
plt.figure(figsize=(7,5))
names = list(results.keys())
means = [results[m]["mean_stability"] for m in names]

plt.bar(names, means)
plt.title("H4 — Temporal Stability Score (TSS)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "temporal_stability_bar.png"))
plt.close()

## Plot 3 — Drift Variance
plt.figure(figsize=(7,5))
vars_ = [results[m]["variance"] for m in names]

plt.bar(names, vars_)
plt.title("H4 — Temporal Drift Variance (Lower = Smoother)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "drift_variance.png"))
plt.close()

## Plot 4 — Failure Heatmap
import seaborn as sns

heat_data = []
for model in names:
    heat_data.append(results[model]["curve"][:50])

plt.figure(figsize=(10,5))
sns.heatmap(heat_data, yticklabels=names, cmap="viridis")

plt.title("H4 — Temporal Failure Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "temporal_failure_heatmap.png"))
plt.close()

print("H4 Complete — Results saved to plots/h4/")
