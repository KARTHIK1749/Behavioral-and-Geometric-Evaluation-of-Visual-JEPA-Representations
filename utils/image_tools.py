import os
from PIL import Image
import torchvision.transforms as T
import torch

PROCESSED_DIR = "data/processed"
SYNTHETIC_DIR = "data/synthetic"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

def save_blur(img, name):
    blur = T.GaussianBlur(kernel_size=7)(img)
    blur.save(os.path.join(PROCESSED_DIR, f"blur_{name}"))
    return blur

def save_bg_blur(img, name):
    blur = T.GaussianBlur(kernel_size=15)(img)
    blur.save(os.path.join(PROCESSED_DIR, f"bg_blur_{name}"))
    return blur

def save_noise(img_tensor, name):
    noisy = img_tensor + 0.15 * torch.randn_like(img_tensor)
    noisy_img = T.ToPILImage()(noisy.squeeze().clamp(0,1))
    noisy_img.save(os.path.join(SYNTHETIC_DIR, f"noise_{name}"))
    return noisy_img
