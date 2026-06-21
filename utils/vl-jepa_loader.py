import sys
import os
import torch

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add VL-JEPA ROOT to Python path
VL_JEPA_ROOT = os.path.join(PROJECT_ROOT, "external", "vl-jepa")
sys.path.append(VL_JEPA_ROOT)

# Import from VL-JEPA package
from src.models.vision_transformer import vit_small

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vit_small(pretrained=True)
model = model.to(device)
model.eval()

print("Loaded vit_small successfully")
