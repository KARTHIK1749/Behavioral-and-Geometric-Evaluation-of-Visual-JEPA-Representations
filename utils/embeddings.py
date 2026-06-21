import sys
import os
import torch
from transformers import AutoTokenizer , AutoModel
## here we convert images to jepa embeddings from the vl-jepa repo

# Resolve project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add VL-JEPA root to path
VL_JEPA_ROOT = os.path.join(PROJECT_ROOT, "external", "vl-jepa")
sys.path.append(VL_JEPA_ROOT)

# Import JEPA vision transformer
from src.models.vision_transformer import vit_small

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained JEPA model
model = vit_small(pretrained=True)
model = model.to(device)
model.eval()

TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
text_model.eval()

def embed_text(text):
    """
    Input: string caption
    Output: text embedding vector
    """
    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = text_model(**tokens)

        # Mean pooling across tokens
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding

def embed_image(image_tensor):
    """
    Input: torch tensor image (B, 3, 224, 224)
    Output: JEPA embedding vector
    """
    with torch.no_grad():
        embedding = model(image_tensor.to(device))
    return embedding


# def cosine_similarity(a, b):
#     sim = torch.nn.functional.cosine_similarity(a,b,dim= -1)
#     return sim.mean().item()
