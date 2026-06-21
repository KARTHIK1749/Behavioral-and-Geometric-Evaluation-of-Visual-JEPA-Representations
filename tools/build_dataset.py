import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import hashlib
import random
from dotenv import load_dotenv

## Load environment variables
load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

if not PEXELS_API_KEY:
    raise RuntimeError("Missing the pixel api key in .env file")

# Configurayions 
SAVE_ROOT = "data/raw"
IMG_SIZE = (224, 224)

CATEGORY_PLAN = {
    "humans": 70,
    "animals": 65,
    "vehicles": 45,
    "indoor": 45,
    "outdoor": 45,
    "objects": 45,
    "ambiguous": 40,
    "abstract": 29
}

SEARCH_QUERIES = {
    "humans": ["people walking", "person working", "group of people", "sports activity"],
    "animals": ["dog", "cat", "wild animal", "bird"],
    "vehicles": ["car", "bus", "motorcycle", "train"],
    "indoor": ["living room", "office room", "kitchen"],
    "outdoor": ["street", "nature", "mountains", "park"],
    "objects": ["tools", "laptop", "phone", "daily objects"],
    "ambiguous": ["crowded scene", "multiple people", "busy market"],
    "abstract": ["abstract art", "weird art", "blurry scene"]
}

HEADERS = {
    "Authorization": PEXELS_API_KEY
}

## Utility Functions 
def ensure_dirs():
    for cat in CATEGORY_PLAN:
        os.makedirs(os.path.join(SAVE_ROOT, cat), exist_ok=True)

def hash_image(data):
    return hashlib.md5(data).hexdigest()

def save_image(data, path):
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img.save(path)

# Download Functions 
def download_images(category, target_count):
    print(f"\nDownloading {target_count} images for {category}")

    save_dir = os.path.join(SAVE_ROOT, category)
    existing_hashes = set()

    count = len(os.listdir(save_dir))

    pbar = tqdm(total=target_count)

    while count < target_count:
        query = random.choice(SEARCH_QUERIES[category])

        url = f"https://api.pexels.com/v1/search?query={query}&per_page=40"

        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = r.json()

            if "photos" not in data:
                continue

            for photo in data["photos"]:
                img_url = photo["src"]["medium"]

                img_data = requests.get(img_url, timeout=10).content
                img_hash = hash_image(img_data)

                if img_hash in existing_hashes:
                    continue

                existing_hashes.add(img_hash)

                filename = f"{category}_{count}.jpg"
                save_path = os.path.join(save_dir, filename)

                save_image(img_data, save_path)

                count += 1
                pbar.update(1)

                if count >= target_count:
                    break

        except:
            continue

    pbar.close()

if __name__ == "__main__":
    ensure_dirs()

    for category, count in CATEGORY_PLAN.items():
        download_images(category, count)

    print("\n Dataset build completed.")
