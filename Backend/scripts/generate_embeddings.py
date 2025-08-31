import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" 
IMAGE_DIR = DATA_DIR / "images"
INPUT_JSON = DATA_DIR / "product.json"
OUTPUT_JSON = DATA_DIR / "product_embeddings.json"

print("Looking for images in:", IMAGE_DIR.resolve())
print("Saving embeddings to:", OUTPUT_JSON.resolve())

print("INPUT_JSON exists:", INPUT_JSON.exists())
print("IMAGE_DIR exists:", IMAGE_DIR.exists())
print("Files in IMAGE_DIR:", [p.name for p in IMAGE_DIR.iterdir() if p.is_file()])


metadata = {}
if INPUT_JSON.exists():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        products = json.load(f)
        for p in products:
            img_path = p.get("image_path")
            if not img_path:
                continue
           
            filename = Path(img_path.strip()).name.lower()
            metadata[filename] = p

print("Metadata keys loaded:", list(metadata.keys())[:10]) 


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()


if not IMAGE_DIR.exists():
    raise SystemExit(f"Images folder not found: {IMAGE_DIR.resolve()}")

image_files = [
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".avif"}
]
if not image_files:
    raise SystemExit(f"No images found in {IMAGE_DIR.resolve()}")

print(f"Found {len(image_files)} images")


results = []

for idx, img_path in enumerate(tqdm(image_files, desc="Embedding images")):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path.name}: {e}")
        continue

    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    emb_list = emb.squeeze(0).cpu().numpy().tolist()

    
    base_meta = metadata.get(img_path.name.lower())
    if not base_meta:
        print(f"⚠️ No metadata found for {img_path.name}, skipping...")
        continue

   
    rel_path = img_path.resolve().relative_to(DATA_DIR.resolve())
    static_path = f"/static/{rel_path.as_posix()}"

    results.append({
        "id": idx + 1,
        "name": base_meta.get("name", img_path.stem.replace("_", " ").title()),
        "category": base_meta.get("category", "Unknown"),
        "brand": base_meta.get("brand", "Unknown"),
        "price": base_meta.get("price", "0"),
        "color": base_meta.get("color", "Unknown"),
        "description": base_meta.get("description", ""),
        "image_path": static_path,
        "embedding": emb_list
    })


if results:
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(results)} embeddings to {OUTPUT_JSON}")
else:
    print("⚠️ No embeddings were generated. Check metadata and image files.")
