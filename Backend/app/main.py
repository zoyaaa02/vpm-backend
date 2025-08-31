
import io
import json
from pathlib import Path

import numpy as np
import requests
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from fastapi import Query
from typing import Optional
import uuid

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data"
EMBEDDINGS_FILE = DATA_DIR / "product_embeddings.json" 

TOP_K = 7


app = FastAPI(title="Visual Product Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"🔎 Looking for static dir at: {DATA_DIR}")
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")
    print("✅ Mounted /static ->", DATA_DIR)
else:
    print("⚠️ Warning: static data folder not found:", DATA_DIR)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model on {device}...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

print("Loading product embeddings from:", EMBEDDINGS_FILE)
if not EMBEDDINGS_FILE.exists():
    raise RuntimeError(
        f"Embeddings file not found at {EMBEDDINGS_FILE}. "
        f"Make sure you generated it and placed it in backend/data/."
    )

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    products = json.load(f)

if not isinstance(products, list) or len(products) == 0:
    raise RuntimeError("Embeddings JSON is empty or malformed.")

product_embeddings = np.array([p["embedding"] for p in products], dtype=np.float32)
product_metadata = [{k: v for k, v in p.items() if k != "embedding"} for p in products]

print(f"✅ Loaded {len(product_metadata)} products")


def get_image_embedding(image: Image.Image) -> np.ndarray:
    # Wrap in list and add padding=True to prevent tensor errors
    inputs = processor(
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # Move tensors safely to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

    return emb.cpu().numpy().flatten()



SIMILARITY_THRESHOLD = 0.7 

def search_similar_products(query_emb: np.ndarray, similarity_threshold: float = 0.8):
    sims = np.dot(product_embeddings, query_emb)
    results = []

    for idx, score in enumerate(sims):
        if score >= similarity_threshold:  
            meta = product_metadata[idx]
            results.append({
                "id": meta.get("id", idx + 1),
                "name": meta.get("name"),
                "category": meta.get("category"),
                "brand": meta.get("brand"),
                "price": meta.get("price"),
                "color": meta.get("color"),
                "description": meta.get("description"),
                "image_path": meta.get("image_path"),
                "score": float(score),
            })

    # Sort results by descending similarity
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


class URLRequest(BaseModel):
    url: str

@app.get("/")
def root():
    return {"message": "Welcome to Visual Product Matcher API 🚀", 
            "endpoints": ["/ping", "/search/file", "/search/url", "/products", "/static/..."]}

@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/search/file")
async def search_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    query_emb = get_image_embedding(image)
    results = search_similar_products(query_emb, similarity_threshold=0.6)

    if len(results) == 0:
        return {"results": [], "message": "No similar product found"}

    return {"results": results}


@app.post("/search/url")
def search_url(req: URLRequest):
    try:
        r = requests.get(req.url, timeout=15)
        r.raise_for_status()
        image = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not fetch or decode image from URL.")

    query_emb = get_image_embedding(image)
    results = search_similar_products(query_emb, similarity_threshold=0.6)

    if len(results) == 0:
        return {"results": [], "message": "No similar product found"}

    return {"results": results}



SEARCH_RESULTS = {}

@app.post("/search")
async def start_search(file: UploadFile = File(None), image_url: str = None):
    if file:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    elif image_url:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
        image = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    query_emb = get_image_embedding(image)
    results = search_similar_products(query_emb)

    query_id = str(uuid.uuid4())
    SEARCH_RESULTS[query_id] = results

    return {"query_id": query_id}

@app.get("/search/result/{query_id}")
async def get_results(query_id: str):
    if query_id not in SEARCH_RESULTS:
        raise HTTPException(status_code=404, detail="Query ID not found")
    return {"results": SEARCH_RESULTS[query_id]}

@app.get("/products")
def get_products(
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
):
    results = product_metadata

    if category:
        results = [p for p in results if p.get("category", "").lower() == category.lower()]

    if brand:
        results = [p for p in results if p.get("brand", "").lower() == brand.lower()]

    if min_price is not None:
        results = [p for p in results if float(p.get("price", 0)) >= min_price]

    if max_price is not None:
        results = [p for p in results if float(p.get("price", 0)) <= max_price]

    return {"products": results}




