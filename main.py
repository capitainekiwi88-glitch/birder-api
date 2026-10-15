import io
import os
from contextlib import asynccontextmanager
import torch

import birder
import numpy as np
from birder.inference.classification import infer_image
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
torch.set_num_threads(1)  # Sinon ça crash car trop gourmand
MODEL_NAME = os.getenv("BIRDER_MODEL", "regnet_y_1_6g_eu-common")
MAX_SIZE = 10 * 1024 * 1024  # 10 Mo

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    net, model_info, transform = birder.load_pretrained_model_and_transform(
        MODEL_NAME, inference=True
    )
    state["net"] = net
    state["transform"] = transform
    state["idx_to_class"] = {v: k for k, v in model_info.class_to_idx.items()}
    yield
    state.clear()


app = FastAPI(title="Birder API", lifespan=lifespan)


@app.get("/")
def home():
    return {"status": "Le serveur Birder est en ligne !", "model": MODEL_NAME}


@app.post("/predict")
def predict(file: UploadFile = File(...), top_k: int = 3):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Le fichier doit être une image.")

    data = file.file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "Image trop lourde (10 Mo max).")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((1024, 1024))
    except UnidentifiedImageError:
        raise HTTPException(400, "Image illisible.")

    out, _ = infer_image(state["net"], image, state["transform"])
    probs = np.asarray(out).squeeze()
    # Sécurité au cas où
    if not np.isclose(probs.sum(), 1.0, atol=1e-2):
        e = np.exp(probs - probs.max())
        probs = e / e.sum()

    top = np.argsort(probs)[::-1][: max(1, min(top_k, 10))]
    results = [
        {"species": state["idx_to_class"][int(i)], "confidence": round(float(probs[i]), 4)}
        for i in top
    ]
    return {"species": results[0]["species"], "predictions": results}
