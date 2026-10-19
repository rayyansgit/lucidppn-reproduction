"""
REST API for the LucidPPN model.

Run from the repository root:
    uvicorn deploy.api:app --port 8000        (interactive docs at http://localhost:8000/docs)
Request:
    curl -F "file=@bird.jpg" "http://localhost:8000/predict?hue_shift=0"
"""
import io
import os
import sys

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lucid_inference import CLASSES, DEFAULT_CHECKPOINT, load_model, predict, pretty  # noqa: E402

app = FastAPI(title='LucidPPN bird classifier', version='1.0')
net = load_model(DEFAULT_CHECKPOINT, 'cuda' if torch.cuda.is_available() else 'cpu')


@app.get('/classes')
def classes():
    return [pretty(c) for c in CLASSES]


@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...), top_k: int = 5, hue_shift: float = 0.0):
    if not -0.5 <= hue_shift <= 0.5:
        raise HTTPException(400, 'hue_shift must be in [-0.5, 0.5]')
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(400, 'file is not an image')
    result, _, _ = predict(net, image, top_k=max(1, min(top_k, len(CLASSES))), hue_shift=hue_shift)
    return result
