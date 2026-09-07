import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="EnviroMat Waste Classifier")

# Loads the pretrained model once when the server starts (not on every request)
model = YOLO("best_model.pt")

# Optional shared secret so random people on the internet can't hit your model for free.
# Leave API_SECRET unset locally; set it once you deploy.
API_SECRET = os.environ.get("API_SECRET")


@app.get("/")
def health():
    return {"status": "ok", "classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(default=None)):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    results = model.predict(image, verbose=False)
    r = results[0]

    # Classification models return probabilities per class (r.probs),
    # not bounding boxes (r.boxes) like the old detection model did.
    top1_index = r.probs.top1
    confidence = round(r.probs.top1conf.item(), 4)
    label = model.names[top1_index]

    return {"label": label, "confidence": confidence}