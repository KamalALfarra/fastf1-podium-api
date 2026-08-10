import json, logging, os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("F1-API")
MODELS = Path(__file__).resolve().parent.parent / "models"
model, meta = None, {}


def live_version() -> str:
    """Env var wins over; otherwise CURRENT file from the models"""
    return os.getenv("MODEL_VERSION") or (MODELS / "CURRENT").read_text().strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup — not on every request."""
    global model, meta
    v = live_version()
    model = joblib.load(MODELS / v / "model.joblib")
    meta = json.loads((MODELS / v / "meta.json").read_text())
    log.info("loaded model v%s", meta["model_version"])
    yield


app = FastAPI(title="F1 API", version="1.0.0", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def read_root():
    ui_path = Path(__file__).resolve().parent.parent / "ui.html"
    if ui_path.exists():
        return ui_path.read_text(encoding="utf-8")
    return "UI not found"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "model_version": meta.get("model_version", "unknown")}


@app.get("/metadata")
def metadata():
    return meta


# ---------------------------------------------------------
# 1. Pydantic Models
# ---------------------------------------------------------
class F1Request(BaseModel):
    # You can define your exact F1 features here.
    # Using 'extra="allow"' lets the model accept all feature columns
    # dynamically without explicitly typing out all of them.
    model_config = {"extra": "allow"}


class F1Response(BaseModel):
    prediction: bool
    podium_probability: float
    model_version: str


# ---------------------------------------------------------
# 2. API Endpoint
# ---------------------------------------------------------
@app.post("/predict", response_model=F1Response)
def predict(payload: F1Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert incoming JSON payload to a DataFrame.
        # It strictly enforces the column order defined in meta["features"] during training.
        row = pd.DataFrame([payload.model_dump()], columns=meta["features"])

        # predict_proba returns an array of shape (n_samples, n_classes).
        # We take the first row [0], and then index [1] for the True/Podium class probability.
        proba = model.predict_proba(row)[0]
        podium_prob = float(proba[1])

        # Binary prediction (True if probability is greater than 50%)
        is_podium = bool(podium_prob > 0.5)

        return F1Response(
            prediction=is_podium,
            podium_probability=round(podium_prob, 4),
            model_version=meta["model_version"]
        )

    except Exception as exc:
        log.exception("prediction failed")  # full detail for YOU in the logs
        raise HTTPException(status_code=500, detail="Prediction failed") from exc  # short for THEM


@app.get("/versions")
def versions():
    """List all available model versions."""
    found = sorted(p.name for p in MODELS.iterdir() if p.is_dir() and p.name.startswith("v"))
    return {"available": found, "live": "v" + meta.get("model_version", "?")}
