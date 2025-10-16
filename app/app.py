import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from joblib import load
import numpy as np
from app.schema import PredictRequest, PredictResponse, HealthResponse, FEATURES

VERSION = os.getenv("APP_VERSION", "v0.1")
MODEL_PATH = Path("artifacts") / f"model_{VERSION}.joblib"
METRICS_PATH = Path("artifacts") / f"metrics_{VERSION}.json"

app = FastAPI(title="Diabetes Progression Scorer", version=VERSION)  # <-- this must be named 'app'

if not Path(MODEL_PATH).exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
pipe = load(MODEL_PATH)

def to_row(req: PredictRequest):
    return np.array([[getattr(req, f) for f in FEATURES]])

@app.exception_handler(Exception)
async def all_errors(request: Request, exc: Exception):
    return JSONResponse(status_code=400, content={"error": str(exc), "hint": "Check input schema."})

@app.get("/health", response_model=HealthResponse)
def health():
    version = VERSION
    try:
        if Path(METRICS_PATH).exists():
            with open(METRICS_PATH) as f:
                m = json.load(f)
                version = m.get("version", version)
    except Exception:
        pass
    return {"status": "ok", "model_version": version}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        pred = float(pipe.predict(to_row(payload))[0])
        return {"prediction": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
