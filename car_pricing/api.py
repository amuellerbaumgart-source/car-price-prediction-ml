"""FastAPI service: POST /predict for single or batch estimates."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from car_pricing.io_utils import load_metadata, load_pipeline
from car_pricing.predict import predict_dataframe
from car_pricing.schemas import CarFeatures, PredictionBatchOut, PredictionOut


def resolve_model_path() -> Path:
    explicit = os.environ.get("CAR_PRICE_MODEL_PATH")
    if explicit:
        return Path(explicit)
    artifact_dir = Path(os.environ.get("CAR_PRICE_ARTIFACT_DIR", "artifacts"))
    return artifact_dir / "pipeline.joblib"


app = FastAPI(title="Car price predictor", version="0.1.0")
_pipe: Pipeline | None = None
_cached_model_file: Path | None = None


def clear_inference_cache() -> None:
    """Forget any loaded estimator (tests or env changes between requests)."""

    global _pipe, _cached_model_file
    _pipe = None
    _cached_model_file = None


def get_pipe() -> Pipeline:
    global _pipe, _cached_model_file
    path = resolve_model_path()
    need_reload = _pipe is None or _cached_model_file != path
    if need_reload:
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {path}. Train with: python -m car_pricing.train",
            )
        _pipe = load_pipeline(path)
        _cached_model_file = path
    return _pipe


@app.get("/health")
def health() -> dict:
    path = resolve_model_path()
    meta = load_metadata(path.parent)
    return {"status": "ok", "model_path": str(path), "metadata_keys": list(meta.keys())}


@app.post("/predict", response_model=PredictionOut)
def predict_one(car: CarFeatures) -> PredictionOut:
    pipe = get_pipe()
    df = pd.DataFrame([car.model_dump()])
    preds = predict_dataframe(pipe, df)
    return PredictionOut(price_predicted=round(preds[0], 2))


@app.post("/predict/batch", response_model=PredictionBatchOut)
def predict_batch(cars: list[CarFeatures]) -> PredictionBatchOut:
    if not cars:
        raise HTTPException(status_code=400, detail="Empty batch")
    pipe = get_pipe()
    df = pd.DataFrame([c.model_dump() for c in cars])
    preds = predict_dataframe(pipe, df)
    return PredictionBatchOut(predictions=[round(float(x), 2) for x in preds])
