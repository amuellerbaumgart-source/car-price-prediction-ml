from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from car_pricing.io_utils import load_metadata, load_pipeline
from car_pricing.predict import predict_dataframe
from car_pricing.train import train_and_publish


def test_train_predict_metadata_roundtrip(tmp_path: Path, tiny_car_csv: Path) -> None:
    out = tmp_path / "artifacts"
    train_and_publish(
        tiny_car_csv,
        out,
        backend="sklearn_hist",
        test_size=0.25,
        random_state=42,
        n_estimators=50,
        learning_rate=0.1,
        max_depth=4,
        eval_only=False,
    )

    joblib_path = out / "pipeline.joblib"
    meta_path = out / "metadata.json"
    assert joblib_path.is_file()
    assert meta_path.is_file()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["target"] == "Price"
    assert meta["backend"] == "sklearn_hist"
    assert "rmse_holdout" in meta

    pipe = load_pipeline(joblib_path)
    extra = load_metadata(out)
    assert extra.get("backend") == "sklearn_hist"

    df = pd.read_csv(tiny_car_csv)
    features = df.drop(columns=["Price"]).iloc[:3]
    preds = predict_dataframe(pipe, features)
    assert len(preds) == 3
    assert all(p > 0 for p in preds)
