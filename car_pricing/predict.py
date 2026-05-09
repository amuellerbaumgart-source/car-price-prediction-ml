"""Batch prediction from CSV or JSON lines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from car_pricing.io_utils import load_pipeline
from car_pricing.pipeline_builder import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def predict_dataframe(pipe: Pipeline, frame: pd.DataFrame) -> list[float]:
    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")
    preds = pipe.predict(frame[cols])
    return [float(x) for x in preds]


def main() -> None:
    p = argparse.ArgumentParser(description="Run saved pipeline on new rows (CSV or stdin JSON).")
    p.add_argument("model", type=Path, help="Path to pipeline.joblib")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=Path, help="CSV with same feature columns as training (no Price required)")
    g.add_argument(
        "--json",
        type=str,
        help='Single JSON object or array of objects, e.g. \'{"Brand":"Toyota",...}\'',
    )
    args = p.parse_args()

    pipe = load_pipeline(args.model)

    if args.csv is not None:
        df = pd.read_csv(args.csv)
        out = predict_dataframe(pipe, df)
        for v in out:
            print(v)
        return

    raw = args.json
    if raw == "-":
        raw = sys.stdin.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    out = predict_dataframe(pipe, df)
    print(json.dumps({"predictions": out}))


if __name__ == "__main__":
    main()
