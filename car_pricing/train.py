"""Train, evaluate, and persist a sklearn Pipeline for car price regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from car_pricing.io_utils import save_pipeline
from car_pricing.pipeline_builder import (
    BackendName,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_training_pipeline,
)


def train_and_publish(
    data_path: Path,
    artifact_dir: Path,
    *,
    backend: BackendName,
    test_size: float,
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    eval_only: bool,
) -> None:
    df = pd.read_csv(data_path)
    missing = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_training_pipeline(
        backend,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "rmse_holdout": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae_holdout": float(mean_absolute_error(y_test, y_pred)),
        "r2_holdout": float(r2_score(y_test, y_pred)),
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target": TARGET_COLUMN,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "backend": backend,
        "hyperparameters": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "random_state": random_state,
        },
    }

    if not eval_only:
        pipe_full = build_training_pipeline(
            backend,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        pipe_full.fit(X, y)
        artifact_pipe = pipe_full
        metrics["fit_scope"] = "full_dataset"
    else:
        artifact_pipe = pipe
        metrics["fit_scope"] = "train_split_only"

    save_pipeline(artifact_pipe, artifact_dir, metadata=metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved pipeline and metadata under: {artifact_dir.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train car price model and save sklearn Pipeline + metadata.")
    p.add_argument(
        "--backend",
        choices=("sklearn_hist", "xgboost"),
        default="sklearn_hist",
        help="Use xgboost to match notebook (requires OpenMP on macOS); sklearn_hist works without extra natives",
    )
    p.add_argument("--data", type=Path, default=Path("car_price_dataset.csv"), help="Training CSV path")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts"),
        help="Directory for pipeline.joblib and metadata.json",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Save the model trained only on the training split (for debugging; not recommended for deployment)",
    )
    args = p.parse_args()

    train_and_publish(
        args.data,
        args.out,
        backend=cast(BackendName, args.backend),
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
