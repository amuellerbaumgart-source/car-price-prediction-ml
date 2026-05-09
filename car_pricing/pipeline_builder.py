"""Training pipeline aligned with FinalML.ipynb (same preprocessing + tree regressor)."""

from __future__ import annotations

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BackendName = Literal["xgboost", "sklearn_hist"]


TARGET_COLUMN = "Price"

NUMERICAL_FEATURES = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]
CATEGORICAL_FEATURES = ["Brand", "Model", "Fuel_Type", "Transmission"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    backend: BackendName = "sklearn_hist",
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    random_state: int = 42,
) -> Pipeline:
    preprocessor = build_preprocessor()

    if backend == "xgboost":
        from xgboost import XGBRegressor

        regressor = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    elif backend == "sklearn_hist":
        regressor = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])
