"""Load/save model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_pipeline(pipe: Pipeline, out_dir: Path, metadata: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "pipeline.joblib"
    meta_path = out_dir / "metadata.json"
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path


def load_pipeline(model_path: Path) -> Pipeline:
    pipe = joblib.load(model_path)
    if not isinstance(pipe, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline at {model_path}, got {type(pipe)}")
    return pipe


def load_metadata(model_dir: Path) -> dict[str, Any]:
    meta_path = model_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
