from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_train_module_cli_smoke(tmp_path: Path, tiny_car_csv: Path) -> None:
    out = tmp_path / "cli-out"
    cmd = [
        sys.executable,
        "-m",
        "car_pricing.train",
        "--data",
        str(tiny_car_csv),
        "--out",
        str(out),
        "--n-estimators",
        "30",
        "--max-depth",
        "3",
        "--test-size",
        "0.3",
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["fit_scope"] == "full_dataset"
    assert (out / "pipeline.joblib").is_file()
