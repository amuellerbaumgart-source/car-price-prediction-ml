from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_car_csv() -> Path:
    path = ROOT / "tests" / "fixtures" / "tiny_car_prices.csv"
    if not path.is_file():
        pytest.skip(f"Missing fixture: {path}")
    return path
