from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from car_pricing import api
from car_pricing.train import train_and_publish


@pytest.fixture
def client_with_model(
    tmp_path: Path, tiny_car_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    out = tmp_path / "api-artifacts"
    train_and_publish(
        tiny_car_csv,
        out,
        backend="sklearn_hist",
        test_size=0.3,
        random_state=1,
        n_estimators=40,
        learning_rate=0.1,
        max_depth=4,
        eval_only=False,
    )
    monkeypatch.setenv("CAR_PRICE_MODEL_PATH", str(out / "pipeline.joblib"))
    api.clear_inference_cache()
    yield TestClient(api.app)
    api.clear_inference_cache()


def test_health_and_predict(client_with_model: TestClient) -> None:
    h = client_with_model.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert "metadata_keys" in body

    payload = {
        "Brand": "Toyota",
        "Model": "Camry",
        "Year": 2015,
        "Engine_Size": 2.5,
        "Fuel_Type": "Petrol",
        "Transmission": "Automatic",
        "Mileage": 90000,
        "Doors": 4,
        "Owner_Count": 2,
    }
    r = client_with_model.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "price_predicted" in data
    assert data["price_predicted"] > 0

    b = client_with_model.post("/predict/batch", json=[payload, payload])
    assert b.status_code == 200
    assert len(b.json()["predictions"]) == 2
