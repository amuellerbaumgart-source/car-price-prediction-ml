"""Request/response models for APIs and validated CLI input."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CarFeatures(BaseModel):
    """One vehicle listing (same semantics as CSV columns without Price)."""

    Brand: str = Field(..., examples=["Toyota"])
    Model: str = Field(..., examples=["Camry"])
    Year: int = Field(..., ge=1950, le=2035)
    Engine_Size: float = Field(..., gt=0, description="Displacement in liters (as in the dataset)")
    Fuel_Type: str = Field(..., examples=["Petrol"])
    Transmission: str = Field(..., examples=["Automatic"])
    Mileage: int = Field(..., ge=0)
    Doors: int = Field(..., ge=2, le=7)
    Owner_Count: int = Field(..., ge=1, le=20)


class PredictionOut(BaseModel):
    price_predicted: float = Field(..., description="Point estimate for resale price")


class PredictionBatchOut(BaseModel):
    predictions: list[float]
