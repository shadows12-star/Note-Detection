from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x_min: float = Field(..., ge=0)
    y_min: float = Field(..., ge=0)
    x_max: float = Field(..., ge=0)
    y_max: float = Field(..., ge=0)


class Detection(BaseModel):
    class_id: int = Field(..., ge=0)
    denomination: str
    confidence: float = Field(..., ge=0, le=1)
    bounding_box: BoundingBox


class PredictionResponse(BaseModel):
    filename: str
    image_width: int = Field(..., gt=0)
    image_height: int = Field(..., gt=0)
    detection_count: int = Field(..., ge=0)
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_path: str
    model_available: bool
