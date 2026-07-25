from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number.") from exc


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "models/best.pt"))
    confidence_threshold: float = _get_float("CONF_THRESHOLD", 0.25)
    iou_threshold: float = _get_float("IOU_THRESHOLD", 0.45)
    image_size: int = _get_int("IMAGE_SIZE", 640)
    max_upload_bytes: int = _get_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)


settings = Settings()
