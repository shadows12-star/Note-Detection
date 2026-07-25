from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from PIL import Image


class ModelNotReadyError(RuntimeError):
    """Raised when the trained weights are not available."""


class BanknoteDetector:
    """Lazy-loading wrapper around an Ultralytics YOLO detection model."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def model_available(self) -> bool:
        return self.model_path.is_file()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_available:
                raise ModelNotReadyError(
                    f"Model weights were not found at '{self.model_path}'. "
                    "Train the model and copy best.pt into models/best.pt."
                )

            # Imported lazily so the API can still start and report a useful
            # health response when the weights are missing.
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def predict(self, image: Image.Image) -> list[dict[str, object]]:
        """Run object detection and return JSON-serializable detections."""
        self._ensure_loaded()
        rgb_image = image.convert("RGB")
        image_array = np.asarray(rgb_image)

        results = self._model.predict(
            source=image_array,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        detections: list[dict[str, object]] = []
        for coords, confidence, class_id in zip(xyxy, confidences, class_ids):
            x_min, y_min, x_max, y_max = (float(value) for value in coords)
            detections.append(
                {
                    "class_id": int(class_id),
                    "denomination": self._class_name(result.names, int(class_id)),
                    "confidence": round(float(confidence), 6),
                    "bounding_box": {
                        "x_min": round(x_min, 2),
                        "y_min": round(y_min, 2),
                        "x_max": round(x_max, 2),
                        "y_max": round(y_max, 2),
                    },
                }
            )

        detections.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return detections
