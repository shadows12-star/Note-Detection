from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module

client = TestClient(main_module.app)


def make_png_bytes() -> bytes:
    image = Image.new("RGB", (100, 50), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_available" in response.json()


def test_predict_success(monkeypatch) -> None:
    fake_detection = {
        "class_id": 5,
        "denomination": "100_taka",
        "confidence": 0.987,
        "bounding_box": {"x_min": 1.0, "y_min": 2.0, "x_max": 99.0, "y_max": 48.0},
    }
    monkeypatch.setattr(main_module.detector, "predict", lambda image: [fake_detection])

    response = client.post(
        "/predict",
        files={"file": ("note.png", make_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["detection_count"] == 1
    assert body["detections"][0]["denomination"] == "100_taka"


def test_missing_file() -> None:
    response = client.post("/predict")
    assert response.status_code == 400


def test_unsupported_media_type() -> None:
    response = client.post(
        "/predict",
        files={"file": ("note.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_corrupt_image() -> None:
    response = client.post(
        "/predict",
        files={"file": ("note.png", b"not really a png", "image/png")},
    )
    assert response.status_code == 400
