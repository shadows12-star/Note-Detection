from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.detector import BanknoteDetector, ModelNotReadyError
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="Bangladeshi Banknote Detection API",
    version="1.0.0",
    description="YOLOv11 API for detecting Bangladeshi banknote denominations.",
)

detector = BanknoteDetector(
    model_path=settings.model_path,
    confidence_threshold=settings.confidence_threshold,
    iou_threshold=settings.iou_threshold,
    image_size=settings.image_size,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Bangladeshi Banknote Detection API",
        "predict_endpoint": "/predict",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    available = detector.model_available
    return HealthResponse(
        status="ready" if available else "model_missing",
        model_path=str(detector.model_path),
        model_available=available,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile | None = File(default=None, description="JPEG or PNG banknote image"),
) -> PredictionResponse:
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An image file is required in the 'file' form field.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG and PNG images are supported.",
        )

    payload = await file.read(settings.max_upload_bytes + 1)
    await file.close()

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.max_upload_bytes} byte upload limit.",
        )

    try:
        image = Image.open(BytesIO(payload))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid JPEG or PNG image.",
        ) from exc

    try:
        detections = detector.predict(image)
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed.",
        ) from exc

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        image_width=image.width,
        image_height=image.height,
        detection_count=len(detections),
        detections=detections,
    )
