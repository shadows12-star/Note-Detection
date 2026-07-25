from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.detector import BanknoteDetector


def draw_detections(image: Image.Image, detections: list[dict[str, object]]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for detection in detections:
        box = detection["bounding_box"]
        coordinates = (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
        label = f"{detection['denomination']} {float(detection['confidence']):.2f}"
        draw.rectangle(coordinates, outline="red", width=4)
        text_box = draw.textbbox((coordinates[0], coordinates[1]), label, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((coordinates[0], coordinates[1]), label, fill="white", font=font)

    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image YOLOv11 banknote inference.")
    parser.add_argument("--image", required=True, type=Path, help="Input JPEG/PNG image")
    parser.add_argument("--model", default=Path("models/best.pt"), type=Path)
    parser.add_argument("--confidence", default=0.25, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--output", default=Path("logs/annotated_prediction.jpg"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    detector = BanknoteDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        image_size=args.imgsz,
    )
    detections = detector.predict(image)

    output = {
        "filename": args.image.name,
        "image_width": image.width,
        "image_height": image.height,
        "detection_count": len(detections),
        "detections": detections,
    }
    print(json.dumps(output, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated = draw_detections(image, detections)
    annotated.save(args.output)
    print(f"Annotated image saved to: {args.output}")


if __name__ == "__main__":
    main()
