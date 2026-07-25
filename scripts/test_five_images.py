from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test /predict with at least five images.")
    parser.add_argument("--images", required=True, type=Path, help="Folder containing test images")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--output", default=Path("logs/api_test_results.json"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) < 5:
        raise ValueError(f"At least five test images are required; found {len(image_paths)}.")

    results: list[dict[str, object]] = []
    for image_path in image_paths[:5]:
        mime_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        with image_path.open("rb") as image_file:
            response = requests.post(
                args.url,
                files={"file": (image_path.name, image_file, mime_type)},
                timeout=120,
            )

        try:
            response_body: object = response.json()
        except ValueError:
            response_body = response.text

        item = {
            "image": str(image_path),
            "status_code": response.status_code,
            "response": response_body,
        }
        results.append(item)
        print(json.dumps(item, indent=2))
        response.raise_for_status()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved five-image API test log to: {args.output}")


if __name__ == "__main__":
    main()
