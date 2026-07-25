# Bangladeshi Banknote Detection API with YOLOv11

This project retrains YOLOv11 for Bangladeshi banknote denomination recognition, provides a single-image inference pipeline, serves the model through FastAPI, tests the API with five images, and packages the application with Docker.

## Important dataset and notebook findings

The supplied notebook is a people-flow tracking project. It uses a pretrained YOLO model, ByteTrack, line crossing, video processing, and a heatmap. It does not train or serve a Bangladeshi banknote model, so a new training notebook is included.

The selected Kaggle dataset contains roughly 70,000 images in eight denomination folders: 2, 5, 10, 20, 50, 100, 500, and 1000 taka. It is an image-classification dataset and does not contain YOLO bounding-box labels. It also does not include 200 taka.

To remain aligned with the requested dataset and the assignment's single-image requirement, the training notebook converts every image into one YOLO detection sample with a full-image bounding box. This allows YOLOv11 to return a denomination, confidence, and box for a single banknote image. It is not a substitute for genuine object-detection annotation and should not be used to claim reliable multi-note detection.

## Project structure

```text
bangladeshi_banknote_yolov11_project/
├── app/
│   ├── config.py
│   ├── detector.py
│   ├── main.py
│   └── schemas.py
├── docs/
│   └── REPORT_TEMPLATE.md
├── logs/
├── models/
│   └── README.md
├── reference/
│   └── original_people_flow_notebook.ipynb
├── sample_images/
├── scripts/
│   ├── curl_test.sh
│   └── test_five_images.py
├── tests/
│   └── test_api.py
├── training/
│   └── Banknote_YOLOv11_Training.ipynb
├── .dockerignore
├── .gitignore
├── Dockerfile
├── inference.py
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## 1. Train the model in Google Colab

1. Open `training/Banknote_YOLOv11_Training.ipynb` in Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run every cell in order.
4. Authenticate with Kaggle if Colab requests it.
5. The notebook downloads `rahnumatasnim1604103/bangladeshi-banknote-dataset`, detects the eight class folders, creates train/validation/test splits, generates YOLO labels, trains YOLOv11n, evaluates the held-out test split, and demonstrates single-image inference.
6. Download the resulting `best.pt` file.
7. Put it in this project as `models/best.pt`.

The notebook defaults to all discovered images. For a quick pipeline check, set `MAX_IMAGES_PER_CLASS` to a smaller value such as `500`; return it to `None` for the final training run.

## 2. Install and run locally

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open the interactive documentation at `http://127.0.0.1:8000/docs`.

## 3. Single-image inference

```bash
python inference.py \
  --image sample_images/test_note.jpg \
  --model models/best.pt \
  --output logs/annotated_prediction.jpg
```

The command prints detected class names, confidence scores, and `xyxy` pixel coordinates. It also saves an annotated image.

## 4. REST API

### Endpoint

```text
POST /predict
```

### Input

Send `multipart/form-data` with one field:

```text
file=<JPEG or PNG image>
```

### curl example

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample_images/test_note.jpg"
```

### Example response format

The values below illustrate the JSON structure; they are not claimed as results from an untrained model.

```json
{
  "filename": "test_note.jpg",
  "image_width": 1280,
  "image_height": 720,
  "detection_count": 1,
  "detections": [
    {
      "class_id": 5,
      "denomination": "100_taka",
      "confidence": 0.973421,
      "bounding_box": {
        "x_min": 14.2,
        "y_min": 10.8,
        "x_max": 1267.5,
        "y_max": 707.1
      }
    }
  ]
}
```

### Error handling

| Condition | Status code |
|---|---:|
| Missing file | 400 |
| Empty/corrupt image | 400 |
| File larger than configured limit | 413 |
| Unsupported media type | 415 |
| Missing model weights | 503 |
| Unexpected inference failure | 500 |

## 5. Test with at least five images

Place five held-out JPEG/PNG images in `sample_images/`, start the API, then run:

```bash
python scripts/test_five_images.py --images sample_images
```

The script sends five requests and saves the complete result log to:

```text
logs/api_test_results.json
```

For Postman:

1. Create a `POST` request to `http://127.0.0.1:8000/predict`.
2. Select **Body → form-data**.
3. Add key `file`, change its type to **File**, and select a JPEG/PNG image.
4. Send the request and capture the response screenshot.
5. Repeat for at least five held-out test images.

## 6. Run automated API validation

These tests use a mocked detector output and verify request validation and response formatting without needing trained weights:

```bash
python -m pytest -q
```

They do not replace the required five-image model accuracy test.

## 7. Docker build and run

Ensure `models/best.pt` exists before building the final submission image.

```bash
docker build -t banknote-yolo-api:1.0 .
```

Run the container and publish container port 8000 to host port 8000:

```bash
docker run --rm -p 8000:8000 banknote-yolo-api:1.0
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Test prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_images/test_note.jpg"
```

Docker's `EXPOSE 8000` documents the container port; the `-p 8000:8000` option publishes it to the host.

## 8. Required screenshots and evidence

Before submission, capture:

- Colab training completion and metrics.
- The single-image printed/visualized inference result.
- Five Postman or curl request/response results.
- `docker build` completion.
- `docker run` output showing Uvicorn on port 8000.
- A successful `/health` response and one `/predict` response from the container.

Use `docs/REPORT_TEMPLATE.md` to assemble the written report and prediction-accuracy discussion.

## Recommended improvement for a genuine detector

For reliable localization or multiple banknotes in one scene, manually annotate boxes in CVAT, Roboflow, or Label Studio, or use a dataset that already has YOLO detection labels. Then update the notebook to point directly to that dataset's `data.yaml`; the API and Docker code can remain unchanged.
