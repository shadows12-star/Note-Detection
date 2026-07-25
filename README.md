# 💵 Bangladeshi Banknote Detector — YOLOv11 + FastAPI + Docker

Object detection system that identifies Bangladeshi Taka banknote denominations
(2, 5, 10, 20, 50, 100, 200, 500, 1000) in an image, served as a REST API and
packaged as a Docker container.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

---

## 📁 Folder structure

```
banknote-detector/
├── app/
│   ├── main.py            # FastAPI app, /predict and /health endpoints
│   ├── inference.py       # BanknoteDetector class - loads model, runs detection
│   └── schemas.py         # Pydantic request/response models
├── models/
│   └── weights/
│       └── best.pt        # <-- put your trained YOLOv11 weights here (not committed)
├── notebooks/
│   ├── train_phase2.ipynb        # retrain on the new Kaggle dataset
│   └── run_inference_demo.py     # Section 1 deliverable: single-image inference demo
├── scripts/
│   └── test_api.py        # Section 3 deliverable: batch API test against sample images
├── tests/
│   └── sample_images/     # test images used for API validation
├── docs/
│   └── screenshots/       # <-- put all submission screenshots here (see slots below)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 1. Model Integration & Inference Pipeline

`app/inference.py` defines `BanknoteDetector`, a thin wrapper around
Ultralytics YOLOv11 that:

- loads weights from `models/weights/best.pt`
- accepts a filepath, raw bytes, or a PIL image
- returns a JSON-serialisable dict: image size, detected classes,
  confidence scores, and bounding box coordinates (`x1,y1,x2,y2` in pixels)

Run the standalone demo:

```bash
pip install -r requirements.txt
python notebooks/run_inference_demo.py --image tests/sample_images/note1.jpg
```

This prints the detections to the console and saves an annotated image to
`outputs/annotated.jpg`.







## 2. REST API

Run locally without Docker:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs: http://localhost:8000/docs

### Endpoint

| | |
|---|---|
| **URL** | `/predict` |
| **Method** | `POST` |
| **Body** | `multipart/form-data`, field name `file`, a JPEG or PNG image |
| **Success** | `200 OK`, JSON body (see below) |
| **Errors** | `400` empty/oversized/corrupt file · `415` wrong content type · `503` model not loaded |

**Example response:**

```json
	

{
  "filename": "20_taka_cc27a36d55d096dc.png",
  "image_width": 256,
  "image_height": 117,
  "detection_count": 1,
  "detections": [
    {
      "class_id": 5,
      "denomination": "100_taka",
      "confidence": 0.912213,
      "bounding_box": {
        "x_min": 9.89,
        "y_min": 3.06,
        "x_max": 252.83,
        "y_max": 113.85
      }
    }
  ]
}
```

### curl example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@tests/sample_images/note1.jpg;type=image/jpeg"
```

### Postman

1. New request → `POST http://localhost:8000/predict`
2. Body → `form-data` → key `file`, type **File** → select an image
3. Send → response body is the JSON shown above

📸 **Screenshot — successful `/predict` call via Postman**

<p align="center">
<img width="2271" height="1299" alt="image" src="https://github.com/user-attachments/assets/51d84e96-e9ed-4430-938d-f4d1f1907cdb" />

</p>


📸 **Screenshot — Graceful Error Handling**

<p align="center">
<img width="2138" height="1240" alt="image" src="https://github.com/user-attachments/assets/bbc7df0e-6860-4f52-a890-7d6955da01f8" />

</p>



---

## 3. API Testing & Validation
## API Testing & Validation

The REST API was tested using **Postman** by sending multiple JPEG/PNG images to the `/predict` endpoint. Each request was successfully processed, and the API returned a **200 OK** response containing the detected denomination(s), confidence score(s), and bounding box coordinates in JSON format.

### Prediction Accuracy

The deployed YOLOv11 model demonstrated satisfactory performance on the tested Bangladeshi Taka note images.

During testing, the API successfully detected **multiple banknotes within a single image**. For example:

- **1000 Taka** note detected with a confidence score of **81.1%**
- **10 Taka** note detected with a confidence score of **65.3%**

The higher confidence score for the 1000 Taka note indicates that the model was highly confident in its prediction. The comparatively lower confidence for the 10 Taka note suggests greater uncertainty, which may be caused by factors such as:

- Small object size in the image
- Lighting variations
- Image blur
- Partial occlusion
- Similar visual characteristics between different denominations

Overall, the API produced accurate detections and returned correctly formatted JSON responses, demonstrating successful integration of the YOLOv11 model with the FastAPI application.

The model performs well on clear, high-quality images, while prediction confidence may decrease when notes are partially visible, rotated, blurred, or captured under poor lighting conditions.



### Example JSON Response

```json
{
  "filename": "images.jpg",
  "image_width": 815,
  "image_height": 376,
  "detection_count": 2,
  "detections": [
    {
      "class_id": 7,
      "denomination": "1000_taka",
      "confidence": 0.811205,
      "bounding_box": {
        "x_min": 33.07,
        "y_min": 9.70,
        "x_max": 600.37,
        "y_max": 366.48
      }
    },
    {
      "class_id": 2,
      "denomination": "10_taka",
      "confidence": 0.653476,
      "bounding_box": {
        "x_min": 642.37,
        "y_min": 9.20,
        "x_max": 815.00,
        "y_max": 365.72
      }
    }
  ]
}
```

### Conclusion

The experimental results demonstrate that the deployed YOLOv11 model can reliably detect Bangladeshi Taka notes through a FastAPI REST API. The system supports both single and multiple banknote detection, provides confidence scores and bounding box coordinates, and produces consistent JSON responses suitable for real-world integration and deployment.


📸 **Screenshots — request/response for each of the 5 test images:**

<p align="center">
<img width="2142" height="1306" alt="image" src="https://github.com/user-attachments/assets/53492704-3690-4b66-8966-0f8e61521d46" />

</p>
<p align="center">
<img width="966" height="1079" alt="image" src="https://github.com/user-attachments/assets/6e15f11f-b656-471e-b747-0bae8de78e82" />

</p>
<p align="center">
  <img src="docs/screenshots/06-test-3.png" alt="Test image 3 result" width="480">
</p>
<p align="center">
<img width="1258" height="1072" alt="image" src="https://github.com/user-attachments/assets/af754381-0f81-4f91-a12d-0d67fdfbb312" />

</p>
<p align="center">
<img width="968" height="1255" alt="image" src="https://github.com/user-attachments/assets/12b4b07d-a349-45b6-a721-5702336a93a9" />

</p>


## 4. Dockerization

**Build:**

```bash
docker build -t banknote-detector:latest .
```

**Run:**

```bash
docker run -d -p 8000:8000 --name banknote-api banknote-detector:latest
```


**Verify it's up:**

```bash
curl http://localhost:8000/health
```

📸 **Screenshot — container running (`docker ps` / `docker logs`) + a successful API call against it:**

<p align="center">
	<img width="1592" height="912" alt="image" src="https://github.com/user-attachments/assets/2bc5964e-3bae-4adc-a244-1c2d1220348b" />
	<img width="1127" height="972" alt="image" src="https://github.com/user-attachments/assets/2cdd0640-e985-4a23-8765-e1868770503c" />


  <img width="1587" height="922" alt="image" src="https://github.com/user-attachments/assets/3238fa79-7abf-4f2c-8b7a-15f150a187eb" />

</p>

<!-- Replace docs/screenshots/10-docker-running.png with your own screenshot -->

To view logs: `docker logs -f banknote-api`
To stop: `docker stop banknote-api`



