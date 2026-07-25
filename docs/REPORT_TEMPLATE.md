# Project Report: Bangladeshi Banknote Detection with YOLOv11

## 1. Model Integration and Inference Pipeline (15 Marks)

### Model
- Architecture: YOLOv11n object detection
- Weight file: `models/best.pt`
- Classes: `2_taka`, `5_taka`, `10_taka`, `20_taka`, `50_taka`, `100_taka`, `500_taka`, `1000_taka`

### Single-image inference
Command used:

```bash
python inference.py --image sample_images/<IMAGE_NAME>.jpg --model models/best.pt
```

Insert the printed JSON and annotated output image here.

## 2. REST API Development (25 Marks)

- Framework: FastAPI
- Endpoint: `POST /predict`
- Input: multipart form field named `file`
- Supported formats: JPEG and PNG

Insert a Postman or curl screenshot here.

## 3. API Testing and Validation (10 Marks)

| Test image | Ground truth | Predicted denomination | Confidence | Correct? |
|---|---:|---:|---:|---|
| Image 1 |  |  |  |  |
| Image 2 |  |  |  |  |
| Image 3 |  |  |  |  |
| Image 4 |  |  |  |  |
| Image 5 |  |  |  |  |

### Accuracy discussion
Report the number of correct predictions out of five. Discuss failures caused by blur, lighting, viewing angle, occlusion, note wear, or the classification-to-detection conversion used for this dataset.

## 4. Dockerization (30 Marks)

Build command:

```bash
docker build -t banknote-yolo-api:1.0 .
```

Run command:

```bash
docker run --rm -p 8000:8000 banknote-yolo-api:1.0
```

Insert build and running-container screenshots here.

## 5. Deployment and Documentation (20 Marks)

Describe the folder structure, setup procedure, API usage, and limitations. Include the GitHub repository URL or zip filename.

## Dataset limitation statement

The selected Kaggle dataset is organized for image classification and does not provide true object-detection bounding boxes. For this assignment, one full-image YOLO box is generated for each single-banknote image. Therefore, the resulting model is intended for single-banknote images and its box localization should not be presented as evidence of robust multi-banknote detection. A production detector requires manually annotated boxes or a genuine detection dataset.
