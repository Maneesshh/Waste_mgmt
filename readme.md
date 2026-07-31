# ♻️ Waste Detection using YOLO11

An end-to-end **Waste Detection System** built with **YOLO11**, **Flask**, and **Python**. The application detects waste objects from images and classifies them into **five recyclable waste categories**.

---

## 📌 Features

* YOLO11 Object Detection
* Five waste categories
* Automatic dataset preparation
* Label remapping from 42 classes to 5 classes
* 70/15/15 train-validation-test split
* Flask web interface
* REST API
* Prediction history logging
* Cross-platform (Windows, macOS, Linux)
* Modular project structure

---

## 📂 Project Structure

```text
WasteDetection/
│
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── yolo11n.pt
│
├── runs/
│
├── uploads/
│
├── logs/
│   ├── application.log
│   └── history.csv
│
├── static/
│   └── predictions/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── 404.html
│   └── 500.html
│
└── src/
    ├── cleanup.py
    ├── config.py
    ├── dataset.py
    ├── history.py
    ├── logger.py
    ├── mappings.py
    ├── predictor.py
    ├── trainer.py
    ├── utils.py
    └── validator.py
```

---

# Dataset

The project uses the **ProjectVerba YOLO Waste Detection Dataset**, originally containing **42 waste classes**.

These classes are automatically remapped into **5 final classes**.

| Final Class | Original Classes                                                                                                                                                                                                                 |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Plastic     | Plastic bottle, Plastic bag, Plastic cup, Plastic can, Plastic canister, Plastic caps, Plastic shaker, Plastic toys, Plastic shavings, Stretch film, Zip plastic bag, Unknown plastic, Combined plastic, Milk bottle, Tetra pack |
| Paper       | Paper, Paper bag, Paper cups, Paper shavings, Cardboard, Postal packaging, Papier mache, Cellulose                                                                                                                               |
| Glass       | Glass bottle                                                                                                                                                                                                                     |
| Metal       | Aluminum can, Aluminum caps, Scrap metal, Tin, Foil, Iron utensils, Metal shavings                                                                                                                                               |
| Trash       | Organic, Textile, Electronics, Ceramic, Wood, Liquid, Disposable tableware, Container for household chemicals, Aerosols, Furniture, Printing industry                                                                            |

Dataset split:

* **Training:** 70%
* **Validation:** 15%
* **Testing:** 15%

---

# Tech Stack

* Python 3.12+
* Ultralytics YOLO11
* Flask
* OpenCV
* NumPy
* PyYAML
* Pillow
* python-dotenv
* tqdm
* scikit-learn

---

# Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/WasteDetection.git

cd WasteDetection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download YOLO11 Model

Download one of the following models from Ultralytics:

* yolo11n.pt
* yolo11s.pt
* yolo11m.pt

Place the model inside:

```text
models/
```

Example:

```text
models/
└── yolo11n.pt
```

---

# Dataset Preparation

Copy the original YOLO dataset into

```text
dataset/raw/
```

Expected structure:

```text
dataset/raw/

train/
valid/
test/

data.yaml
```

Prepare the dataset:

```bash
python test_dataset.py
```

This automatically:

* Reads all annotations
* Remaps labels
* Removes invalid annotations
* Splits dataset
* Creates processed dataset
* Generates new data.yaml

---

# Training

Edit `.env`

```env
MODEL_NAME=yolo11n.pt

IMAGE_SIZE=640

EPOCHS=100

BATCH_SIZE=8

DEVICE=cpu

CONFIDENCE=0.25
```

Start training

```bash
python train.py
```

Training results are saved in:

```text
runs/

waste_detection/
```

---

# Evaluate Model

```bash
python evaluate.py
```

Metrics include:

* Precision
* Recall
* mAP50
* mAP50-95
* Confusion Matrix

---

# Run Prediction

```bash
python predict.py
```

---

# Run Flask Application

```bash
python app.py
```

Open

```text
http://127.0.0.1:5000
```

---

# REST API

## Predict

```http
POST /api/predict
```

Body:

```
image : File
```

Example Response

```json
{
  "success": true,
  "filename": "plastic.jpg",
  "prediction_count": 2,
  "predictions": [
    {
      "class": "Plastic",
      "confidence": 98.74
    },
    {
      "class": "Metal",
      "confidence": 91.20
    }
  ]
}
```

---

# Prediction History

Every prediction is automatically stored in

```text
logs/history.csv
```

Example

| Date             | Image      | Class   | Confidence |
| ---------------- | ---------- | ------- | ---------- |
| 2026-07-27 10:20 | bottle.jpg | Plastic | 98.45      |

---

# Configuration

Project settings are stored inside

```text
.env
```

Example

```env
MODEL_NAME=yolo11n.pt
IMAGE_SIZE=640
EPOCHS=100
BATCH_SIZE=8
DEVICE=cpu
CONFIDENCE=0.25
```

---

# Output

Training generates:

```text
runs/
└── waste_detection/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── confusion_matrix.png
    ├── results.csv
    ├── results.png
    ├── F1_curve.png
    ├── PR_curve.png
    └── P_curve.png
```

---

# Future Improvements

* Live webcam detection
* Video inference
* Drag-and-drop image upload
* User authentication
* Database integration
* Docker deployment
* Cloud deployment
* Mobile application integration

---

# License

This project is intended for educational and research purposes.


# Acknowledgements

* Ultralytics YOLO
* Flask
* ProjectVerba YOLO Waste Detection Dataset
* Open Source Python Community
