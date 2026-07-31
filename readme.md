# ♻️ Waste Detection using YOLO26

An end-to-end **Waste Detection System** built using **YOLO26**, **Flask**, and **Python**. The system detects waste objects from images and classifies them into **five recyclable waste categories** (Plastic, Paper, Glass, Metal, and Trash). It includes automatic dataset preprocessing, model training, evaluation, and a user-friendly web interface for inference.

---

# 📌 Features

- ✅ YOLO26 Object Detection
- ✅ Automatic dataset preprocessing
- ✅ Label remapping (42 → 5 classes)
- ✅ Dataset split (70% Train / 15% Validation / 15% Test)
- ✅ Model training and evaluation
- ✅ Flask web application
- ✅ Prediction history logging
- ✅ Upload image and detect waste
- ✅ Cross-platform (Windows, macOS, Linux)
- ✅ Modular and scalable project structure

---

# 📂 Project Structure

```text
WasteDetection/
│
├── app.py                     # Flask application
├── train.py                   # Train the YOLO26 model
├── predict.py                 # Command-line prediction
├── evaluate.py                # Evaluate trained model
├── download_model.py          # Download YOLO26 pretrained weights
├── requirements.txt
├── requirements-lock.txt
├── README.md
├── .env.example
├── .gitignore
│
├── models/
│   └── yolo26n.pt             # Pretrained model (download separately)
│
├── dataset/
│   ├── raw/                   # Original dataset (not included)
│   └── processed/             # Generated dataset (not included)
│
├── runs/
│   └── waste_detection/        # Generated after training
│
├── uploads/                   # Uploaded images
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
│   ├── history.html
│   ├── index.html
│   └── result.html
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── history.py
    ├── logger.py
    ├── mappings.py
    ├── predictor.py
    ├── trainer.py
    ├── utils.py
    ├── validator.py
    └── Windows_run_guide.txt
```

---

# 🗂 Dataset

This project uses the **ProjectVerba YOLO Waste Detection Dataset**, which originally contains **42 waste classes**.

The dataset is automatically remapped into **5 final categories** during preprocessing.

| Final Class | Original Classes |
|-------------|------------------|
| **Plastic** | Plastic bottle, Plastic bag, Plastic cup, Plastic can, Plastic canister, Plastic caps, Plastic shaker, Plastic toys, Plastic shavings, Stretch film, Zip plastic bag, Unknown plastic, Combined plastic, Milk bottle, Tetra pack |
| **Paper** | Paper, Paper bag, Paper cups, Paper shavings, Cardboard, Postal packaging, Papier mache, Cellulose |
| **Glass** | Glass bottle |
| **Metal** | Aluminum can, Aluminum caps, Scrap metal, Tin, Foil, Iron utensils, Metal shavings |
| **Trash** | Organic, Textile, Electronics, Ceramic, Wood, Liquid, Disposable tableware, Container for household chemicals, Aerosols, Furniture, Printing industry |

Dataset split:

- **Training:** 70%
- **Validation:** 15%
- **Testing:** 15%

---

# 🛠 Tech Stack

- Python 3.12+
- Ultralytics YOLO26
- PyTorch
- Flask
- OpenCV
- NumPy
- Pillow
- PyYAML
- scikit-learn
- python-dotenv
- tqdm

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/WasteDetection.git

cd WasteDetection
```

---

## 2. Create Virtual Environment

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

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Download the YOLO26 Pretrained Model

The repository does **not** include pretrained or trained model weights to keep its size small.

Download the pretrained model automatically:

```bash
python download_model.py
```

or manually download **yolo26n.pt** and place it inside:

```text
models/
└── yolo26n.pt
```

---

# 📂 Dataset Preparation

Copy your original YOLO dataset into:

```text
dataset/raw/
```

Expected structure:

```text
dataset/raw/
│
├── train/
├── valid/
├── test/
└── data.yaml
```

Prepare the dataset:

```bash
python test_dataset.py
```

The preprocessing pipeline automatically:

- Loads the dataset
- Remaps 42 classes into 5 classes
- Removes invalid annotations
- Creates train/validation/test datasets
- Generates a new `data.yaml`

---

# ⚙️ Configuration

Project settings are stored inside the `.env` file.

Example:

```env
MODEL_NAME=best.pt

IMAGE_SIZE=512

EPOCHS=100

BATCH_SIZE=16

DEVICE=auto

CONFIDENCE=0.25

HOST=127.0.0.1

PORT=8000

DEBUG=True
```

---

# 🏋️ Training

Start training using the YOLO26 pretrained model:

```bash
python train.py
```

The model automatically starts from:

```text
models/yolo26n.pt
```

Training outputs are saved to:

```text
runs/
└── waste_detection/
    └── weights/
        ├── best.pt
        └── last.pt
```

The Flask application automatically loads:

```text
runs/waste_detection/weights/best.pt
```

for inference after training.

---

# 📈 Evaluate the Model

```bash
python evaluate.py
```

Evaluation includes:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix
- Validation Loss

---

# 🔍 Command-Line Prediction

```bash
python predict.py
```

---

# 🌐 Run the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:8000
```

Upload an image and the application will:

- Detect waste objects
- Draw bounding boxes
- Display detected classes
- Show confidence scores
- Save prediction history

---

# 📜 Prediction History

Every prediction is automatically saved to:

```text
logs/history.csv
```

Example:

| Date | Image | Class | Confidence |
|------|-------|-------|-----------|
| 2026-07-27 10:20 | bottle.jpg | Plastic | 98.45% |

The history page can be viewed at:

```text
http://127.0.0.1:8000/history
```

---

# 📦 Repository Notes

To keep the repository lightweight, the following files are **not included**:

- Original dataset
- Processed dataset
- Training outputs (`runs/`)
- Prediction images
- Uploaded images
- Log files
- Trained model (`best.pt`)

After cloning the repository:

1. Install dependencies.
2. Download `yolo26n.pt`.
3. Place your dataset in `dataset/raw/`.
4. Prepare the dataset.
5. Train the model.
6. Run the Flask application.

---

# 📊 Training Output

After training, YOLO26 generates:

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
    ├── P_curve.png
    └── labels.jpg
```

---

# 💡 Future Improvements

- Live webcam detection
- Video inference
- Drag-and-drop image upload
- Real-time object tracking
- User authentication
- Database integration
- Docker support
- Cloud deployment
- Mobile application

---

# 📄 License

This project is intended for educational and research purposes.

---

# 🙏 Acknowledgements

- Ultralytics YOLO26
- Flask
- PyTorch
- ProjectVerba Waste Detection Dataset
- Open Source Python Community
