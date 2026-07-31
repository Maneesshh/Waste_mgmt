from pathlib import Path
from typing import cast

from flask import (
    Flask,
    render_template,
    request,
    jsonify
)

from ultralytics import YOLO

from src.config import *
from src.validator import allowed
from src.history import save_prediction, get_history

# ------------------------------------------------------------
# Flask App
# ------------------------------------------------------------

app = Flask(__name__)

# ------------------------------------------------------------
# Upload Folder
# ------------------------------------------------------------

UPLOAD_FOLDER = Path(UPLOAD_DIR)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

MODEL_PATH = (
    RUNS_DIR /
    "waste_detection" /
    "weights" /
    "best.pt"
)

model = YOLO(MODEL_PATH)


# ------------------------------------------------------------
# Helper Function
# ------------------------------------------------------------

def run_prediction(image_path: Path):
    """
    Runs YOLO prediction and returns detected objects.
    """

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        save=True,
        project=str(STATIC_DIR),
        name="predictions",
        exist_ok=True,
        verbose=False
    )

    result = results[0]

    detected = []

    if result.boxes is None:
        return detected

    for box in result.boxes:

        class_id = int(box.cls)

        confidence = float(box.conf)

        class_name = model.names[class_id]

        detected.append({
            "class": class_name,
            "confidence": round(confidence * 100, 2)
        })

        save_prediction(
            image_path.name,
            class_name,
            confidence
        )

    return detected


# ------------------------------------------------------------
# Home
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------
# History
# ------------------------------------------------------------

@app.route("/history")
def history():
    return render_template(
        "history.html",
        history=get_history()
    )


# ------------------------------------------------------------
# HTML Prediction
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]

    if file.filename == "":
        return "No file selected.", 400

    if not allowed(file.filename):
        return "Only JPG, JPEG and PNG images are allowed.", 400

    filename = cast(str, file.filename)

    image_path = UPLOAD_FOLDER / filename

    file.save(image_path)

    predictions = run_prediction(image_path)

    return render_template(
        "result.html",
        predictions=predictions,
        image="predictions/" + image_path.name
    )


# ------------------------------------------------------------
# REST API
# ------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "No image uploaded."
        }), 400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "No file selected."
        }), 400

    if not allowed(file.filename):

        return jsonify({
            "success": False,
            "message": "Only JPG, JPEG and PNG images are allowed."
        }), 400

    filename = cast(str, file.filename)

    image_path = UPLOAD_FOLDER / filename

    file.save(image_path)

    predictions = run_prediction(image_path)

    return jsonify({

        "success": True,

        "filename": image_path.name,

        "prediction_count": len(predictions),

        "predictions": predictions

    })


# ------------------------------------------------------------
# API Information
# ------------------------------------------------------------

@app.route("/api")
def api():

    return jsonify({

        "name": "Waste Detection API",

        "version": "1.0",

        "endpoint": "/api/predict",

        "method": "POST",

        "parameter": "image"

    })


# ------------------------------------------------------------
# Run Application
# ------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )