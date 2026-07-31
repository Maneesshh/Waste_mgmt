from pathlib import Path
from src.history import save_prediction
from src.config import *

from ultralytics import YOLO

from src.config import (
    MODEL_PATH,
    IMAGE_SIZE,
    CONFIDENCE,
    STATIC_DIR
)

from src.history import save_prediction


class WastePredictor:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(

                f"Model not found:\n{MODEL_PATH}"

            )

        self.model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------

    def predict(self, image_path: Path):

        """
        Predict objects in an image.

        Returns
        -------
        list
            List of detected objects.
        """

        results = self.model.predict(

            source=str(image_path),

            imgsz=IMAGE_SIZE,

            conf=CONFIDENCE,

            save=True,

            project=str(STATIC_DIR),

            name="predictions",

            exist_ok=True,

            verbose=False

        )

        detections = []

        if len(results) == 0:

            return detections

        result = results[0]

        if result.boxes is None:

            return detections

        for box in result.boxes:

            class_id = int(box.cls.item())

            confidence = float(box.conf.item())

            class_name = self.model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({

                "class": class_name,

                "confidence": round(confidence * 100, 2),

                "bbox": {

                    "x1": round(x1, 2),

                    "y1": round(y1, 2),

                    "x2": round(x2, 2),

                    "y2": round(y2, 2)

                }

            })

            save_prediction(

                image_path.name,

                class_name,

                confidence

            )

        return detections