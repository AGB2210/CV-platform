"""
RF-DETR predictor — run a trained RF-DETR checkpoint on an image.

The inference counterpart to the RF-DETR trainers. `rfdetr.from_checkpoint`
rebuilds the exact architecture the .pth was trained with, and predict()
returns supervision Detections (absolute xyxy + confidence + class_id).

class_id is the COCO category id the model was TRAINED on — our exporter
assigns those 1-based in class order, so id N maps to class_names[N-1]. The
class list is passed in by the caller (resolved from the run's dataset
version) rather than trusted from the checkpoint, for the same reason the
YOLO predictor does it: labels must match the project exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ml.annotators.base import Box
from app.ml.predictors.base import Predictor

logger = logging.getLogger(__name__)


class RfDetrPredictor(Predictor):
    key = "rfdetr"

    def __init__(self, checkpoint_path: str | Path, class_names: list[str]) -> None:
        super().__init__()
        self._checkpoint = str(checkpoint_path)
        self._class_names = class_names
        self._model = None

    def _load_impl(self) -> None:
        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is not installed. Open Train once to install the ML "
                "dependencies, then retry."
            ) from exc
        self._model = rfdetr.from_checkpoint(self._checkpoint)

    def _unload_impl(self) -> None:
        self._model = None

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        if self._model is None:
            raise RuntimeError("predict() called before load()")

        from PIL import Image as PILImage

        with PILImage.open(image_path) as im:
            image = im.convert("RGB")
            img_w, img_h = image.size
            detections = self._model.predict(image, threshold=conf_threshold)

        boxes: list[Box] = []
        for xyxy, conf, cls in zip(
            detections.xyxy.tolist(),
            detections.confidence.tolist(),
            detections.class_id.tolist(),
        ):
            # COCO category ids are 1-based in our exports; index our list.
            idx = int(cls) - 1
            label = (
                self._class_names[idx]
                if 0 <= idx < len(self._class_names)
                else str(cls)
            )
            box = Box(
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                label=label, confidence=float(conf),
            ).clamp(img_w, img_h)
            if box.is_valid():
                boxes.append(box)
        return boxes
