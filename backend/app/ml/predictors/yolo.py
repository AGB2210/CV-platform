"""
YOLO predictor — run a trained YOLO11 checkpoint on an image.

The inference counterpart to YoloTrainer. Loads a `best.pt` and runs
`model.predict()`, converting ultralytics' output into the project's canonical
`Box` (absolute xyxy). Everything heavy (ultralytics, torch) imports lazily
inside methods — importing this module must stay free.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ml.annotators.base import Box
from app.ml.predictors.base import Predictor

logger = logging.getLogger(__name__)


class YoloPredictor(Predictor):
    key = "yolo"

    def __init__(self, checkpoint_path: str | Path, class_names: list[str]) -> None:
        super().__init__()
        self._checkpoint = str(checkpoint_path)
        # The class list the checkpoint was trained on, in the order that fixes
        # channel-to-class mapping. ultralytics stores its own names in the .pt,
        # but we pass ours so the labels match the project exactly, and so a
        # future re-export can't silently reorder them underneath us.
        self._class_names = class_names
        self._model = None

    def _load_impl(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Open Auto-annotate or Train once "
                "to install the ML dependencies, then retry."
            ) from exc
        self._model = YOLO(self._checkpoint)

    def _unload_impl(self) -> None:
        self._model = None

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        if self._model is None:
            raise RuntimeError("predict() called before load()")

        from app.ml.device import get_device

        device = 0 if get_device() == "cuda" else "cpu"
        # verbose off: we render results ourselves. One image at a time — the
        # playground and the evaluator both call per-image.
        results = self._model.predict(
            source=image_path,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )

        boxes: list[Box] = []
        if not results:
            return boxes
        result = results[0]
        # result.boxes is an ultralytics Boxes object: .xyxy (absolute pixels),
        # .conf, .cls (class index into the model's names). We map the index
        # through OUR class list so the stored label is the project's name.
        img_h, img_w = result.orig_shape  # (height, width)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            idx = int(cls)
            label = self._class_names[idx] if 0 <= idx < len(self._class_names) else str(idx)
            box = Box(
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                label=label, confidence=float(conf),
            ).clamp(img_w, img_h)
            if box.is_valid():
                boxes.append(box)
        return boxes
