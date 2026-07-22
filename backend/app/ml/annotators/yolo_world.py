"""
YOLO-World — real-time open-vocabulary detection (via ultralytics).

The fast lane for bulk auto-annotation: it detects arbitrary classes from
text like Grounding DINO, but runs an order of magnitude faster because the
text encoding happens ONCE (set_classes embeds the vocabulary with CLIP) and
every image after that is a plain YOLO forward pass. On a 500-image batch
that's the difference between minutes and most of an hour.

Trade-off worth knowing: on rare or oddly-phrased classes DINO's deeper
language grounding wins. That's why both are in the roster — YOLO-World for
throughput on ordinary vocabulary, DINO when the classes get weird.

ultralytics pulls its CLIP text encoder in on first use (requirements-ml
includes it, and ultralytics auto-installs it as a fallback).
"""

from __future__ import annotations

import logging
import time

from PIL import Image as PILImage

from app.ml.annotators.base import (
    AnnotationRequest,
    AnnotationResult,
    AutoAnnotator,
    Box,
)
from app.ml.device import get_device
from app.ml.registry import register

logger = logging.getLogger(__name__)


@register
class YoloWorldAnnotator(AutoAnnotator):
    key = "yolo_world_s"
    family = "YOLO-World"
    variant = "small (v2)"
    display_name = "YOLO-World S"
    description = (
        "Real-time open-vocabulary detection — the fast choice for large "
        "batches. Encodes your class names once, then runs at YOLO speed. "
        "Grounding DINO grounds unusual phrasings better; this wins on "
        "throughput."
    )
    approx_vram_gb = 2.0
    base_weights = "yolov8s-worldv2.pt"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._device = get_device()
        # What set_classes was last called with. Re-encoding the vocabulary
        # loads a CLIP text model each time — cheap once per batch, absurd per
        # image — so predict() only re-sets on an actual change.
        self._vocab: tuple[str, ...] | None = None
        self._vocab_class_names: list[str] = []

    def _load_impl(self) -> None:
        # Lazy import: ultralytics pulls torch, and listing models in the UI
        # must never pay for that.
        from ultralytics import YOLOWorld

        self._model = YOLOWorld(self.base_weights)
        self._vocab = None

    def _unload_impl(self) -> None:
        self._model = None
        self._vocab = None

    def _ensure_vocab(self, request: AnnotationRequest) -> None:
        """set_classes with the request's prompts, once per distinct vocabulary.

        The PROMPT text is what gets embedded ("a parked car"), but the model's
        class index maps back to our CLASS NAME by position — so unlike DINO
        there is no phrase-echo ambiguity to resolve: index i IS class i.
        """
        phrases = tuple(request.prompt_for(name) for name in request.class_names)
        if phrases == self._vocab:
            return
        self._model.set_classes(list(phrases))
        self._vocab = phrases
        self._vocab_class_names = list(request.class_names)

    def predict(self, request: AnnotationRequest) -> AnnotationResult:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() or use as a context manager")

        image = PILImage.open(request.image_path).convert("RGB")
        width, height = image.size

        self._ensure_vocab(request)

        started = time.perf_counter()
        device = 0 if self._device == "cuda" else "cpu"
        results = self._model.predict(
            source=request.image_path,
            conf=request.box_threshold,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000

        boxes: list[Box] = []
        if results:
            r = results[0]
            for xyxy, conf, cls in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
            ):
                idx = int(cls)
                if not 0 <= idx < len(self._vocab_class_names):
                    continue  # an index outside our vocabulary labels nothing
                box = Box(
                    x1=xyxy[0],
                    y1=xyxy[1],
                    x2=xyxy[2],
                    y2=xyxy[3],
                    label=self._vocab_class_names[idx],
                    confidence=float(conf),
                ).clamp(width, height)
                if box.is_valid():
                    boxes.append(box)

        logger.info(
            "YOLO-World: %d boxes in %.0f ms for %s",
            len(boxes),
            elapsed_ms,
            request.image_path,
        )
        return AnnotationResult(
            boxes=boxes, image_width=width, image_height=height, inference_ms=elapsed_ms
        )
