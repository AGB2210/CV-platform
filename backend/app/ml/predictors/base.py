"""
The Predictor interface — run a trained checkpoint on an image.

WHY IT LOOKS LIKE AutoAnnotator, NOT Trainer
--------------------------------------------
A trainer runs ONCE per job and owns its whole loop, so its interface is a single
train() call. A predictor is the opposite: it is called REPEATEDLY — 30 val images
during an evaluation, or a playground the user pokes at for ten minutes — so the
weights must stay resident across calls. That is exactly AutoAnnotator's shape, so
this mirrors it: a load/unload lifecycle plus predict(), with a registry keeping at
most one resident (see registry.py).

WHY Box IS REUSED, NOT RE-INVENTED
----------------------------------
`predict()` returns the SAME `Box` the annotators return — absolute xyxy pixels,
with clamp() and is_valid(). Box exists precisely because box-format mismatches
(normalised vs absolute, xywh vs xyxy, centre vs corner) are the most common silent
bug in detection code, and having one canonical type is what prevents it. A parallel
"PredictionBox" would be that bug waiting to happen, so there isn't one.

HOW A PREDICTOR IS OBTAINED
---------------------------
Not from a registry keyed by architecture — from the TRAINER that produced the
checkpoint (`Trainer.load_predictor(...)`). The trainer that wrote the weights is the
only thing that knows how to read them, so the pairing is structural rather than a
second mapping that could drift out of step with the first. torch is imported lazily
inside the concrete implementation, never at module load.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.ml.annotators.base import Box

logger = logging.getLogger(__name__)


class Predictor(ABC):
    """Base class for running a trained checkpoint. Load once, predict many.

    Usage is through the context manager, which guarantees unload() even if a
    prediction raises — the same contract as AutoAnnotator, and for the same
    reason (a model that OOMs mid-batch must still release what it allocated, or
    every job after it fails too):

        with trainer.load_predictor(ckpt, class_names) as p:
            boxes = p.predict("image.jpg", conf_threshold=0.25)
    """

    #: The trainer key this predictor pairs with — set by load_predictor so the
    #: registry can tell a resident YOLO predictor from a future DETR one.
    key: str = ""

    def __init__(self) -> None:
        self._loaded = False

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the checkpoint onto the device. Idempotent."""
        if self._loaded:
            return
        logger.info("Loading predictor %s…", self.key or type(self).__name__)
        self._load_impl()
        self._loaded = True

    def unload(self) -> None:
        """Release the model and its VRAM. Idempotent."""
        if not self._loaded:
            return
        self._unload_impl()
        self._loaded = False
        # Dropping the Python reference is not enough — torch's caching allocator
        # keeps the freed blocks. On a tight card the next model OOMs unless we
        # hand them back. Same reason AutoAnnotator.unload does this.
        from app.ml.device import empty_cache

        empty_cache()
        logger.info("Unloaded predictor %s", self.key or type(self).__name__)

    def __enter__(self) -> "Predictor":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.unload()

    # --- to implement ------------------------------------------------------

    @abstractmethod
    def _load_impl(self) -> None:
        """Load the checkpoint. Called once by load()."""

    @abstractmethod
    def _unload_impl(self) -> None:
        """Drop references to model objects. Called once by unload()."""

    @abstractmethod
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        """Detect objects in one image.

        Returns boxes in ABSOLUTE xyxy pixels, clamped to the image, degenerate
        boxes removed — converting from the framework's native format is the
        adapter's job. `conf_threshold` is the precision/recall dial: only boxes
        the model scores at or above it are returned.
        """
