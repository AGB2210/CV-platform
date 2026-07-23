"""
OWLv2 — Google's zero-shot detector (via transformers).

A different lineage from Grounding DINO (OWL-ViT scaled up with
self-training), which is exactly why it earns a slot: the two models miss
DIFFERENT things. OWLv2 is notably strong on rare and fine-grained classes —
when DINO returns nothing for an unusual vocabulary, this is the second
opinion to try. Slower per image than DINO-tiny; not a bulk-throughput choice.

Cleanest label story in the roster: queries are passed as a LIST and results
come back as an INDEX into it, so a box's class is a straight lookup — none of
the phrase-echo ambiguity DINO needs _resolve_label for.
"""

from __future__ import annotations

import logging
import time

from PIL import Image as PILImage

from app.ml.annotators._hf import from_pretrained_with_fallback
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
class Owlv2Annotator(AutoAnnotator):
    key = "owlv2_base"
    family = "OWLv2"
    variant = "base (ensemble)"
    display_name = "OWLv2 base"
    description = (
        "Google's zero-shot detector — strong on rare and fine-grained "
        "classes where Grounding DINO comes up empty. Slower per image; use "
        "it as the second opinion, not the bulk annotator."
    )
    approx_vram_gb = 3.0
    model_id = "google/owlv2-base-patch16-ensemble"
    download_size = "~1.2 GB"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._device = get_device()

    def _load_impl(self) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._processor = from_pretrained_with_fallback(
            Owlv2Processor, self.model_id, self.download_size
        )
        model = from_pretrained_with_fallback(
            Owlv2ForObjectDetection, self.model_id, self.download_size
        )
        self._model = model.to(self._device)
        self._model.eval()

    def _unload_impl(self) -> None:
        self._model = None
        self._processor = None

    def predict(self, request: AnnotationRequest) -> AnnotationResult:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() or use as a context manager")

        image = PILImage.open(request.image_path).convert("RGB")
        width, height = image.size

        # One query per class, prompts honoured. Index i of this list IS
        # class_names[i] in the results — the whole label story.
        queries = [request.prompt_for(name) for name in request.class_names]
        if not queries:
            return AnnotationResult(boxes=[], image_width=width, image_height=height)

        started = time.perf_counter()
        inputs = self._processor(
            text=[queries], images=image, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        # (height, width) ordering — the opposite of PIL's (width, height),
        # and the classic way to get silently transposed boxes.
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=request.box_threshold,
            target_sizes=[(height, width)],
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        boxes: list[Box] = []
        for box_t, score_t, label_t in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            idx = int(label_t.item())
            if not 0 <= idx < len(request.class_names):
                continue
            x1, y1, x2, y2 = (float(v) for v in box_t.tolist())
            box = Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=request.class_names[idx],
                confidence=float(score_t.item()),
            ).clamp(width, height)
            if box.is_valid():
                boxes.append(box)

        logger.info(
            "OWLv2: %d boxes in %.0f ms for %s",
            len(boxes),
            elapsed_ms,
            request.image_path,
        )
        return AnnotationResult(
            boxes=boxes, image_width=width, image_height=height, inference_ms=elapsed_ms
        )


@register
class Owlv2LargeAnnotator(Owlv2Annotator):
    """The large ensemble — the same second-opinion role, stronger still on
    fine-grained classes, at roughly triple the compute of base."""

    key = "owlv2_large"
    variant = "large (ensemble)"
    display_name = "OWLv2 large"
    description = (
        "The large OWLv2 — the strongest rare-class detector in the roster. "
        "Slow per image; reserve it for the classes everything else misses."
    )
    approx_vram_gb = 6.0
    model_id = "google/owlv2-large-patch14-ensemble"
    download_size = "~1.8 GB"
