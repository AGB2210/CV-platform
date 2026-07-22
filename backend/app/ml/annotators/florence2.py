"""
Florence-2 — Microsoft's vision-language model, used for grounded detection.

A generative VLM rather than a detector: it literally WRITES the boxes as
text tokens, which transformers' processor parses back into coordinates. That
buys unusually good grounding of descriptive prompts ("the red container on
the left") and costs two things worth knowing:

  - ONE GENERATION PER CLASS per image. The open-vocabulary task takes a
    single phrase, so a 5-class project is 5 generate() calls per image.
    Fine for a careful pass, wrong for bulk annotation of large batches.
  - NO CONFIDENCE SCORES. Generated text has no per-box probability, so its
    boxes store confidence = NULL (like human-drawn boxes) and the box
    threshold does not apply. The review pass is the filter.

Loaded through transformers' NATIVE Florence-2 support (the
florence-community mirror), not trust_remote_code: the remote code in
microsoft/Florence-2-base predates transformers 5 and crashes against it —
verified here — while the native port is maintained with transformers itself.
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

_TASK = "<OPEN_VOCABULARY_DETECTION>"


@register
class Florence2Annotator(AutoAnnotator):
    key = "florence2_base"
    family = "Florence-2"
    variant = "base"
    display_name = "Florence-2 base"
    description = (
        "Microsoft's vision-language model — the best at grounding "
        "descriptive prompts ('the red container on the left'). Runs one pass "
        "per class per image and reports no confidence scores, so keep it for "
        "careful passes, not bulk batches."
    )
    approx_vram_gb = 2.5
    model_id = "florence-community/Florence-2-base"
    download_size = "~500 MB"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._device = get_device()

    def _load_impl(self) -> None:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self._processor = from_pretrained_with_fallback(
            AutoProcessor, self.model_id, self.download_size
        )
        model = from_pretrained_with_fallback(
            Florence2ForConditionalGeneration, self.model_id, self.download_size
        )
        self._model = model.to(self._device)
        self._model.eval()

    def _unload_impl(self) -> None:
        self._model = None
        self._processor = None

    def _detect_one(self, image: PILImage.Image, phrase: str) -> list[list[float]]:
        """Run the open-vocabulary task for ONE phrase; absolute xyxy boxes."""
        import torch

        inputs = self._processor(
            text=_TASK + phrase, images=image, return_tensors="pt"
        ).to(self._device)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=256,
                # Greedy. Beam search buys nothing for coordinate tokens and
                # multiplies the cost of what is already the slow annotator.
                num_beams=1,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            text, task=_TASK, image_size=image.size
        )
        return parsed.get(_TASK, {}).get("bboxes", [])

    def predict(self, request: AnnotationRequest) -> AnnotationResult:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() or use as a context manager")

        image = PILImage.open(request.image_path).convert("RGB")
        width, height = image.size
        started = time.perf_counter()

        boxes: list[Box] = []
        # One generation per class — see the module docstring. The label needs
        # no resolution: we asked about exactly one class, so every box the
        # answer contains is that class.
        for name in request.class_names:
            for x1, y1, x2, y2 in self._detect_one(image, request.prompt_for(name)):
                box = Box(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    label=name,
                    confidence=None,  # generated text carries no score
                ).clamp(width, height)
                if box.is_valid():
                    boxes.append(box)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Florence-2: %d boxes (%d classes) in %.0f ms for %s",
            len(boxes),
            len(request.class_names),
            elapsed_ms,
            request.image_path,
        )
        return AnnotationResult(
            boxes=boxes, image_width=width, image_height=height, inference_ms=elapsed_ms
        )
