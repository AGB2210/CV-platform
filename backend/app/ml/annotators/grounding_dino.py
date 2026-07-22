"""
Grounding DINO — open-vocabulary detection from a text prompt.

The default auto-annotator. Give it "car. person." and it returns boxes for cars
and people, with no training and no fixed class list. That's what makes
zero-shot labelling possible: the model was trained to ground arbitrary language
in images, so your classes don't have to have existed when it was trained.

We use the `transformers` port rather than the original IDEA-Research repo. The
research repo pins old torch versions, needs a CUDA toolchain to build custom
ops, and is genuinely painful on Windows. The transformers port is pure PyTorch,
installs from a wheel, and shares one weight cache with SAM.
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
class GroundingDinoAnnotator(AutoAnnotator):
    key = "grounding_dino"
    family = "Grounding DINO"
    # The variant is named, not just the family: Grounding DINO ships tiny
    # (Swin-T) and base (Swin-B) checkpoints that differ several-fold in size and
    # accuracy, and which one is running is exactly what a user picking from a
    # list needs to know.
    variant = "tiny (Swin-T)"
    display_name = "Grounding DINO tiny"
    description = (
        "Zero-shot detection from text prompts, using the grounding-dino-tiny "
        "checkpoint. Best default for open-vocabulary bounding boxes — no "
        "training needed."
    )
    approx_vram_gb = 2.5
    #: HF checkpoint. Subclasses swap this to register the bigger sibling.
    model_id = "IDEA-Research/grounding-dino-tiny"
    download_size = "~700 MB"

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None
        self._device = get_device()

    # --- lifecycle ---------------------------------------------------------

    def _load_impl(self) -> None:
        # Imported inside the method, not at module top level. This module is
        # imported at startup to populate the registry (so the UI can list
        # models), and importing transformers costs seconds and hundreds of MB.
        # Serving /api/projects must not pay for that.
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        # First call downloads the weights to the HuggingFace cache; later
        # calls must work offline — the fallback policy lives in _hf.py, shared
        # by every transformers-backed annotator.
        self._processor = from_pretrained_with_fallback(
            AutoProcessor, self.model_id, self.download_size
        )
        model = from_pretrained_with_fallback(
            AutoModelForZeroShotObjectDetection, self.model_id, self.download_size
        )

        self._model = model.to(self._device)
        # eval() disables dropout and batchnorm updates. Mandatory for
        # inference; forgetting it produces subtly worse, non-deterministic
        # boxes rather than an error.
        self._model.eval()

    def _unload_impl(self) -> None:
        # Dropping the references is all we do here; base.unload() calls
        # torch.cuda.empty_cache() afterwards to actually return the VRAM.
        self._model = None
        self._processor = None

    # --- prompt handling ---------------------------------------------------

    @staticmethod
    def _build_prompt(request: AnnotationRequest) -> tuple[str, dict[str, str]]:
        """Build DINO's text prompt and a map back to our class names.

        Grounding DINO expects lowercase phrases separated by ' . ' and ending
        in a period — e.g. "car . person . traffic light ." This is not
        cosmetic: the model was trained on that exact format, and deviating
        (commas, capitals, no trailing period) measurably degrades detection.

        We also return prompt_text -> class_name, because the model echoes back
        the phrase it matched, not our class. If the user prompted "a parked
        car" for the class "car", the result says "a parked car" and we must map
        it home before storing the annotation.
        """
        phrases: list[str] = []
        back_map: dict[str, str] = {}
        for name in request.class_names:
            phrase = request.prompt_for(name).lower().strip()
            if not phrase:
                continue
            phrases.append(phrase)
            back_map[phrase] = name
        prompt = " . ".join(phrases) + " ."
        return prompt, back_map

    @staticmethod
    def _resolve_label(raw: str, back_map: dict[str, str]) -> str | None:
        """Map a model-returned phrase back to one of our class names.

        Returns None when the phrase does not identify exactly one class. The
        caller DROPS those boxes.

        WHY NONE RATHER THAN A BEST GUESS. This used to fall back to
        `class_names[0]` whenever it couldn't decide, which is how a real
        mislabelling bug shipped: Grounding DINO returns an EMPTY label for
        low-confidence detections, so on a two-class "car, person" project every
        unlabelled box became a "car". Measured on the shapes demo, every
        person-shaped box the model found below ~0.3 confidence was stored as a
        car. That surfaces as "people don't get annotated", but the truth is
        worse: they ARE annotated, as the wrong class. The box looks right in
        review and only the label is wrong, which is training data being
        quietly poisoned.

        A box whose class we cannot determine is not data. Dropping it loses a
        detection the human can redraw; keeping it invents ground truth that
        looks legitimate and is wrong.

        Ambiguity is treated the same way. The model merges adjacent prompt
        phrases — "car . person ." comes back as the single span "car person" —
        and the old code resolved that by taking whichever class it iterated
        first, so the answer depended on class ORDER rather than on the image.
        A span that names two classes identifies neither.
        """
        cleaned = raw.lower().strip(" .")
        if not cleaned:
            return None

        # 1. Exact match on the prompted phrase — unambiguous by construction.
        if cleaned in back_map:
            return back_map[cleaned]

        # 2. The returned span is contained in a prompt phrase, or vice versa.
        #    Collect every match: "car person" contains both "car" and "person",
        #    and answering that with one of them is a coin toss.
        matches = {
            class_name
            for phrase, class_name in back_map.items()
            if cleaned in phrase or phrase in cleaned
        }
        if len(matches) == 1:
            return matches.pop()
        if matches:
            return None  # names several classes — identifies none

        # 3. Loosest pass: a shared word ("parked car" -> "car").
        words = set(cleaned.split())
        matches = {
            class_name
            for phrase, class_name in back_map.items()
            if words & set(phrase.split())
        }
        return matches.pop() if len(matches) == 1 else None

    # --- inference ---------------------------------------------------------

    def predict(self, request: AnnotationRequest) -> AnnotationResult:
        import torch

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded — call load() or use as a context manager")

        image = PILImage.open(request.image_path).convert("RGB")
        width, height = image.size

        prompt, back_map = self._build_prompt(request)
        if not back_map:
            return AnnotationResult(boxes=[], image_width=width, image_height=height)

        started = time.perf_counter()

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(
            self._device
        )

        # inference_mode is torch.no_grad's stricter sibling: it also skips
        # version-counter bookkeeping. Without it, torch builds an autograd
        # graph for every forward pass — pure waste here, and on a small card
        # the stored activations are enough to OOM.
        with torch.inference_mode():
            outputs = self._model(**inputs)

        # target_sizes converts the model's normalised cxcywh boxes into
        # absolute xyxy at the ORIGINAL image resolution — undoing the resize
        # the processor applied. Note the (height, width) ordering, which is the
        # opposite of PIL's (width, height) and an easy way to silently get
        # transposed boxes.
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        elapsed_ms = (time.perf_counter() - started) * 1000

        boxes: list[Box] = []
        raw_labels = results.get("text_labels", results.get("labels", []))
        unlabelled = 0

        for box_t, score_t, raw_label in zip(
            results["boxes"], results["scores"], raw_labels, strict=False
        ):
            # .tolist() pulls the tensor to CPU python floats. Keeping tensors
            # around here would pin VRAM for the whole batch.
            x1, y1, x2, y2 = (float(v) for v in box_t.tolist())
            label = self._resolve_label(str(raw_label), back_map)
            if label is None:
                # The model found something but didn't say what. Guessing a
                # class here is how wrong ground truth gets manufactured — see
                # _resolve_label. Count it so the log shows recall being lost.
                unlabelled += 1
                continue

            box = Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=label,
                confidence=float(score_t.item()),
            ).clamp(width, height)

            if box.is_valid():
                boxes.append(box)

        logger.info(
            "Grounding DINO: %d boxes in %.0f ms for %s%s",
            len(boxes),
            elapsed_ms,
            request.image_path,
            f" ({unlabelled} dropped — model gave no usable class)" if unlabelled else "",
        )
        return AnnotationResult(
            boxes=boxes,
            image_width=width,
            image_height=height,
            inference_ms=elapsed_ms,
        )


@register
class GroundingDinoBaseAnnotator(GroundingDinoAnnotator):
    """The Swin-B sibling: noticeably better boxes for ~2x the memory and time.

    Same code path end to end — only the checkpoint differs. Tiny remains the
    default recommendation on small cards; base is the incremental upgrade for
    anyone with the VRAM to spend on fewer review corrections.
    """

    key = "grounding_dino_base"
    variant = "base (Swin-B)"
    display_name = "Grounding DINO base"
    description = (
        "The bigger Grounding DINO — better boxes than tiny, roughly twice the "
        "memory and inference time. Worth it on 8 GB+ cards."
    )
    approx_vram_gb = 5.5
    model_id = "IDEA-Research/grounding-dino-base"
    download_size = "~1.8 GB"
