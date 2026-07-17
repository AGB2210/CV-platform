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

from app.ml.annotators.base import (
    AnnotationRequest,
    AnnotationResult,
    AutoAnnotator,
    Box,
)
from app.ml.device import get_device
from app.ml.registry import register

logger = logging.getLogger(__name__)

# Swin-T backbone, ~700 MB. The "base" variant (Swin-B) is ~1.8 GB and needs
# roughly 5-6 GB to run — it does not fit this machine's 4 GB card. Tiny is the
# correct choice here, not a compromise we're apologising for: on common object
# categories the gap is small, and the review UI exists to fix what it misses.
MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@register
class GroundingDinoAnnotator(AutoAnnotator):
    key = "grounding_dino"
    display_name = "Grounding DINO"
    description = (
        "Zero-shot detection from text prompts. Best default for open-vocabulary "
        "bounding boxes — no training needed."
    )
    approx_vram_gb = 2.5

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

        # First call downloads ~700 MB to the HuggingFace cache; later calls
        # should be local. The cache lives outside the repo and is gitignored.
        #
        # ONCE CACHED, THIS MUST NOT NEED THE NETWORK.
        #
        # from_pretrained() still phones home to check for updates even when
        # every byte is already on disk, so anything that upsets that call —
        # offline, HF down, rate limiting, or a stale auth token — fails a job
        # whose weights are sitting right there. That happened here: an expired
        # token in ~/.cache/huggingface/token got a 401, which transformers
        # reports as the deeply misleading "not a valid model identifier".
        #
        # For a tool whose whole premise is that it runs locally, depending on
        # huggingface.co at inference time is a bug. So: try online (needed for
        # the genuine first download), and on ANY failure fall back to the
        # cache. Broad except on purpose — the failure modes are network errors,
        # HTTP errors, and OSError depending on how deep it got, and the
        # response is identical for all of them.
        try:
            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not reach HuggingFace for %s (%s). Falling back to the "
                "local cache.",
                MODEL_ID,
                exc,
            )
            try:
                self._processor = AutoProcessor.from_pretrained(
                    MODEL_ID, local_files_only=True
                )
                model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    MODEL_ID, local_files_only=True
                )
            except Exception:
                # Genuinely not cached: the first run does need a download, and
                # saying so beats re-raising a 401 that blames the model id.
                raise RuntimeError(
                    f"{MODEL_ID} is not in the local cache and HuggingFace could "
                    f"not be reached. The first run needs to download ~700 MB. "
                    f"If you have an expired HuggingFace token, remove "
                    f"~/.cache/huggingface/token — a stale token is rejected with "
                    f"401 even for public models, while anonymous access works."
                ) from exc
            logger.info("Loaded %s from the local cache", MODEL_ID)

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
    def _resolve_label(raw: str, back_map: dict[str, str], fallback: str) -> str:
        """Map a model-returned phrase back to one of our class names.

        Needed because the model doesn't always echo a prompt phrase verbatim.
        It may return a sub-span ("car" from "a parked car"), merge adjacent
        phrases, or return an empty string. Three passes, most-specific first;
        a wrong label silently mislabels training data, so this is worth the
        care.
        """
        cleaned = raw.lower().strip(" .")
        if not cleaned:
            return fallback

        # 1. Exact match on the prompted phrase.
        if cleaned in back_map:
            return back_map[cleaned]

        # 2. The returned span is contained in a prompt phrase, or vice versa.
        for phrase, class_name in back_map.items():
            if cleaned in phrase or phrase in cleaned:
                return class_name

        # 3. Any prompt phrase shares a word with the returned span.
        words = set(cleaned.split())
        for phrase, class_name in back_map.items():
            if words & set(phrase.split()):
                return class_name

        return fallback

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
        # graph for every forward pass — pure waste here, and on a 4 GB card the
        # stored activations are enough to OOM.
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
        fallback = request.class_names[0] if request.class_names else "object"

        for box_t, score_t, raw_label in zip(
            results["boxes"], results["scores"], raw_labels, strict=False
        ):
            # .tolist() pulls the tensor to CPU python floats. Keeping tensors
            # around here would pin VRAM for the whole batch.
            x1, y1, x2, y2 = (float(v) for v in box_t.tolist())
            label = self._resolve_label(str(raw_label), back_map, fallback)

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
            "Grounding DINO: %d boxes in %.0f ms for %s",
            len(boxes),
            elapsed_ms,
            request.image_path,
        )
        return AnnotationResult(
            boxes=boxes,
            image_width=width,
            image_height=height,
            inference_ms=elapsed_ms,
        )
