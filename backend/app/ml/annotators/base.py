"""
The AutoAnnotator interface.

This is the seam that makes auto-annotation models pluggable. Adding a new model
means writing one class here and registering it — the job pipeline, the API, and
the UI never change, because they only ever talk to this interface.

DESIGN NOTES
------------
Why an abstract base class rather than a Protocol?

A Protocol gives structural typing with no runtime enforcement — you find out a
method is missing when the job crashes, ten minutes into a batch. These classes
are loaded dynamically from a registry keyed by a string that arrives from an
HTTP request, so an ABC's "can't instantiate with unimplemented abstract
methods" error at construction time is worth having. It also gives us a place to
put shared behaviour (the load/unload lifecycle) rather than copying it into
every implementation.

Why an explicit load()/unload() lifecycle rather than loading in __init__?

Because this machine has 4 GB of VRAM. Three annotators loaded at import time
would OOM before serving a single request. Models are loaded on demand, and the
registry guarantees at most one is resident (see registry.py). Grounded SAM
depends on this directly: it cannot hold Grounding DINO and SAM simultaneously,
so it loads DINO, runs it, unloads, then loads SAM.

This constraint is a gift. Even on a 24 GB card the right design is lazy loading
with explicit lifetimes — the small GPU just makes it non-optional.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Box:
    """One predicted bounding box.

    Coordinates are ABSOLUTE PIXELS in xyxy order (left, top, right, bottom),
    relative to the original image's top-left origin.

    Fixing this convention here, once, is the entire point. Every model in this
    space disagrees:
      - Grounding DINO outputs normalised cxcywh (centre-x, centre-y, w, h)
      - YOLO outputs normalised xywh
      - COCO stores absolute xywh (top-left + width/height)
      - torchvision expects absolute xyxy
    Silent box-format mismatches are the single most common bug in detection
    code, and they don't crash — they just produce boxes in slightly wrong
    places, which you notice three phases later. Each adapter converts to THIS
    format at its boundary, and nothing downstream has to ask.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    label: str  # the class name this box was matched to
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def to_coco_bbox(self) -> list[float]:
        """Convert to COCO's [x, y, width, height] with a top-left origin."""
        return [self.x1, self.y1, self.width, self.height]

    def clamp(self, img_width: int, img_height: int) -> "Box":
        """Clip the box to the image bounds.

        Models routinely predict boxes that extend past the edge for objects
        that are partially out of frame. A negative x1, or an x2 beyond the
        image width, produces a COCO file that pycocotools will happily accept
        and then compute nonsense mAP from — so we clamp at the boundary rather
        than letting it leak downstream.
        """
        return Box(
            x1=max(0.0, min(self.x1, img_width)),
            y1=max(0.0, min(self.y1, img_height)),
            x2=max(0.0, min(self.x2, img_width)),
            y2=max(0.0, min(self.y2, img_height)),
            label=self.label,
            confidence=self.confidence,
        )

    def is_valid(self, min_size: float = 2.0) -> bool:
        """Reject degenerate boxes (zero/negative area) after clamping.

        A box entirely outside the frame clamps to zero width — real, and worth
        discarding rather than writing a 0x0 annotation into the dataset.
        """
        return self.width >= min_size and self.height >= min_size


@dataclass
class AnnotationRequest:
    """What to annotate, and how.

    A dataclass rather than a pile of positional arguments: every annotator
    receives the same request shape, so the job pipeline builds one object and
    hands it to whichever model the user picked, without knowing which.
    """

    image_path: str
    # Class names to look for, e.g. ["car", "person"]. These come from the
    # project's Category rows.
    class_names: list[str]
    # Optional richer phrasing per class, e.g. {"car": "a parked car"}.
    # Grounding DINO is genuinely sensitive to prompt wording, so we let the
    # user override the bare class name without changing the label we store.
    prompts: dict[str, str] = field(default_factory=dict)
    box_threshold: float = 0.30
    text_threshold: float = 0.25

    def prompt_for(self, class_name: str) -> str:
        """The text to feed the model for a class; falls back to the name."""
        return self.prompts.get(class_name, class_name)


@dataclass
class AnnotationResult:
    """Boxes found in one image, plus what it cost."""

    boxes: list[Box]
    image_width: int
    image_height: int
    inference_ms: float = 0.0


class AutoAnnotator(ABC):
    """Base class for zero-shot annotation models.

    Subclasses implement three things: metadata (name/description), how to load,
    and how to predict. The lifecycle plumbing lives here.

    Usage is always through the context manager, which guarantees unload():

        with GroundingDinoAnnotator() as ann:
            result = ann.predict(request)
        # VRAM released here, even if predict() raised
    """

    #: Stable identifier used in the registry, the API, and the DB. Changing it
    #: breaks existing job records, so treat it as permanent.
    key: str = ""
    #: Shown in the UI's model dropdown.
    display_name: str = ""
    description: str = ""
    #: Rough VRAM cost, surfaced in the UI so the user can tell what will fit.
    approx_vram_gb: float = 0.0

    def __init__(self) -> None:
        self._loaded = False

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load weights onto the device. Idempotent.

        Idempotency matters because the registry may call load() on a model it
        believes is already resident. Downloading multi-GB weights twice because
        of a double call is a bad way to find out.
        """
        if self._loaded:
            return
        logger.info("Loading %s…", self.display_name)
        self._load_impl()
        self._loaded = True
        logger.info("Loaded %s", self.display_name)

    def unload(self) -> None:
        """Release the model and its VRAM. Idempotent."""
        if not self._loaded:
            return
        logger.info("Unloading %s…", self.display_name)
        self._unload_impl()
        self._loaded = False

        # Dropping the Python reference is NOT enough — torch's caching
        # allocator holds the freed blocks. On a 4 GB card the next model will
        # OOM unless we hand them back. See device.empty_cache().
        from app.ml.device import empty_cache

        empty_cache()
        logger.info("Unloaded %s", self.display_name)

    def __enter__(self) -> "AutoAnnotator":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        # Runs on the exception path too, which is the whole reason for the
        # context manager: a model that OOMs mid-batch must still release what
        # it did allocate, or every subsequent job fails too.
        self.unload()

    # --- to implement ------------------------------------------------------

    @abstractmethod
    def _load_impl(self) -> None:
        """Actually load the weights. Called once by load()."""

    @abstractmethod
    def _unload_impl(self) -> None:
        """Drop references to model objects. Called once by unload()."""

    @abstractmethod
    def predict(self, request: AnnotationRequest) -> AnnotationResult:
        """Detect the requested classes in one image.

        Must return boxes in ABSOLUTE xyxy pixels, clamped to the image, with
        degenerate boxes removed. Converting from the model's native format is
        the adapter's job, not the caller's — that is precisely what this
        interface is for.
        """
