"""
Model registry — maps a string key to an annotator implementation.

This is the indirection that makes the system pluggable. The API accepts a
model key from the client, the registry resolves it to a class, and the job
pipeline runs it. Adding a model is one `register()` call; no route, schema, or
UI component changes.

It also owns the VRAM policy: AT MOST ONE MODEL RESIDENT AT A TIME.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.annotators.base import AutoAnnotator

logger = logging.getLogger(__name__)

# key -> class (not instance). Instances are created on demand so that merely
# importing this module never touches the GPU.
_REGISTRY: dict[str, type["AutoAnnotator"]] = {}

# Guards the single-resident-model invariant below. FastAPI runs sync endpoints
# in a threadpool and BackgroundTasks share the process, so two jobs really can
# race here — and two models loading concurrently on a small card is a
# guaranteed OOM rather than a theoretical one.
_lock = threading.Lock()
_resident: "AutoAnnotator | None" = None


def register(cls: type["AutoAnnotator"]) -> type["AutoAnnotator"]:
    """Register an annotator class. Usable as a decorator.

    Rejects duplicate keys loudly. A silent overwrite would mean the UI offers
    "Grounding DINO" and quietly runs something else — the kind of bug that
    wastes an afternoon.
    """
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty `key`")
    if cls.key in _REGISTRY and _REGISTRY[cls.key] is not cls:
        raise ValueError(f"Duplicate annotator key: {cls.key!r}")
    _REGISTRY[cls.key] = cls
    return cls


def available() -> list[dict]:
    """Metadata for every registered annotator — drives the UI's dropdown.

    Read off the CLASS, so nothing is instantiated and no weights load just to
    render a menu.
    """
    return [
        {
            "key": cls.key,
            "display_name": cls.display_name,
            # Fall back to the display name so an annotator that predates the
            # family/variant fields still groups (as a family of one).
            "family": cls.family or cls.display_name,
            "variant": cls.variant or "default",
            "description": cls.description,
            "approx_vram_gb": cls.approx_vram_gb,
        }
        for cls in _REGISTRY.values()
    ]


def get_class(key: str) -> type["AutoAnnotator"]:
    """Resolve a key to its class, or raise KeyError with a useful message."""
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown annotator {key!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def acquire(key: str) -> "AutoAnnotator":
    """Get a loaded annotator, evicting whatever else was resident.

    THE CORE VRAM POLICY. On a typical consumer GPU there is room for exactly
    one of these models, so acquiring a new one first unloads the old.

    Note the deliberate *caching* when the same key is requested twice in a row:
    annotating 200 images issues 200 acquire("grounding_dino") calls, and
    reloading 700 MB of weights each time would dominate the runtime. So we keep
    the model resident across a batch and only evict on a genuine switch.

    Callers must NOT use this as a context manager — that would unload the model
    after a single image and defeat the caching. Use it inside a job that owns
    the whole batch, and call `release()` when the batch is done.
    """
    global _resident

    with _lock:
        if _resident is not None and _resident.key == key and _resident.is_loaded:
            return _resident  # cache hit — the common path inside a batch

        if _resident is not None:
            logger.info("Evicting %s to make room for %s", _resident.key, key)
            _resident.unload()
            _resident = None

        cls = get_class(key)
        model = cls()
        model.load()
        _resident = model
        return model


def release(only: "AutoAnnotator | None" = None) -> None:
    """Unload whatever is resident. Call when a batch finishes.

    Without this, a model stays in VRAM indefinitely after a job ends — which is
    fine on a big card and fatal on this one, because the next job (or a
    different model) then has nothing to load into.

    `only`: unload ONLY IF the resident is this exact instance. This is a job
    runner saying "clean up MY model" — and it closes a real race: with GPU
    admission, the next queued job is admitted the moment the previous one's
    terminal status commits, which is BEFORE the previous runner's `finally`
    executes. An unconditional release there unloaded the NEXT job's
    freshly-acquired model out from under it, and every predict() after that
    raised "Model not loaded". (Observed live: three queued annotators, each
    unloaded by its predecessor's cleanup.) Identity, not key — two jobs using
    the same model key would still race on a key comparison.
    """
    global _resident
    with _lock:
        if _resident is None:
            return
        if only is not None and _resident is not only:
            return  # someone else's model now — their runner owns its cleanup
        _resident.unload()
        _resident = None


def resident_key() -> str | None:
    """Which model is currently in VRAM, if any. For the status UI."""
    return _resident.key if _resident is not None else None
