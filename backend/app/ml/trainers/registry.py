"""
Trainer registry — maps a string key to a Trainer implementation.

Mirrors app.ml.registry (the annotator one) so the two read the same, but it is
deliberately SIMPLER. The annotator registry owns a resident-model cache because
we call a model 200 times in a batch and reloading weights each time would
dominate. A trainer is called ONCE per job and manages its own model internally,
so there is nothing to cache and no acquire/release/evict dance here — just
metadata lookup and key resolution.

The single-resident-model rule still applies, but it's enforced elsewhere,
because it spans registries:
  - the job runner calls the ANNOTATOR registry's release() before training, so
    a model left resident from an auto-annotate run can't collide with training;
  - the train route refuses to queue a second training (or an overlapping
    annotate) job, so two heavy GPU workloads never run at once.
Putting that policy here would be the wrong place — it isn't the trainer's to
know about the annotator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.trainers.base import Trainer

logger = logging.getLogger(__name__)

# key -> class (not instance). Instances are created per job; importing this
# module must never touch the GPU or the heavy training deps.
_REGISTRY: dict[str, type["Trainer"]] = {}


def register(cls: type["Trainer"]) -> type["Trainer"]:
    """Register a trainer class. Usable as a decorator.

    Rejects duplicate keys loudly — a silent overwrite would mean the UI offers
    "YOLO" and quietly runs something else, the kind of bug that wastes an
    afternoon (same rule as the annotator registry).
    """
    if not cls.key:
        raise ValueError(f"{cls.__name__} must define a non-empty `key`")
    if cls.key in _REGISTRY and _REGISTRY[cls.key] is not cls:
        raise ValueError(f"Duplicate trainer key: {cls.key!r}")
    _REGISTRY[cls.key] = cls
    return cls


def available() -> list[dict]:
    """Metadata for every registered trainer — drives the UI's dropdown and
    pre-fills the config form's defaults.

    Read off the CLASS, so nothing is instantiated and none of the heavy
    training frameworks import just to render a menu. This is why the Train page
    still works with the Phase 4 deps uninstalled: the list is simply empty.
    """
    return [
        {
            "key": cls.key,
            "display_name": cls.display_name,
            # Fall back to the display name so a trainer that predates the
            # family/variant fields still groups (as a family of one).
            "family": cls.family or cls.display_name,
            "variant": cls.variant or "default",
            "description": cls.description,
            "approx_vram_gb": cls.approx_vram_gb,
            "export_format": cls.export_format,
            "default_epochs": cls.default_epochs,
            "default_batch_size": cls.default_batch_size,
            "default_image_size": cls.default_image_size,
        }
        for cls in _REGISTRY.values()
        # Legacy trainers stay resolvable by key (existing checkpoints must
        # keep deploying) but are not offered for new runs.
        if cls.listed
    ]


def get_class(key: str) -> type["Trainer"]:
    """Resolve a key to its class, or raise KeyError with a useful message."""
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown trainer {key!r}. Available: {sorted(_REGISTRY)}"
        ) from None
