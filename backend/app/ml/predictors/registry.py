"""
Predictor registry — one resident model, keyed by (trainer, checkpoint).

Mirrors the annotator registry (app/ml/registry.py): a predictor is called
repeatedly, so reloading its weights per call would dominate, and there is room
for at most one model on a typical card. So the last-acquired predictor stays
resident and is reused until a different checkpoint is asked for.

THE VRAM RULE, MADE EXPLICIT
----------------------------
A resident predictor is a THIRD GPU consumer, alongside the resident annotator
and a running trainer — and worse than either, because it is pinned by an idle
playground rather than a job that ends. So:

  - acquiring a predictor EVICTS any resident annotator (this module does it, as
    the newest consumer taking responsibility for coexistence);
  - the training and annotation job runners RELEASE the predictor before they
    start (they own the heavy GPU window and must have the whole card).

Get any one of those wrong and the symptom is an OOM that looks like a flaky
training failure rather than the policy bug it is.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.predictors.base import Predictor

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_resident: "Predictor | None" = None
#: (trainer_key, checkpoint_path) the resident predictor was loaded for — the
#: cache key. A new pair means evict-and-reload; the same pair is a cache hit.
_resident_id: tuple[str, str] | None = None


def acquire(trainer_key: str, checkpoint_path: str, class_names: list[str]) -> "Predictor":
    """Get a loaded predictor for this checkpoint, reusing it if already resident.

    Loads via the TRAINER that produced the checkpoint — the only thing that
    knows how to read its weights. Evicts a different resident predictor, and any
    resident annotator, first.
    """
    global _resident, _resident_id

    key = (trainer_key, str(checkpoint_path))
    with _lock:
        if _resident is not None and _resident_id == key and _resident.is_loaded:
            return _resident  # cache hit — the common path in a playground/eval loop

        # Evict a different predictor…
        if _resident is not None:
            logger.info("Evicting predictor %s for %s", _resident_id, key)
            _resident.unload()
            _resident = None
            _resident_id = None

        # …and any resident annotator: on a small card they cannot coexist, and a
        # prediction the user just asked for is what should win the memory.
        from app.ml import registry as annotator_registry

        annotator_registry.release()

        from app.ml.trainers import registry as trainer_registry

        trainer = trainer_registry.get_class(trainer_key)()
        predictor = trainer.load_predictor(checkpoint_path, class_names)
        predictor.load()
        _resident = predictor
        _resident_id = key
        return predictor


def release() -> None:
    """Unload the resident predictor, if any. Called by the job runners before
    they take the card, and safe to call when nothing is resident."""
    global _resident, _resident_id
    with _lock:
        if _resident is not None:
            _resident.unload()
            _resident = None
            _resident_id = None


def resident_id() -> tuple[str, str] | None:
    """Which (trainer, checkpoint) is resident, if any. For the status UI."""
    return _resident_id
