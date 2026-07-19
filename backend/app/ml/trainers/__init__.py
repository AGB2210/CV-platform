"""
Trainer implementations.

Importing a module here runs its @register decorator, which is what puts the
trainer in the registry and therefore in the UI's dropdown. A trainer nobody
imports is invisible — so this file is the single list of "which trainers exist".

Adding one is two steps: write the class, add the import below. No route, schema
or component changes.

Phase 4a ships the pipeline (interface, registry, job, routes, page) with NO
concrete trainer registered yet — the dropdown is intentionally empty until the
heavy training deps are installed. Phase 4b adds the first one:

    from app.ml.trainers.yolo import YoloTrainer   # noqa: F401
"""

from app.ml.trainers.base import (
    EpochMetrics,
    TrainConfig,
    TrainResult,
    Trainer,
)

__all__ = [
    "EpochMetrics",
    "TrainConfig",
    "TrainResult",
    "Trainer",
]
