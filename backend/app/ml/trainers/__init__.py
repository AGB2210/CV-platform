"""
Trainer implementations.

Importing a module here runs its @register decorator, which is what puts the
trainer in the registry and therefore in the UI's dropdown. A trainer nobody
imports is invisible — so this file is the single list of "which trainers exist".

Adding one is two steps: write the class, add the import below. No route, schema
or component changes.

Phase 4b registers the first concrete trainer, YOLO. Its module imports
ultralytics lazily (inside train()), so this import stays free — the trainer
appears in the registry, but the heavy deps only load when a run actually starts.
"""

from app.ml.trainers.base import (
    EpochMetrics,
    TrainConfig,
    TrainResult,
    Trainer,
)
from app.ml.trainers.yolo import YoloTrainer

__all__ = [
    "EpochMetrics",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "YoloTrainer",
]
