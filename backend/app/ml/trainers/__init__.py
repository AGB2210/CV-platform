"""
Trainer implementations.

Importing a module here runs its @register decorator, which is what puts the
trainer in the registry and therefore in the UI's dropdown. A trainer nobody
imports is invisible — so this file is the single list of "which trainers exist".

Adding one is two steps: write the class, add the import below. No route, schema
or component changes.

The ultralytics module registers the whole roster (YOLO12 and YOLO26
nano..xlarge, RT-DETR L/X, plus the delisted legacy YOLO11 keys) via its own
@register decorators; rf_detr adds the RF-DETR family. It imports ultralytics lazily
(inside train()), so this import stays free — the trainers appear in the
registry, but the heavy deps only load when a run actually starts.
"""

from app.ml.trainers.base import (
    EpochMetrics,
    TrainConfig,
    TrainResult,
    Trainer,
)
from app.ml.trainers.rf_detr import RfDetrTrainer
from app.ml.trainers.yolo import UltralyticsTrainer

__all__ = [
    "EpochMetrics",
    "RfDetrTrainer",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "UltralyticsTrainer",
]
