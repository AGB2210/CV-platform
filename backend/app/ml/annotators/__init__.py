"""
AutoAnnotator implementations.

Importing a module here runs its @register decorator, which is what puts the
model in the registry and therefore in the UI's dropdown. A model nobody imports
is invisible — so this file is the single list of "which annotators exist".

Adding one is two steps: write the class, add the import below. No route, schema
or component changes.
"""

from app.ml.annotators.base import (
    AnnotationRequest,
    AnnotationResult,
    AutoAnnotator,
    Box,
)
from app.ml.annotators.florence2 import Florence2Annotator
from app.ml.annotators.grounding_dino import (
    GroundingDinoAnnotator,
    GroundingDinoBaseAnnotator,
)
from app.ml.annotators.owlv2 import Owlv2Annotator
from app.ml.annotators.yolo_world import YoloWorldAnnotator

__all__ = [
    "AnnotationRequest",
    "AnnotationResult",
    "AutoAnnotator",
    "Box",
    "Florence2Annotator",
    "GroundingDinoAnnotator",
    "GroundingDinoBaseAnnotator",
    "Owlv2Annotator",
    "YoloWorldAnnotator",
]
