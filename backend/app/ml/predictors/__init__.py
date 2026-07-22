"""
Predictors: run a TRAINED model on new images.

The fourth pluggable seam, alongside AutoAnnotator, Trainer and DatasetExporter.
A predictor loads a checkpoint a trainer produced and runs inference — for the
inference playground (Phase 5) and for evaluation.

Importing this package must stay cheap (no torch), like the others, so a menu
can render with the heavy deps uninstalled. Concrete predictors live beside
their trainer and import their framework lazily inside methods.
"""

from app.ml.predictors.base import Predictor
from app.ml.predictors import registry

__all__ = ["Predictor", "registry"]
