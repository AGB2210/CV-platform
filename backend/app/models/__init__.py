"""
ORM models package.

Importing every model here is what makes `Base.metadata.create_all()` work:
SQLAlchemy only knows about a table if its class has been imported. A model that
nothing imports is invisible, and its table silently never gets created.

Re-exporting also lets callers write `from app.models import Project` rather
than reaching into each module.
"""

from app.models.annotation import Annotation
from app.models.annotation_job import AnnotationJob, JobStatus
from app.models.category import Category
from app.models.dataset_version import DatasetVersion
from app.models.image import Image
from app.models.project import Project
from app.models.training_job import JobControl, TrainingJob

__all__ = [
    "Annotation",
    "AnnotationJob",
    "DatasetVersion",
    "JobStatus",
    "Category",
    "Image",
    "JobControl",
    "Project",
    "TrainingJob",
]
