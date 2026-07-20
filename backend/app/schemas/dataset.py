"""Schemas for dataset stats and split management.

The staging -> dataset commit schemas (CommitMode, DatasetCommit, CommitPreview)
are gone along with the two-stage model. See routes/dataset.py.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DatasetVersionCreate(BaseModel):
    """Body for "Save dataset" — an optional human note for the version."""

    note: str | None = Field(default=None, max_length=255)


class DatasetVersionRead(BaseModel):
    """One saved dataset version, as the version list renders it."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    #: 1-based per project — what the UI shows, not the row id.
    version: int
    note: str | None
    total_images: int
    train_images: int
    val_images: int
    test_images: int
    total_boxes: int
    num_classes: int
    created_at: datetime


class RestoreResult(BaseModel):
    """What a restore actually did — reported rather than assumed."""

    restored_version: int
    images_restored: int
    boxes_restored: int
    images_removed: int
    #: Images the version referenced whose file is gone from disk. Non-empty
    #: means the restore was partial, and the UI says so.
    missing_files: list[str]
    #: The safety version taken of the pre-restore state, so this is undoable.
    backup_version: int


class SplitRequest(BaseModel):
    """Assign splits across a project by percentage."""

    train_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    val_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    test_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Only touch images currently in `train`. This is the "you imported a
    #: dataset with no val folder" case: carve a validation set out of train
    #: without disturbing an existing test set.
    only_train: bool = False


class BulkSplitRequest(BaseModel):
    """Move specific images to one split."""

    image_ids: list[int]
    split: str


class SplitCounts(BaseModel):
    train: int = 0
    val: int = 0
    test: int = 0


class DatasetStats(BaseModel):
    """Everything the Dataset page header needs in one request."""

    total_images: int
    annotated_images: int
    unannotated_images: int
    splits: SplitCounts
    #: ACCEPTED boxes. Proposals are reported separately, not folded in — "you
    #: have 90 boxes" is a lie if 60 are suggestions nobody has looked at.
    total_boxes: int
    proposed_boxes: int = 0
    proposed_images: int = 0
