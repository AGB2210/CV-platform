"""Schemas for dataset stats and split management.

The staging -> dataset commit schemas (CommitMode, DatasetCommit, CommitPreview)
are gone along with the two-stage model. See routes/dataset.py.
"""

from pydantic import BaseModel, Field


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
