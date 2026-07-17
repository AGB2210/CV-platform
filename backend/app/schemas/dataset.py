"""Schemas for the staging -> dataset lifecycle and split management."""

from pydantic import BaseModel, Field


class CommitMode:
    """How a staged batch combines with the existing dataset."""

    APPEND = "append"
    MERGE = "merge"
    REPLACE = "replace"


class DatasetCommit(BaseModel):
    """Request to move approved staging images into the dataset."""

    mode: str = Field(default=CommitMode.APPEND, pattern="^(append|merge|replace)$")

    #: Percentages for auto-assigning splits, used only when assign_splits=True.
    #: They must sum to 1.0 — validated in the route so the error message can
    #: say something useful.
    train_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    val_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    test_pct: float = Field(default=0.0, ge=0.0, le=1.0)

    #: When False, each image keeps whatever split it already has (e.g. from an
    #: import's folder names). When True, the percentages above are applied.
    assign_splits: bool = True


class CommitPreview(BaseModel):
    """What a commit would do, so the UI can say so before it happens.

    Replace is destructive; showing the numbers first is the difference between
    an informed click and a support request.
    """

    staged_total: int
    staged_approved: int
    staged_unapproved: int
    dataset_current: int
    would_add: int
    would_remove: int
    dataset_after: int


class SplitRequest(BaseModel):
    """Assign splits across a project by percentage."""

    train_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    val_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    test_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Only touch images currently in `train`. This is the "you imported a
    #: dataset with no val folder" case: carve a validation set out of train
    #: without disturbing an existing test set.
    only_train: bool = False


class SplitCounts(BaseModel):
    train: int = 0
    val: int = 0
    test: int = 0


class DatasetStats(BaseModel):
    """Everything the Dataset page header needs in one request."""

    staging_total: int
    staging_annotated: int
    staging_approved: int
    dataset_total: int
    splits: SplitCounts
    total_boxes: int
    reviewed_boxes: int
