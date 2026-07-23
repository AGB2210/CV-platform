"""Pydantic schemas for training jobs and trainers."""

from __future__ import annotations

import json
from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.timestamps import UtcDatetime


class TrainerInfo(BaseModel):
    """One entry in the UI's trainer dropdown, plus the defaults it pre-fills
    the config form with."""

    key: str
    display_name: str
    #: Architecture family ("YOLO12", "RT-DETR") and the size within it
    #: ("nano", "L") — the UI's picker groups by these two axes.
    family: str
    variant: str
    description: str
    approx_vram_gb: float
    export_format: str
    default_epochs: int
    default_batch_size: int
    default_image_size: int


class TrainingJobCreate(BaseModel):
    """Request body to launch a training run.

    Bounds exist to catch fat-finger inputs before they cost an out-of-memory
    error or a days-long run. They are generous ceilings, not recommendations —
    the trainer's own defaults are the sane starting point, surfaced per-backend
    via TrainerInfo, and what actually fits depends on the GPU this is running
    on rather than on any figure baked in here.
    """

    trainer_key: str

    # 1..1000 epochs. The upper bound is a guard against a typo'd 5000 that would
    # run for days unnoticed, not a claim that 1000 is sensible here.
    epochs: int = Field(default=50, ge=1, le=1000)
    # Batch size is the first thing to cut on a small card. 1..128; the trainer
    # default is deliberately tiny.
    batch_size: int = Field(default=8, ge=1, le=128)
    # Square side. Multiple-of-32 is a YOLO requirement the trainer enforces;
    # here we just bound it so a GPU isn't asked to hold a 4096px batch.
    image_size: int = Field(default=640, ge=64, le=2048)
    # None = the framework's own schedule, which is usually right. A real value
    # only when the user deliberately overrides it.
    learning_rate: float | None = Field(default=None, gt=0.0, le=1.0)

    # Continue/finetune from a previous run's checkpoint instead of the
    # pretrained base. None = fresh start. Validated at the route (same project,
    # completed, has a checkpoint, matching class count).
    init_from_job_id: int | None = None

    # Finetune from an UPLOADED checkpoint (imported weights) instead. Mutually
    # exclusive with init_from_job_id — the route rejects both at once, because
    # a run has exactly one starting point.
    init_weights_id: int | None = None

    # Which SAVED dataset version to train. None = the latest saved version.
    # Training always runs against a saved version, never the live rows, so a
    # run's results stay attributable to a specific dataset.
    dataset_version_id: int | None = None


class TrainingJobRename(BaseModel):
    """Rename a model version. Blank clears it, reverting to "v{n}"."""

    name: str | None = Field(default=None, max_length=120)


class BulkDeleteJobs(BaseModel):
    """Delete several model versions — also how "delete all" is sent."""

    job_ids: list[int]


class DeleteJobsResult(BaseModel):
    deleted: int
    not_found: list[int]
    #: job id -> why it was refused (a run still in flight can't be deleted).
    skipped: dict[int, str] = {}


class EpochPoint(BaseModel):
    """One epoch on the loss/mAP curve."""

    epoch: int
    train_loss: float | None = None
    val_map: float | None = None
    val_map50: float | None = None


class TrainingJobRead(BaseModel):
    """Training job state — polled by the frontend while a run is in flight."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    trainer_key: str
    #: 1-based version number within this project + trainer (what the UI shows).
    version: int
    #: User-given name; None means it displays as "v{version}".
    name: str | None
    status: str
    #: Why a queued job hasn't started — the live "waiting for GPU" reason
    #: from the admission loop. NULL once running.
    status_detail: str | None = None

    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float | None
    train_images: int
    val_images: int
    num_classes: int
    #: Set when this run continued another run's checkpoint (finetune).
    init_from_job_id: int | None
    #: Set when this run started from an uploaded checkpoint instead.
    init_weights_id: int | None
    #: The saved dataset version this run trained on.
    dataset_version_id: int | None
    #: "stop" | "cancel" once requested, so the UI can show it's winding down.
    control: str | None
    #: True when the run ended because the user stopped it short.
    stopped_early: bool

    current_epoch: int
    total_epochs: int
    train_loss: float | None
    val_map: float | None
    best_map: float | None

    checkpoint_path: str | None
    error: str | None

    created_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None

    # Loaded from the ORM but not serialised raw — exposed parsed via `metrics`
    # below, so the frontend gets an array to plot rather than a JSON string to
    # parse itself.
    metrics_json: str | None = Field(default=None, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress_pct(self) -> float:
        """Derived, not stored — a stored percentage is just a chance to drift
        out of sync with the counters it's computed from."""
        if not self.total_epochs:
            return 0.0
        return round(100.0 * self.current_epoch / self.total_epochs, 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def metrics(self) -> list[EpochPoint]:
        """Per-epoch history, parsed from metrics_json for the curve. Empty
        (never null) when a run hasn't reported an epoch yet, so the frontend can
        map over it unconditionally."""
        if not self.metrics_json:
            return []
        try:
            return [EpochPoint(**row) for row in json.loads(self.metrics_json)]
        except (json.JSONDecodeError, TypeError, ValueError):
            # A malformed history must not 500 the poll the whole page depends
            # on — an empty curve is a survivable degradation.
            return []


class ImportedWeightsRead(BaseModel):
    """An uploaded checkpoint, listable in the trainer's "Initialize from"."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    filename: str
    size_bytes: int
    created_at: UtcDatetime
