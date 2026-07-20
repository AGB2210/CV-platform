"""Pydantic schemas for training jobs and trainers."""

from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field


class TrainerInfo(BaseModel):
    """One entry in the UI's trainer dropdown, plus the defaults it pre-fills
    the config form with."""

    key: str
    display_name: str
    description: str
    approx_vram_gb: float
    export_format: str
    default_epochs: int
    default_batch_size: int
    default_image_size: int


class TrainingJobCreate(BaseModel):
    """Request body to launch a training run.

    Bounds exist to catch fat-finger inputs before they cost an OOM or a
    days-long run: this is a 4 GB laptop GPU, not a cluster. They are generous
    ceilings, not recommendations — the trainer's own defaults are the sane
    starting point, surfaced via TrainerInfo.
    """

    trainer_key: str

    # 1..1000 epochs. The upper bound is a guard against a typo'd 5000 that would
    # run for days unnoticed, not a claim that 1000 is sensible here.
    epochs: int = Field(default=50, ge=1, le=1000)
    # Batch size is the first thing to cut on a small card. 1..128; the trainer
    # default is deliberately tiny.
    batch_size: int = Field(default=8, ge=1, le=128)
    # Square side. Multiple-of-32 is a YOLO requirement the trainer enforces;
    # here we just bound it so 4 GB isn't asked to hold a 4096px batch.
    image_size: int = Field(default=640, ge=64, le=2048)
    # None = the framework's own schedule, which is usually right. A real value
    # only when the user deliberately overrides it.
    learning_rate: float | None = Field(default=None, gt=0.0, le=1.0)

    # Continue/finetune from a previous run's checkpoint instead of the
    # pretrained base. None = fresh start. Validated at the route (same project,
    # completed, has a checkpoint, matching class count).
    init_from_job_id: int | None = None


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
    status: str

    epochs: int
    batch_size: int
    image_size: int
    learning_rate: float | None
    train_images: int
    val_images: int
    num_classes: int
    #: Set when this run continued another run's checkpoint (finetune).
    init_from_job_id: int | None

    current_epoch: int
    total_epochs: int
    train_loss: float | None
    val_map: float | None
    best_map: float | None

    checkpoint_path: str | None
    error: str | None

    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

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
