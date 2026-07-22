"""
TrainingJob ORM model — one training run of one model over a project's dataset.

The training sibling of AnnotationJob, and a table for the same reasons (see that
model's docstring): the frontend polls, so progress must survive a
`uvicorn --reload`; failures need a durable record with their traceback; and job
history is a SELECT, not in-memory bookkeeping.

The seam is identical too: `services/training_job.py::run_training_job` takes a
job id and makes its own session, touching no request state — so moving off
FastAPI BackgroundTasks to Celery/RQ is a one-line change at the route and
nothing else. That is again why the ML code never imports FastAPI.

WHAT DIFFERS FROM AnnotationJob
-------------------------------
Annotation progress is measured in IMAGES (processed / total). Training progress
is measured in EPOCHS, and the interesting output is metrics over time — loss and
mAP — not a box count. So the counters and result columns are different, but the
lifecycle (queued -> running -> done/failed) and the JobStatus vocabulary are
shared, which keeps the frontend's StatusBadge working unchanged.
"""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class JobControl:
    """What the user has asked an in-flight run to do.

    Plain string constants, not a DB Enum — same reason as JobStatus: SQLite
    can't ALTER a CHECK constraint, so adding a third instruction later would
    mean rebuilding the table.
    """

    #: Finish the epoch in flight, then stop and KEEP the model.
    STOP = "stop"
    #: Finish the epoch in flight, then stop and THROW THE RUN AWAY.
    CANCEL = "cancel"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which trainer ran — a registry key like "yolo". A string, not a FK,
    # because the registry lives in code, not the DB (same as AnnotationJob).
    trainer_key: Mapped[str] = mapped_column(String(64), nullable=False)

    # Per-(project, trainer) version number, 1-based — "which iteration of THIS
    # model on THIS project is this". Assigned at creation. Distinct from `id`,
    # which is a global primary key shared across every project and model: this
    # is what the UI shows, so the sequence starts at 1 per model and reads like
    # versions rather than exposing internal row ids.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Optional user-given name. NULL means the run shows as "v{version}".
    #: Uniqueness is enforced against every other run's LABEL for the same
    #: project AND trainer, so no two versions of a model read the same.
    name: Mapped[str | None] = mapped_column(String(120), default=None)

    # Reuses AnnotationJob's JobStatus vocabulary (queued/running/done/failed).
    # Not imported as a type here — it's a bag of string constants, and the
    # column just stores the string — but the values are deliberately identical
    # so one StatusBadge component renders both kinds of job.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued", index=True
    )

    # --- Hyperparameters, recorded so a run is reproducible ------------------
    # "which settings produced this checkpoint" is unanswerable months later
    # unless the settings are stored next to the result.
    epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    image_size: Mapped[int] = mapped_column(Integer, nullable=False, default=640)
    # NULL means "the framework's own default schedule" — see TrainConfig. A real
    # column, not omitted, because a reproducible record must distinguish "we
    # chose the default" from "we chose 0.01".
    learning_rate: Mapped[float | None] = mapped_column(Float, default=None)

    # Dataset size the run actually trained on, snapshotted at export time.
    # Stored rather than recomputed because the split can change afterwards and
    # the job record should reflect what THIS run saw.
    train_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Class count this run trained on, snapshotted. Used to reject finetuning a
    # checkpoint whose class set no longer matches the project's current classes.
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Which SAVED dataset version this run trained on. The run exports that
    # version's snapshot, not the live rows — so "trained on dataset v3" is a
    # fact that stays true even after the dataset changes underneath it. Nullable
    # only for runs recorded before dataset versions existed.
    dataset_version_id: Mapped[int | None] = mapped_column(Integer, default=None)

    # When set, this run was initialised from ANOTHER run's checkpoint (continue
    # / finetune) rather than the pretrained base — provenance: "run 7 continued
    # run 4". A plain nullable int, not a FK: the _add_missing_columns stopgap
    # can't ALTER in a constraint, and a dangling id is harmless — the runner
    # re-checks the source exists and has a checkpoint before using it.
    init_from_job_id: Mapped[int | None] = mapped_column(Integer, default=None)

    # --- Live progress the frontend polls -----------------------------------
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Latest and best validation mAP (@.50:.95), plus the most recent train loss.
    # `best_map` is tracked separately from `val_map` because mAP is noisy epoch
    # to epoch — the last epoch is often not the best, and it's the best
    # checkpoint we keep and report.
    train_loss: Mapped[float | None] = mapped_column(Float, default=None)
    val_map: Mapped[float | None] = mapped_column(Float, default=None)
    best_map: Mapped[float | None] = mapped_column(Float, default=None)

    # Per-epoch history as a JSON array of {epoch, train_loss, val_map, ...},
    # for a loss/mAP curve. Text + hand-serialised for the same reason as
    # AnnotationJob.prompts_json: we only ever read it whole, so a JSON column
    # type buys nothing over a string.
    metrics_json: Mapped[str | None] = mapped_column(Text, default=None)

    # Filesystem path to the best checkpoint. On disk, not in the DB, exactly
    # like image bytes — weights are large binaries and belong on the filesystem
    # with only their path recorded here. Consumed by Phase 5 (evaluate/deploy).
    checkpoint_path: Mapped[str | None] = mapped_column(String(512), default=None)

    # --- User control of a run in flight ------------------------------------
    #
    # NULL | "stop" | "cancel", set by the route and read by the runner between
    # epochs. A DB column rather than an in-process event because the runner and
    # the request that interrupts it are different sessions (and, after a
    # reload, potentially different processes) — a flag in memory would be
    # invisible to exactly the code that needs it.
    #
    # Both mean "stop after the epoch in flight"; they differ in what happens
    # next. "stop" keeps the model trained so far; "cancel" throws it away.
    control: Mapped[str | None] = mapped_column(String(16), default=None)

    #: Why a QUEUED job hasn't started yet — e.g. "Waiting for GPU: 0.4 GB
    #: free, needs ~3 GB". Written by the admission wait loop with LIVE
    #: numbers, cleared the moment the run starts, so the UI never invents a
    #: hardcoded explanation. NULL when there's nothing to explain.
    status_detail: Mapped[str | None] = mapped_column(String(255), default=None)

    #: True when the run finished because the user asked it to stop early, so
    #: the UI can say "stopped at epoch 13 of 50" rather than implying it ran
    #: the full schedule.
    stopped_early: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Populated on failure. Text, not String(n) — tracebacks are long, and
    # truncating the one thing that explains a failure is a cruel default.
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    # Split for the same reason as AnnotationJob: created->started is queue wait,
    # started->finished is actual compute, and conflating them makes "why was
    # that slow" unanswerable.
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    project: Mapped["Project"] = relationship()  # noqa: F821

    @property
    def progress_pct(self) -> float:
        if not self.total_epochs:
            return 0.0
        return round(100.0 * self.current_epoch / self.total_epochs, 1)

    def __repr__(self) -> str:
        return f"<TrainingJob id={self.id} {self.trainer_key} {self.status}>"
