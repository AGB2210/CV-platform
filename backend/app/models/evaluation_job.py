"""
EvaluationJob — score a trained model on a held-out TEST split.

WHY THIS IS A DISTINCT THING FROM THE TRAINING mAP
--------------------------------------------------
Training reports a VAL mAP: measured on the val split, which the trainer watches
during training to pick the best checkpoint. That number is honest but it is not
independent — the model's selection was tuned against it. The TEST split is data
touched by neither training nor selection, so a test mAP is the first genuinely
held-out estimate of how the model does on data it has never seen in any role.

And where val gives one number, evaluation records PER-CLASS AP too: an aggregate
of 0.44 can hide "cars 0.81, people 0.07", and which class is weak is the
actionable fact.

Mirrors TrainingJob/AnnotationJob: it is a job (needs the GPU, takes time, must be
pollable) and shares the JobStatus vocabulary so one StatusBadge renders all
three.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.annotation_job import JobStatus  # noqa: F401 — shared vocabulary


class EvaluationJob(Base):
    __tablename__ = "evaluation_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # The model being scored — a finished TrainingJob with a checkpoint. Plain
    # int, not a FK, for the same reason TrainingJob.dataset_version_id is: the
    # add-column migration stopgap can't ALTER in a constraint, and the runner
    # re-checks the row exists and has a checkpoint before using it.
    training_job_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # The dataset version whose TEST split provides ground truth. A (model ×
    # dataset version) pair IS the claim being made, so evaluating one model
    # against two datasets is two rows, never an overwrite.
    dataset_version_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Which split was scored. "test" is the point, but stored so the record is
    # self-describing and a future "evaluate on val to compare" is a value, not a
    # schema change.
    split: Mapped[str] = mapped_column(String(16), nullable=False, default="test")

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=JobStatus.QUEUED, index=True
    )

    # How many test images were scored — surfaced so a suspiciously-good number
    # on 3 images reads as the small sample it is.
    num_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # The headline COCO numbers. Nullable until the run finishes.
    map_50_95: Mapped[float | None] = mapped_column(default=None)  # the headline mAP
    map_50: Mapped[float | None] = mapped_column(default=None)
    map_75: Mapped[float | None] = mapped_column(default=None)

    # Per-class AP and the small/medium/large breakdown, as JSON — read whole,
    # so a JSON column buys nothing over Text (same call as TrainingJob.metrics).
    per_class_json: Mapped[str | None] = mapped_column(Text, default=None)

    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    project: Mapped["Project"] = relationship()  # noqa: F821

    def __repr__(self) -> str:
        return f"<EvaluationJob id={self.id} model={self.training_job_id} {self.status}>"
