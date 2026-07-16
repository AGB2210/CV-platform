"""
AnnotationJob ORM model — one auto-annotation run over a set of images.

WHY A TABLE RATHER THAN AN IN-MEMORY DICT
-----------------------------------------
The job itself runs in-process via FastAPI's BackgroundTasks, so an in-memory
dict of progress would "work". It's still wrong:

  - The frontend polls. Progress must survive `uvicorn --reload` restarting the
    process every time you save a file, or every dev-mode edit orphans the UI.
  - Failures need a record. "Why did last night's run stop at image 340?" is
    unanswerable if the error died with the process.
  - It's queryable. Job history per project is a SELECT, not bookkeeping.

WHERE THIS BREAKS AT SCALE
--------------------------
BackgroundTasks run inside the web process. That means: the job dies if the
server restarts, there's no retry, no cross-process queue, and a long job holds
a worker. For a single-user local tool that's the right trade — a Celery/Redis
setup here would be more infrastructure than application.

The seam is deliberate: `services/annotation_job.py::run_annotation_job` takes
(job_id, db_session_factory) and touches no request state. Moving to Celery/RQ
means decorating that function and swapping the `background_tasks.add_task` call
in the route. Nothing else changes. That's the whole reason the ML code never
imports FastAPI.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class JobStatus:
    """Job lifecycle states.

    A plain class of string constants, not a DB-level Enum — same reasoning as
    TaskType (see app/enums.py): SQLite can't ALTER a CHECK constraint, so
    adding a state later would mean rebuilding the table. These map 1:1 to the
    frontend's StatusBadge vocabulary.
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class AnnotationJob(Base):
    __tablename__ = "annotation_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Which annotator ran — a registry key like "grounding_dino". Stored as a
    # string rather than a FK because the registry lives in code, not the DB.
    model_key: Mapped[str] = mapped_column(String(64), nullable=False)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=JobStatus.QUEUED, index=True
    )

    # Progress counters the frontend polls. processed/total drives the bar;
    # boxes_created tells you whether the run actually found anything, which is
    # the difference between "done" and "done, and useless".
    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    boxes_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # The parameters used, kept so a run is reproducible and so the UI can show
    # "this job used threshold 0.35" months later.
    box_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.30)
    text_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    # JSON blob of {class_name: prompt}. Stored as Text and serialised by hand
    # rather than using a JSON column: SQLite's JSON support is fine, but we
    # only ever read this whole-value, so a column type buys nothing.
    prompts_json: Mapped[str | None] = mapped_column(Text, default=None)

    # Populated on failure. Text, not String(n) — tracebacks are long and
    # truncating the one thing that explains a failure is a cruel default.
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    # Separate from created_at: the gap between them is queue wait, and the gap
    # between started and finished is actual compute. Conflating them makes
    # "why was that slow" unanswerable.
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    project: Mapped["Project"] = relationship()  # noqa: F821

    @property
    def progress_pct(self) -> float:
        if not self.total_images:
            return 0.0
        return round(100.0 * self.processed_images / self.total_images, 1)

    def __repr__(self) -> str:
        return f"<AnnotationJob id={self.id} {self.model_key} {self.status}>"
