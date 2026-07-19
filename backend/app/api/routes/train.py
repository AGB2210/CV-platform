"""Training: launch runs, poll progress, list history, and check readiness."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.ml.trainers import registry as trainer_registry
from app.models import (
    Annotation,
    AnnotationJob,
    Category,
    Image,
    JobStatus,
    TrainingJob,
)
from app.models.image import Split
from app.schemas.training import TrainerInfo, TrainingJobCreate, TrainingJobRead
from app.services.training_job import run_training_job

router = APIRouter(tags=["train"])


# --- Capability discovery ---------------------------------------------------
# Like /annotators: the frontend asks what trainers exist rather than hardcoding
# them, so registering one updates the dropdown with no frontend change. The list
# is empty until the Phase 4 training deps are installed — the page handles that.


@router.get("/trainers", response_model=list[TrainerInfo])
def list_trainers() -> list[dict]:
    return trainer_registry.available()


# --- Jobs -------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/train",
    response_model=TrainingJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_training(
    project_id: int,
    payload: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJob:
    """Queue a training run.

    202 Accepted, not 201: the work is accepted, not done. The client gets a job
    id and polls — the checkpoint does not exist yet.
    """
    get_project_or_404(project_id, db)

    # Validate the trainer key before creating a row, so a typo (or a request
    # sent while the deps are uninstalled and the list is empty) doesn't create a
    # job that exists only to fail.
    try:
        trainer_registry.get_class(payload.trainer_key)
    except KeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    # One heavy GPU job at a time. Training AND annotation both want the whole
    # card, so refuse to start training while either is in flight for this
    # project — two at once would evict each other's weights or simply OOM.
    _reject_if_gpu_busy(db, project_id)

    # Fail fast on an empty train split. The runner checks this too, but a job
    # that flashes up and immediately fails is worse UX than a plain 400 that
    # never creates a row.
    if _accepted_box_count(db, project_id, Split.TRAIN) == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No accepted boxes in the train split. Accept proposals or draw boxes "
            "and assign images to 'train' on the Dataset page, then train.",
        )

    job = TrainingJob(
        project_id=project_id,
        trainer_key=payload.trainer_key,
        status=JobStatus.QUEUED,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        image_size=payload.image_size,
        learning_rate=payload.learning_rate,
        total_epochs=payload.epochs,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue AFTER the commit: BackgroundTasks fire once the response is sent, and
    # the runner looks the job up by id in its own session — an uncommitted row
    # could miss. This single line is the Celery seam.
    background_tasks.add_task(run_training_job, job.id)
    return job


@router.get("/training-jobs/{job_id}", response_model=TrainingJobRead)
def get_training_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    """Poll one training job. The frontend hits this every ~1s while it runs."""
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Training job {job_id} not found")
    # expire() forces a re-read: the background task writes progress from a
    # DIFFERENT session, so without this the poller serves a stale row and the
    # epoch counter never moves. Same trick as the annotation poller.
    db.expire(job)
    return db.get(TrainingJob, job_id)


@router.get(
    "/projects/{project_id}/training-jobs", response_model=list[TrainingJobRead]
)
def list_training_jobs(
    project_id: int, db: Session = Depends(get_db)
) -> list[TrainingJob]:
    get_project_or_404(project_id, db)
    return list(
        db.scalars(
            select(TrainingJob)
            .where(TrainingJob.project_id == project_id)
            .order_by(TrainingJob.id.desc())
            .limit(20)
        ).all()
    )


@router.get("/projects/{project_id}/train/preview")
def train_preview(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Dataset readiness, so the UI can warn before the click.

    Answers the questions you'd otherwise discover by launching a doomed run:
    are there classes, is there anything in the train split to learn from, and is
    there a val split to measure against?
    """
    get_project_or_404(project_id, db)

    num_classes = db.scalar(
        select(func.count(Category.id)).where(Category.project_id == project_id)
    ) or 0

    counts = {
        split: {
            "images": _image_count(db, project_id, split),
            "boxes": _accepted_box_count(db, project_id, split),
        }
        for split in Split.ALL
    }

    warnings: list[str] = []
    if num_classes == 0:
        warnings.append("No classes defined — add at least one on the Dataset page.")
    if counts[Split.TRAIN]["boxes"] == 0:
        warnings.append(
            "The train split has no accepted boxes — nothing to learn from."
        )
    if counts[Split.VAL]["images"] == 0:
        # Not fatal: the exporter falls back to evaluating on train, but the mAP
        # is then meaningless as a generalisation estimate. Say so up front.
        warnings.append(
            "No validation split — mAP will be measured on the training data. "
            "Assign some images to 'val' on the Dataset page for a real score."
        )

    return {
        "num_classes": num_classes,
        "splits": counts,
        "can_train": num_classes > 0 and counts[Split.TRAIN]["boxes"] > 0,
        "warnings": warnings,
    }


# --- helpers ----------------------------------------------------------------


def _reject_if_gpu_busy(db: Session, project_id: int) -> None:
    """409 if a training or annotation job is already active for this project."""
    active_train = db.scalar(
        select(TrainingJob).where(
            TrainingJob.project_id == project_id,
            TrainingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
    )
    if active_train is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Training job {active_train.id} is already {active_train.status}.",
        )
    active_annotate = db.scalar(
        select(AnnotationJob).where(
            AnnotationJob.project_id == project_id,
            AnnotationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
    )
    if active_annotate is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"An auto-annotation job ({active_annotate.id}) is still "
            f"{active_annotate.status}; it holds the GPU. Wait for it to finish.",
        )


def _accepted_box_count(db: Session, project_id: int, split: str) -> int:
    return db.scalar(
        select(func.count(Annotation.id))
        .join(Image, Image.id == Annotation.image_id)
        .where(
            Image.project_id == project_id,
            Image.split == split,
            Annotation.proposed.is_(False),
        )
    ) or 0


def _image_count(db: Session, project_id: int, split: str) -> int:
    return db.scalar(
        select(func.count(Image.id)).where(
            Image.project_id == project_id, Image.split == split
        )
    ) or 0
