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
    DatasetVersion,
    Image,
    JobStatus,
    TrainingJob,
)
from app.models.image import Split
from app.schemas.training import TrainerInfo, TrainingJobCreate, TrainingJobRead
from app.services.training_job import run_training_job

router = APIRouter(tags=["train"])

# Below this many training images, fine-tuning a detector effectively can't work:
# batch-sized steps per epoch mean the freshly-initialised class head never gets
# enough gradient updates, so mAP sits at ~0 no matter how long you run. It's not
# a hard block — you can still launch a run — but saying so up front turns a
# baffling "training looks broken" into an understood "add more images". Dozens
# per class is a realistic floor; 10 total is the point below which it's futile.
MIN_USEFUL_TRAIN_IMAGES = 10


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

    # Training runs against a SAVED dataset version, never the live rows — that's
    # what makes a run reproducible and its provenance truthful. Resolve which
    # one (explicit, or the latest save) and refuse if the dataset was never
    # saved.
    dataset_version = _resolve_dataset_version(db, project_id, payload.dataset_version_id)

    # Fail fast on an empty train split. The runner checks this too, but a job
    # that flashes up and immediately fails is worse UX than a plain 400 that
    # never creates a row.
    if dataset_version.train_boxes == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Dataset v{dataset_version.version} has no boxes in its train split. "
            "Annotate some images, save the dataset, and train the new version.",
        )

    # Finetuning from a previous run: validate the source is usable and its class
    # set still matches, so we fail with a clear 400 now rather than a cryptic
    # framework error 30 seconds into the run.
    if payload.init_from_job_id is not None:
        _validate_finetune_source(db, project_id, payload.init_from_job_id)

    # Version number scoped to THIS project + model, 1-based. Counts every prior
    # run of the same trainer here (any status) — a version is a training attempt,
    # so failures occupy a number too and the sequence never silently reuses one.
    version = 1 + (
        db.scalar(
            select(func.count(TrainingJob.id)).where(
                TrainingJob.project_id == project_id,
                TrainingJob.trainer_key == payload.trainer_key,
            )
        )
        or 0
    )

    job = TrainingJob(
        project_id=project_id,
        trainer_key=payload.trainer_key,
        version=version,
        status=JobStatus.QUEUED,
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        image_size=payload.image_size,
        learning_rate=payload.learning_rate,
        total_epochs=payload.epochs,
        init_from_job_id=payload.init_from_job_id,
        dataset_version_id=dataset_version.id,
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

    # Training trains a SAVED version, so readiness starts with "has it been
    # saved at all?".
    latest = db.scalar(
        select(DatasetVersion)
        .where(DatasetVersion.project_id == project_id)
        .order_by(DatasetVersion.version.desc())
    )

    warnings: list[str] = []
    if latest is None:
        warnings.append(
            "The dataset has never been saved. Click “Save dataset” to create v1 — "
            "training runs against a saved version."
        )
    if num_classes == 0:
        warnings.append("No classes defined — add at least one on the Dataset page.")
    if counts[Split.TRAIN]["boxes"] == 0:
        warnings.append(
            "The train split has no accepted boxes — nothing to learn from."
        )
    else:
        # Only when there IS something to train on: flagging "too few images" on
        # a project with zero boxes would just be noise on top of the real
        # problem above.
        train_imgs = counts[Split.TRAIN]["images"]
        if train_imgs < MIN_USEFUL_TRAIN_IMAGES:
            warnings.append(
                f"Only {train_imgs} training image{'s' if train_imgs != 1 else ''} — "
                "far too few to fine-tune a detector. Expect a near-zero mAP until "
                "you add more (dozens per class is a realistic floor)."
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
        # A saved version is now part of readiness — without one there is
        # nothing reproducible to train.
        "has_saved_version": latest is not None,
        "latest_version": latest.version if latest else None,
        "latest_version_id": latest.id if latest else None,
        "can_train": (
            latest is not None
            and latest.train_boxes > 0
            and latest.num_classes > 0
        ),
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


def _resolve_dataset_version(db: Session, project_id: int, version_id: int | None):
    """The saved dataset version this run will train, or 400 with what to do.

    Defaults to the latest save, which is what "just train it" means once you've
    clicked Save dataset. An explicit id lets you train an older version.
    """
    if version_id is not None:
        version = db.get(DatasetVersion, version_id)
        if version is None or version.project_id != project_id:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Dataset version {version_id} was not found in this project.",
            )
        return version

    latest = db.scalar(
        select(DatasetVersion)
        .where(DatasetVersion.project_id == project_id)
        .order_by(DatasetVersion.version.desc())
    )
    if latest is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Save the dataset before training. Training runs against a saved "
            "dataset version so a run's results stay reproducible.",
        )
    return latest


def _validate_finetune_source(db: Session, project_id: int, source_id: int) -> None:
    """400 unless `source_id` is a finished run in this project with a checkpoint
    and a class set matching the project's current classes."""
    source = db.get(TrainingJob, source_id)
    if source is None or source.project_id != project_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Run {source_id} was not found in this project.",
        )
    if source.status != JobStatus.DONE or not source.checkpoint_path:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Run {source_id} has no finished checkpoint to continue from.",
        )
    current_classes = (
        db.scalar(
            select(func.count(Category.id)).where(Category.project_id == project_id)
        )
        or 0
    )
    # source.num_classes is 0 on runs from before it was recorded — skip the
    # check there rather than block continuing a perfectly good old checkpoint.
    if source.num_classes and source.num_classes != current_classes:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Run {source_id} was trained on {source.num_classes} class"
            f"{'es' if source.num_classes != 1 else ''}, but the project now has "
            f"{current_classes}. Finetuning needs a matching class set — train a "
            "fresh model instead.",
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
