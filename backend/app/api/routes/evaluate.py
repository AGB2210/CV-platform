"""
Evaluation: score a trained model on a held-out TEST split, and report test mAP.

The test split is the one split training never touches — it trains on train,
watches val to pick the best checkpoint, and leaves test alone. So a test mAP is
the first genuinely independent estimate of how the model generalises, and it is
what this produces, alongside per-class AP.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.config import from_storage_path
from app.database import get_db
from app.models import (
    AnnotationJob,
    DatasetVersion,
    EvaluationJob,
    JobStatus,
    TrainingJob,
)
from app.models.image import Split
from app.schemas.evaluation import EvaluationCreate, EvaluationJobRead
from app.services.dataset_version import load_snapshot
from app.services.evaluation_job import run_evaluation_job

router = APIRouter(tags=["evaluate"])


def _gpu_busy(db: Session, project_id: int) -> bool:
    for model in (TrainingJob, AnnotationJob, EvaluationJob):
        if db.scalar(
            select(model.id).where(
                model.project_id == project_id,
                model.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
            )
        ):
            return True
    return False


def _test_image_count(version: DatasetVersion, split: str) -> int:
    """How many images the version holds in `split` — read off the row where the
    counts are already stored, no snapshot load needed."""
    return {
        Split.TRAIN: version.train_images,
        Split.VAL: version.val_images,
        Split.TEST: version.test_images,
    }.get(split, 0)


@router.post(
    "/projects/{project_id}/evaluate",
    response_model=EvaluationJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_evaluation(
    project_id: int,
    payload: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> EvaluationJob:
    """Queue an evaluation of one model against one dataset version's split.

    202: the work is accepted, not done. The client polls for the score.
    """
    get_project_or_404(project_id, db)

    model_job = db.get(TrainingJob, payload.training_job_id)
    if model_job is None or model_job.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found in this project.")
    if model_job.status != JobStatus.DONE or not model_job.checkpoint_path:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Only a finished run with a saved checkpoint can be evaluated.",
        )
    if from_storage_path(model_job.checkpoint_path) is None or not from_storage_path(
        model_job.checkpoint_path
    ).exists():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "The model's checkpoint is missing on disk."
        )

    version = db.get(DatasetVersion, payload.dataset_version_id)
    if version is None or version.project_id != project_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset version not found.")

    # Fail fast on an empty split — the runner checks too, but a job that flashes
    # up and dies is worse than a plain 400 that never creates a row.
    if _test_image_count(version, payload.split) == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Dataset v{version.version} has no {payload.split} images. Assign "
            "images to the test split on the Dataset page, or upload a test set, "
            "then evaluate. Test data the model never trained on is what makes a "
            "test score honest.",
        )

    _reject_if_busy(db, project_id)

    job = EvaluationJob(
        project_id=project_id,
        training_job_id=payload.training_job_id,
        dataset_version_id=payload.dataset_version_id,
        split=payload.split,
        status=JobStatus.QUEUED,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(run_evaluation_job, job.id)
    return job


def _reject_if_busy(db: Session, project_id: int) -> None:
    if _gpu_busy(db, project_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A training, annotation or evaluation job is already running for this "
            "project. They share the GPU — wait for it to finish.",
        )


@router.get(
    "/projects/{project_id}/evaluations", response_model=list[EvaluationJobRead]
)
def list_evaluations(
    project_id: int, db: Session = Depends(get_db)
) -> list[EvaluationJob]:
    get_project_or_404(project_id, db)
    return list(
        db.scalars(
            select(EvaluationJob)
            .where(EvaluationJob.project_id == project_id)
            .order_by(EvaluationJob.id.desc())
            .limit(20)
        ).all()
    )


@router.get("/evaluation-jobs/{job_id}", response_model=EvaluationJobRead)
def get_evaluation(job_id: int, db: Session = Depends(get_db)) -> EvaluationJob:
    job = db.get(EvaluationJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Evaluation {job_id} not found")
    return job
