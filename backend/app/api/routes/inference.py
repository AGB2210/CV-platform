"""
Inference playground: run a trained model on an uploaded image and see the boxes.

The visible payoff of Phase 5 — "upload a picture, watch your model find things".

TWO RULES THAT KEEP THIS HONEST
-------------------------------
1. **Predictions are NOT proposals.** Nothing here is written to the database.
   The playground is for looking; persisting a prediction would resurrect the
   staging concept the project deliberately removed (see the proposals model).
2. **The model runs against a SAVED run's checkpoint, with the class list that
   run trained on** — resolved from its dataset version's snapshot, in the same
   order the exporter assigned indices, so a box's label is the project's name
   and not whatever ultralytics happened to store.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.config import from_storage_path
from app.database import get_db
from app.ml.predictors import registry as predictor_registry
from app.models import AnnotationJob, JobStatus, TrainingJob
from app.services.version_naming import label_for

router = APIRouter(tags=["inference"])


class DeployableModel(BaseModel):
    """A finished run whose checkpoint can be run. Drives the Deploy dropdown."""

    job_id: int
    trainer_key: str
    version: int
    label: str
    best_map: float | None


class PredictionBox(BaseModel):
    label: str
    confidence: float
    # COCO-style absolute pixels (top-left + size), matching how annotations are
    # stored and how the canvas draws — so the read-only overlay needs no new
    # coordinate convention.
    x: float
    y: float
    width: float
    height: float


class PredictionResult(BaseModel):
    image_width: int
    image_height: int
    boxes: list[PredictionBox]


@router.get("/projects/{project_id}/models", response_model=list[DeployableModel])
def list_models(project_id: int, db: Session = Depends(get_db)) -> list[DeployableModel]:
    """Finished runs with a checkpoint on disk — the models you can deploy."""
    get_project_or_404(project_id, db)
    jobs = db.scalars(
        select(TrainingJob)
        .where(
            TrainingJob.project_id == project_id,
            TrainingJob.status == JobStatus.DONE,
            TrainingJob.checkpoint_path.is_not(None),
        )
        .order_by(TrainingJob.id.desc())
    ).all()

    out: list[DeployableModel] = []
    for job in jobs:
        path = from_storage_path(job.checkpoint_path)
        if path is None or not path.exists():
            continue  # a run whose weights were cleaned up is not deployable
        out.append(
            DeployableModel(
                job_id=job.id,
                trainer_key=job.trainer_key,
                version=job.version,
                label=label_for(job.name, job.version),
                best_map=job.best_map,
            )
        )
    return out


@router.get("/models/{job_id}/weights")
def download_weights(job_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Download a finished run's checkpoint (.pt) — the model, portable.

    The file is streamed as-is from storage: what ultralytics saved as best.pt
    is exactly what you get, loadable anywhere with `YOLO("file.pt")`. The
    download name carries the version label and trainer key so a folder of
    exported weights stays tellable-apart.
    """
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model {job_id} not found")
    if job.status != JobStatus.DONE or not job.checkpoint_path:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "That run has no usable checkpoint — only a finished run has weights.",
        )
    checkpoint = from_storage_path(job.checkpoint_path)
    if checkpoint is None or not checkpoint.exists():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The checkpoint file for that run is missing on disk.",
        )

    label = label_for(job.name, job.version)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    return FileResponse(
        checkpoint,
        media_type="application/octet-stream",
        filename=f"{safe}_{job.trainer_key}.pt",
    )


def _gpu_busy(db: Session, project_id: int) -> bool:
    """Is a heavy GPU job running for this project? Inference must not overlap it."""
    for model in (TrainingJob, AnnotationJob):
        running = db.scalar(
            select(model.id).where(
                model.project_id == project_id,
                model.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
            )
        )
        if running is not None:
            return True
    return False


@router.post("/models/{job_id}/predict", response_model=PredictionResult)
async def predict(
    job_id: int,
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    db: Session = Depends(get_db),
) -> PredictionResult:
    """Run the model on one uploaded image. Boxes back; nothing stored."""
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Model {job_id} not found")
    if job.status != JobStatus.DONE or not job.checkpoint_path:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "That run has no usable checkpoint — only a finished run can be deployed.",
        )

    # Re-resolve the checkpoint on disk (it can be cleaned up between the row
    # being written and this request), and the class list the run trained on.
    checkpoint = from_storage_path(job.checkpoint_path)
    if checkpoint is None or not checkpoint.exists():
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "The checkpoint file for that run is missing on disk.",
        )
    class_names = _class_names_for(db, job)

    # One heavy GPU model at a time: refuse if a job is already using the card.
    if _gpu_busy(db, job.project_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "A training or annotation job is running for this project. "
            "Inference shares the GPU — wait for it to finish, then try again.",
        )

    data = await file.read()
    # The predictor takes a path; write to a temp file the model can open, and
    # remove it afterwards whatever happens.
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        predictor = predictor_registry.acquire(
            job.trainer_key, str(checkpoint), class_names
        )
        boxes = predictor.predict(tmp.name, conf_threshold=conf_threshold)
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    # Image dimensions for the overlay's viewBox, read from the bytes we already
    # have rather than trusting the model's report.
    from io import BytesIO

    from PIL import Image as PILImage

    with PILImage.open(BytesIO(data)) as im:
        width, height = im.size

    return PredictionResult(
        image_width=width,
        image_height=height,
        boxes=[
            PredictionBox(
                label=b.label,
                confidence=round(b.confidence, 4),
                x=round(b.x1, 2),
                y=round(b.y1, 2),
                width=round(b.width, 2),
                height=round(b.height, 2),
            )
            for b in boxes
        ],
    )


def _class_names_for(db: Session, job: TrainingJob) -> list[str]:
    """The class list the run trained on, in index order.

    From the run's dataset-version snapshot — the same source, in the same order,
    the exporter used to assign class indices, so label N here means the same
    thing it meant to the model. Falls back to the project's current classes only
    if the version is gone (a degraded but labelled result beats numbers).
    """
    from app.models import Category, DatasetVersion
    from app.services.dataset_version import load_snapshot

    if job.dataset_version_id is not None:
        version = db.get(DatasetVersion, job.dataset_version_id)
        if version is not None:
            try:
                snapshot = load_snapshot(version)
                return [c.name for c in snapshot.categories]
            except Exception:  # noqa: BLE001 — fall back rather than 500 the playground
                pass
    return [
        c.name
        for c in db.scalars(
            select(Category).where(Category.project_id == job.project_id).order_by(Category.id)
        ).all()
    ]
