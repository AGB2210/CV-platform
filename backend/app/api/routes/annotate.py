"""Auto-annotation, annotation reads, and dataset export."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.ml import registry
from app.ml.device import device_info
from app.models import Annotation, AnnotationJob, Image, JobStatus
from app.schemas.annotation import (
    AnnotationJobCreate,
    AnnotationJobRead,
    AnnotationRead,
    AnnotatorInfo,
    DeviceInfo,
    ExportFormatInfo,
)
from app.services import exporters
from app.services.annotation_job import run_annotation_job

router = APIRouter(tags=["annotate"])


# --- Capability discovery ---------------------------------------------------
# The frontend asks what exists rather than hardcoding a model list. That's what
# makes the registry pay off: add an annotator, and the dropdown updates with no
# frontend change at all.


@router.get("/annotators", response_model=list[AnnotatorInfo])
def list_annotators() -> list[dict]:
    return registry.available()


@router.get("/export-formats", response_model=list[ExportFormatInfo])
def list_export_formats() -> list[dict]:
    return exporters.available()


@router.get("/device", response_model=DeviceInfo)
def get_device() -> dict:
    """Report GPU/CPU so the UI can set expectations before a long run."""
    return device_info()


# --- Jobs -------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/annotate",
    response_model=AnnotationJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_annotation(
    project_id: int,
    payload: AnnotationJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnnotationJob:
    """Queue an auto-annotation run.

    202 Accepted, not 201 Created: the work has been *accepted*, not completed.
    The client gets a job id and polls. Returning 200 here would imply the
    annotations exist, which they emphatically do not yet.
    """
    get_project_or_404(project_id, db)

    # Validate the model key BEFORE creating the job row. Otherwise a typo
    # produces a job that exists only to immediately fail, cluttering history.
    try:
        registry.get_class(payload.model_key)
    except KeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    # Refuse to queue a second job while one is in flight. With 4 GB of VRAM two
    # concurrent runs would evict each other's model on every image — thrashing
    # weights in and out and running slower than either alone, if they didn't
    # simply OOM.
    active = db.scalar(
        select(AnnotationJob).where(
            AnnotationJob.project_id == project_id,
            AnnotationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
        )
    )
    if active is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Job {active.id} is already {active.status} for this project",
        )

    image_count = len(
        db.scalars(select(Image.id).where(Image.project_id == project_id)).all()
    )
    if image_count == 0:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "This project has no images to annotate"
        )

    job = AnnotationJob(
        project_id=project_id,
        model_key=payload.model_key,
        status=JobStatus.QUEUED,
        total_images=image_count,
        box_threshold=payload.box_threshold,
        text_threshold=payload.text_threshold,
        prompts_json=json.dumps(payload.prompts) if payload.prompts else None,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue AFTER the commit. BackgroundTasks fire once the response is sent,
    # but the runner opens its own session and looks the job up by id — if the
    # row weren't committed first, that lookup could miss.
    #
    # This single line is the Celery seam: `run_annotation_job.delay(job.id)`
    # replaces it, and nothing else in the codebase changes.
    background_tasks.add_task(run_annotation_job, job.id)
    return job


@router.get("/jobs/{job_id}", response_model=AnnotationJobRead)
def get_job(job_id: int, db: Session = Depends(get_db)) -> AnnotationJob:
    """Poll one job. The frontend hits this every ~1s while a run is active."""
    job = db.get(AnnotationJob, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found")
    # expire() forces a re-read from the DB rather than returning this session's
    # cached copy. The background task writes progress from a DIFFERENT session,
    # so without this the poller would happily serve a stale row and the
    # progress bar would never move.
    db.expire(job)
    return db.get(AnnotationJob, job_id)


@router.get("/projects/{project_id}/jobs", response_model=list[AnnotationJobRead])
def list_jobs(project_id: int, db: Session = Depends(get_db)) -> list[AnnotationJob]:
    get_project_or_404(project_id, db)
    return list(
        db.scalars(
            select(AnnotationJob)
            .where(AnnotationJob.project_id == project_id)
            .order_by(AnnotationJob.id.desc())
            .limit(20)
        ).all()
    )


# --- Annotations ------------------------------------------------------------


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationRead])
def list_annotations(image_id: int, db: Session = Depends(get_db)) -> list[Annotation]:
    """Boxes for one image — what the Phase 3 canvas loads."""
    if db.get(Image, image_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")
    return list(
        db.scalars(select(Annotation).where(Annotation.image_id == image_id)).all()
    )


@router.get("/projects/{project_id}/annotations/summary")
def annotations_summary(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Counts for the dataset overview.

    A single query with conditional aggregation rather than four COUNT queries —
    same reasoning as the project list's subqueries.
    """
    get_project_or_404(project_id, db)
    rows = db.execute(
        select(Annotation.source, Annotation.reviewed, Annotation.image_id)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
    ).all()

    total_images = len(
        db.scalars(select(Image.id).where(Image.project_id == project_id)).all()
    )
    annotated_image_ids = {r.image_id for r in rows}
    return {
        "total_images": total_images,
        "annotated_images": len(annotated_image_ids),
        "unannotated_images": total_images - len(annotated_image_ids),
        "total_boxes": len(rows),
        "auto_boxes": sum(1 for r in rows if r.source == "auto"),
        "manual_boxes": sum(1 for r in rows if r.source == "manual"),
        "reviewed_boxes": sum(1 for r in rows if r.reviewed),
    }


# --- Export -----------------------------------------------------------------


@router.get("/projects/{project_id}/export")
def export_dataset(
    project_id: int,
    background_tasks: BackgroundTasks,
    format: str = "coco",
    include_unreviewed: bool = True,
    db: Session = Depends(get_db),
) -> FileResponse:
    """Build and download the dataset as a zip.

    Generated on demand, exactly like Roboflow's "Generate version": the DB is
    the source of truth and the export is a derived artifact. Nothing is cached,
    so an export can never be stale relative to your annotations.
    """
    project = get_project_or_404(project_id, db)

    try:
        exporter = exporters.get(format)
    except KeyError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    # mkdtemp, not TemporaryDirectory: the context manager would delete the zip
    # before FileResponse has streamed it. Cleanup is deferred to a background
    # task that runs after the response completes.
    tmp_root = Path(tempfile.mkdtemp(prefix="cvexport_"))
    try:
        dataset_dir = tmp_root / "dataset"
        exporter.export(
            db,
            exporters.ExportRequest(
                project_id=project_id,
                out_dir=dataset_dir,
                include_unreviewed=include_unreviewed,
            ),
        )
        safe_name = "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in project.name
        )
        archive = shutil.make_archive(
            str(tmp_root / f"{safe_name}_{format}"), "zip", root_dir=dataset_dir
        )
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise

    background_tasks.add_task(shutil.rmtree, tmp_root, True)
    return FileResponse(
        archive,
        media_type="application/zip",
        filename=f"{safe_name}_{format}.zip",
    )
