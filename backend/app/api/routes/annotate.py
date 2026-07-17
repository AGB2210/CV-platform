"""Auto-annotation, annotation reads, and dataset export."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.ml import registry
from app.ml.device import device_info
from app.models import Annotation, AnnotationJob, Category, Image, JobStatus
from app.schemas.annotation import (
    AnnotationCreate,
    AnnotationJobCreate,
    AnnotationJobRead,
    AnnotationRead,
    AnnotationUpdate,
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
        clear_existing=payload.clear_existing,
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


@router.post(
    "/images/{image_id}/annotations",
    response_model=AnnotationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_annotation(
    image_id: int, payload: AnnotationCreate, db: Session = Depends(get_db)
) -> Annotation:
    """Add a human-drawn box.

    source="manual" and reviewed=True: a box a person drew by hand is, by
    definition, already reviewed. Making them click "approve" on their own work
    would be busywork.
    confidence stays NULL — a human doesn't have one, and 1.0 would be a lie
    that pollutes any later look at model calibration.
    """
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    category = db.get(Category, payload.category_id)
    if category is None or category.project_id != image.project_id:
        # The project check is not paranoia: without it you could attach a class
        # from project A to an image in project B, producing a dataset that
        # exports with a category_id pointing at nothing.
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Class {payload.category_id} does not belong to this image's project",
        )

    ann = Annotation(
        image_id=image_id,
        category_id=payload.category_id,
        # Clamp to the image. The canvas already clamps, but a box is data we
        # will train on — validating at the boundary means a frontend bug can't
        # silently poison the dataset.
        x=max(0.0, min(payload.x, image.width)),
        y=max(0.0, min(payload.y, image.height)),
        width=min(payload.width, image.width - max(0.0, payload.x)),
        height=min(payload.height, image.height - max(0.0, payload.y)),
        confidence=None,
        source="manual",
        reviewed=True,
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann


@router.patch("/annotations/{annotation_id}", response_model=AnnotationRead)
def update_annotation(
    annotation_id: int, payload: AnnotationUpdate, db: Session = Depends(get_db)
) -> Annotation:
    """Move, resize, relabel, or approve one box.

    ANY human edit — geometry OR label — promotes an auto box to
    source="manual". Once a person has corrected it, it is no longer the model's
    output, and a re-run of that model must not silently delete their work: the
    job runner only replaces source="auto" boxes.

    Relabelling counts. It's tempting to treat only geometry as a "real" edit,
    but fixing person -> car is exactly as much human judgement as nudging a
    corner, and leaving it source="auto" means the next re-run throws it away
    with no error and no warning. (Verified: it did.)

    Setting `reviewed` alone does NOT promote — that's the approve action
    confirming the model was right, not a human overriding it. A box can be
    reviewed and still be the model's own output, which is precisely the
    distinction that makes "trained on verified model output" meaningful.
    """
    ann = db.get(Annotation, annotation_id)
    if ann is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Annotation {annotation_id} not found"
        )

    fields = payload.model_dump(exclude_unset=True)

    if "category_id" in fields:
        category = db.get(Category, fields["category_id"])
        image = db.get(Image, ann.image_id)
        if category is None or (image and category.project_id != image.project_id):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "Class does not belong to this project"
            )

    # A human overriding the model, in either dimension: where the box is, or
    # what it is. Deliberately excludes `reviewed` — approving is confirming the
    # model, not overriding it.
    human_edit = any(
        k in fields for k in ("x", "y", "width", "height", "category_id")
    )

    for field, value in fields.items():
        setattr(ann, field, value)

    if human_edit and ann.source == "auto":
        ann.source = "manual"
        ann.reviewed = True

    db.commit()
    db.refresh(ann)
    return ann


@router.delete("/annotations/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)) -> None:
    ann = db.get(Annotation, annotation_id)
    if ann is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Annotation {annotation_id} not found"
        )
    db.delete(ann)
    db.commit()


@router.post("/images/{image_id}/annotations/approve", response_model=list[AnnotationRead])
def approve_image(image_id: int, db: Session = Depends(get_db)) -> list[Annotation]:
    """Mark every box on this image as reviewed.

    The bulk path that makes review tractable: when the model got an image
    right — which is most of them — the whole interaction should be one key
    press, not one click per box.
    """
    if db.get(Image, image_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    anns = list(
        db.scalars(select(Annotation).where(Annotation.image_id == image_id)).all()
    )
    for ann in anns:
        ann.reviewed = True
    db.commit()
    for ann in anns:
        db.refresh(ann)
    return anns


@router.get("/projects/{project_id}/annotate/preview")
def annotate_preview(project_id: int, db: Session = Depends(get_db)) -> dict:
    """What a run would destroy, so the UI can say so before the click.

    Auto-annotation is not additive — it clears prior output before writing new
    output. Without this, the only way to discover that a re-run wiped your 40
    hand-drawn boxes is to notice they're gone.
    """
    get_project_or_404(project_id, db)

    rows = db.execute(
        select(Annotation.source, func.count(Annotation.id))
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
        .group_by(Annotation.source)
    ).all()
    by_source = {source: n for source, n in rows}

    in_dataset = db.scalar(
        select(func.count(Image.id)).where(
            Image.project_id == project_id, Image.in_dataset.is_(True)
        )
    ) or 0

    return {
        "auto_boxes": by_source.get("auto", 0),
        # These two survive a default run and are destroyed by clear_existing —
        # the number the user actually needs before ticking that box.
        "manual_boxes": by_source.get("manual", 0),
        "imported_boxes": by_source.get("imported", 0),
        # Annotated images return to staging for re-review, which empties this.
        "images_in_dataset": in_dataset,
    }


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
