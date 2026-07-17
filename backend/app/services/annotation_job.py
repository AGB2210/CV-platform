"""
Auto-annotation job runner.

`run_annotation_job` is the function FastAPI's BackgroundTasks executes. Note
what it does NOT take: no Request, no Depends, no HTTP anything. It takes a job
id and makes its own session.

That's deliberate. It's what lets the same function be called from a Celery
task, an RQ worker, or a plain `python -m` script with no changes — the seam
described in models/annotation_job.py. If this took a request-scoped `db`
session it would be permanently welded to the web layer.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.ml import registry
from app.ml.annotators.base import AnnotationRequest
from app.models import Annotation, AnnotationJob, Category, Image, JobStatus
from app.services import storage

logger = logging.getLogger(__name__)


def _fail(db: Session, job: AnnotationJob, exc: Exception) -> None:
    """Record a job failure with its traceback."""
    job.status = JobStatus.FAILED
    job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    job.finished_at = datetime.now()
    db.commit()
    logger.exception("Annotation job %s failed", job.id)


def run_annotation_job(job_id: int) -> None:
    """Execute one auto-annotation job. Safe to call from any worker.

    Opens its own Session rather than receiving one: the request that queued
    this job has already returned by the time this runs, and its session is
    closed. Reusing it would raise, or worse, silently operate on a detached
    object.
    """
    db = SessionLocal()
    try:
        job = db.get(AnnotationJob, job_id)
        if job is None:
            logger.error("Job %s vanished before it ran", job_id)
            return

        try:
            _run(db, job)
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose. This runs in a background thread with no
            # caller to propagate to — an uncaught exception here would be
            # swallowed by the executor and the job would sit at "running"
            # forever, which is the worst possible failure mode for a UI that
            # polls. Every exit path must leave a terminal status.
            _fail(db, job, exc)
    finally:
        # Always release VRAM, even on failure. A crashed job holding 2.5 GB on
        # a 4 GB card means every subsequent job OOMs — one failure becomes
        # permanent breakage until restart.
        registry.release()
        db.close()


def _run(db: Session, job: AnnotationJob) -> None:
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now()
    db.commit()

    # Classes define what we're looking for. No classes means nothing to prompt
    # with — fail loudly rather than run an expensive no-op.
    categories = list(
        db.scalars(select(Category).where(Category.project_id == job.project_id)).all()
    )
    if not categories:
        raise ValueError(
            "This project has no classes. Add at least one class before auto-annotating."
        )

    by_name = {c.name: c for c in categories}
    class_names = [c.name for c in categories]
    prompts = json.loads(job.prompts_json) if job.prompts_json else {}

    images = list(
        db.scalars(select(Image).where(Image.project_id == job.project_id)).all()
    )
    job.total_images = len(images)
    db.commit()

    if not images:
        job.status = JobStatus.DONE
        job.finished_at = datetime.now()
        db.commit()
        return

    # acquire() loads the model ONCE and keeps it resident for the batch. Using
    # the annotator as a context manager here would unload after each image and
    # reload 700 MB for the next one — turning a 1-second-per-image job into a
    # 10-second-per-image one.
    annotator = registry.acquire(job.model_key)

    total_boxes = 0
    for image in images:
        path = storage.project_dir(image.project_id) / image.filename
        if not path.exists():
            # A DB row whose file is missing. Skip rather than abort: one absent
            # file must not cost you the other 499 images' worth of inference.
            logger.warning("Image file missing, skipping: %s", path)
            job.processed_images += 1
            db.commit()
            continue

        try:
            result = annotator.predict(
                AnnotationRequest(
                    image_path=str(path),
                    class_names=class_names,
                    prompts=prompts,
                    box_threshold=job.box_threshold,
                    text_threshold=job.text_threshold,
                )
            )
        except Exception:
            logger.exception("Inference failed for image %s", image.id)
            job.processed_images += 1
            db.commit()
            continue

        # Clear out what's there before writing this run's output, so re-running
        # doesn't pile duplicate boxes on top of each other.
        #
        # Two policies, and the choice belongs to the user:
        #
        #   clear_existing=False (default) — delete only this pipeline's own
        #     previous `auto` boxes. Human and imported work survives, so a
        #     re-run can never destroy corrections. The cost is that the image
        #     then shows a MIX of model output and human boxes.
        #
        #   clear_existing=True — delete everything first, so the result is
        #     purely what the model said. This is what people mean by "run the
        #     model and show me the output", and without it there was no way to
        #     get a clean slate short of deleting boxes by hand.
        query = select(Annotation).where(Annotation.image_id == image.id)
        if not job.clear_existing:
            query = query.where(Annotation.source == "auto")
        for old in db.scalars(query).all():
            db.delete(old)

        # This image's annotations just changed and no human has looked at the
        # new ones, so it goes back to staging. Leaving it in the dataset would
        # mean training on boxes nobody has reviewed — which is the exact thing
        # the staging/dataset split exists to prevent.
        #
        # It also repairs a dead end: once an image was committed, re-annotating
        # it left no route back to "Add to dataset", because that button only
        # offers approved STAGING images. Now the loop closes:
        #   annotate -> staging -> review -> approve -> commit.
        image.in_dataset = False

        for box in result.boxes:
            category = by_name.get(box.label)
            if category is None:
                continue  # label didn't resolve to a class; drop rather than guess
            x, y, w, h = box.to_coco_bbox()
            db.add(
                Annotation(
                    image_id=image.id,
                    category_id=category.id,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=box.confidence,
                    source="auto",
                    reviewed=False,  # drafts, pending human review
                )
            )
            total_boxes += 1

        job.processed_images += 1
        job.boxes_created = total_boxes
        # Commit per image, not once at the end. It's more fsyncs, but it's what
        # makes the progress bar actually move — and it means a crash at image
        # 400 keeps the first 399 images' annotations instead of losing the lot.
        db.commit()

    job.status = JobStatus.DONE
    job.finished_at = datetime.now()
    db.commit()
    logger.info("Job %s done: %d boxes over %d images", job.id, total_boxes, len(images))
