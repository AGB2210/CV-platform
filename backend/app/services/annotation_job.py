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
        # Always release VRAM, even on failure. A crashed job still holding the
        # model means every subsequent job OOMs — one failure becomes
        # permanent breakage until restart.
        registry.release()
        db.close()


def _images_in_scope(db: Session, job: AnnotationJob) -> list[Image]:
    """The images this job covers.

      selected     exactly the ids the request named.
      unannotated  images with no ACCEPTED boxes — fill the gaps.
      all          every image in the project.
    """
    query = select(Image).where(Image.project_id == job.project_id)

    if job.image_ids_json:
        ids = json.loads(job.image_ids_json)
        # Still filtered by project_id above: an id from another project can't
        # be annotated just because it appeared in the request body.
        query = query.where(Image.id.in_(ids))
    elif job.scope == "unannotated":
        # Images with no accepted boxes. A pending proposal doesn't count as
        # annotated — it isn't an annotation — so an image whose only boxes are
        # last run's un-actioned suggestions is still a gap worth filling.
        annotated = select(Annotation.image_id).where(
            Annotation.proposed.is_(False)
        ).distinct()
        query = query.where(Image.id.not_in(annotated))
    # "all" adds no filter.

    return list(db.scalars(query).all())


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

    images = _images_in_scope(db, job)
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

        # A run writes PROPOSALS. It does not touch your existing boxes.
        #
        # The only thing cleared is this pipeline's own leftover proposals from
        # an earlier run — otherwise running the model twice would stack two
        # sets of suggestions for the same objects. Accepted boxes (yours,
        # imported, or previously-accepted model output) are never touched here:
        # the accept/reject decision belongs to you, at review time.
        #
        # The image is NOT re-staged either. Its accepted annotations haven't
        # changed — a proposal is not an annotation until you say so — so a
        # committed image stays committed and the dataset is undisturbed even
        # under scope="all".
        # Scoped to source="auto": THIS pipeline's leftovers from an earlier
        # run, which would otherwise stack two sets of suggestions for the same
        # objects. Imported proposals (source="imported", from re-uploading an
        # annotation file) are a different person's work awaiting the same
        # review, and running the model must not throw them away.
        for stale in db.scalars(
            select(Annotation).where(
                Annotation.image_id == image.id,
                Annotation.proposed.is_(True),
                Annotation.source == "auto",
            )
        ).all():
            db.delete(stale)

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
                    reviewed=False,
                    # A suggestion, not an annotation. Invisible to exports,
                    # training and counts until accepted.
                    proposed=True,
                    job_id=job.id,
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
