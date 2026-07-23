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
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.timestamps import utcnow
from app.ml import registry
from app.ml.predictors import registry as predictor_registry
from app.ml.annotators.base import AnnotationRequest
from app.models import Annotation, AnnotationJob, Category, Image, JobStatus
from app.services import storage

logger = logging.getLogger(__name__)


def _fail(db: Session, job: AnnotationJob, exc: Exception) -> None:
    """Record a job failure with its traceback."""
    job.status = JobStatus.FAILED
    job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    job.finished_at = utcnow()
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
    # The annotator THIS run acquired, if it got that far — set by _run so the
    # cleanup below can release exactly it and nothing else.
    owned: list = []
    try:
        job = db.get(AnnotationJob, job_id)
        if job is None:
            logger.error("Job %s vanished before it ran", job_id)
            return

        try:
            _run(db, job, owned)
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose. This runs in a background thread with no
            # caller to propagate to — an uncaught exception here would be
            # swallowed by the executor and the job would sit at "running"
            # forever, which is the worst possible failure mode for a UI that
            # polls. Every exit path must leave a terminal status.
            _fail(db, job, exc)
    finally:
        # Release OUR model, even on failure — a crashed job still holding it
        # means every subsequent job OOMs. `only=` matters: with GPU admission
        # the next queued job may already be admitted and loading ITS model by
        # the time this line runs (terminal status commits before the finally),
        # and an unconditional release here unloaded that model mid-job.
        registry.release(only=owned[0] if owned else None)
        # A predictor left resident by an idle playground would also be freed —
        # but only when no evaluation could own it. Admission blocks this
        # runner while an evaluation is queued/running, so the check mirrors
        # the boundary case: an eval started the instant we finished.
        if not _evaluation_active(db):
            predictor_registry.release()
        db.close()


def _evaluation_active(db: Session) -> bool:
    """Is an evaluation queued or running? Its predictor must not be evicted."""
    from app.models import EvaluationJob, JobStatus

    return bool(
        db.scalar(
            select(EvaluationJob.id).where(
                EvaluationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
            )
        )
    )


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


def _run(db: Session, job: AnnotationJob, owned: list) -> None:
    # GPU admission first, while still QUEUED: another job (any project) may
    # hold the card. status_detail carries the live waiting reason; False
    # means the user cancelled the wait.
    from app.services import gpu_admission

    annotator_cls = registry.get_class(job.model_key)
    if not gpu_admission.wait_for_gpu(
        db, job, "annotation", annotator_cls.approx_vram_gb
    ):
        _discard(db, job)
        return

    job.status = JobStatus.RUNNING
    job.started_at = utcnow()
    db.commit()

    # Cancelled while still queued — bail before loading a 700 MB model for
    # work that has already been called off.
    if job.control == "cancel":
        _discard(db, job)
        return

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
        job.finished_at = utcnow()
        db.commit()
        return

    # Free the card before loading the annotator: a playground can leave a
    # predictor resident, and DINO + a detector will not both fit on a small GPU.
    predictor_registry.release()

    # acquire() loads the model ONCE and keeps it resident for the batch. Using
    # the annotator as a context manager here would unload after each image and
    # reload 700 MB for the next one — turning a 1-second-per-image job into a
    # 10-second-per-image one.
    annotator = registry.acquire(job.model_key)
    # Recorded for the caller's cleanup: the finally releases exactly this
    # instance, so a successor job's freshly-loaded model is never collateral.
    owned.append(annotator)

    total_boxes = 0
    for image in images:
        # Cancel is read BETWEEN images, like training reads its control between
        # epochs: the per-image commit below re-queries the row (expire_on_commit),
        # so a flag written by the cancel request's session becomes visible here.
        # Mid-image is not interruptible — one image costs about a second, and a
        # half-written image's proposals would be exactly the kind of partial
        # state the discard exists to prevent.
        if job.control == "cancel":
            _discard(db, job)
            return

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

    # A cancel that landed during the final image still wins: the user asked for
    # this run not to exist, and "it finished anyway" is not an answer.
    if job.control == "cancel":
        _discard(db, job)
        return

    job.status = JobStatus.DONE
    job.finished_at = utcnow()
    db.commit()
    logger.info("Job %s done: %d boxes over %d images", job.id, total_boxes, len(images))


def _discard(db: Session, job: AnnotationJob) -> None:
    """Cancel: throw the run's OUTPUT away, keep its record.

    The run's own proposals (job_id = this job) are deleted; proposals from
    OTHER runs or imports are untouched, and accepted boxes were never touched
    to begin with.

    The ROW survives, as status "cancelled". An earlier design deleted it and
    let the poller read the resulting 404 as the end of the run — which had two
    real costs, both found in use: SQLite reuses the freed rowid, so a later
    job wore a cancelled job's number; and a cancel interrupted by a server
    restart looked identical to a crash, so startup marked it FAILED — the
    user cancelled and was told it failed.
    """
    for stale in db.scalars(
        select(Annotation).where(Annotation.job_id == job.id, Annotation.proposed.is_(True))
    ).all():
        db.delete(stale)
    job.status = JobStatus.CANCELLED
    job.control = None
    job.status_detail = None
    job.finished_at = utcnow()
    db.commit()
    logger.info("Annotation job %s cancelled; proposals discarded", job.id)
