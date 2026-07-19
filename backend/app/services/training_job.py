"""
Training job runner.

`run_training_job` is the function FastAPI's BackgroundTasks executes. Like
`run_annotation_job`, it takes a job id and makes its own session — no Request,
no Depends, no HTTP anything — so the same function runs unchanged under a Celery
worker or a plain script. That seam is the whole reason the ML layer never
imports FastAPI.

WHAT THIS ORCHESTRATES (the trainer stays a thin adapter)
--------------------------------------------------------
  1. Evict any resident annotator, so an auto-annotate model left in VRAM can't
     collide with training on this 4 GB card.
  2. Export the project's dataset to disk in the trainer's format, respecting the
     per-image train/val/test split. Only accepted boxes export — proposals are
     never training data.
  3. Run the trainer, forwarding each epoch's metrics to the DB so the frontend's
     poll shows a live loss/mAP curve.
  4. Record the best checkpoint's path (Phase 5 evaluates and serves it).

Every exit path must leave a terminal status — a job stuck at "running" forever
is the worst failure mode for a UI that polls — so the whole body is wrapped and
failures are recorded with their traceback.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.ml import registry as annotator_registry
from app.ml.device import empty_cache, get_device
from app.ml.trainers import registry as trainer_registry
from app.ml.trainers.base import EpochMetrics, TrainConfig
from app.models import Annotation, Category, Image, JobStatus, TrainingJob
from app.models.image import Split
from app.services import exporters

logger = logging.getLogger(__name__)


def _fail(db: Session, job: TrainingJob, exc: Exception) -> None:
    """Record a job failure with its traceback."""
    job.status = JobStatus.FAILED
    job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    job.finished_at = datetime.now()
    db.commit()
    logger.exception("Training job %s failed", job.id)


def run_training_job(job_id: int) -> None:
    """Execute one training job. Safe to call from any worker.

    Opens its own Session: the request that queued this has long since returned
    and closed its session, so reusing it would raise or, worse, operate on a
    detached object.
    """
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if job is None:
            logger.error("Training job %s vanished before it ran", job_id)
            return

        try:
            _run(db, job)
        except Exception as exc:  # noqa: BLE001
            # Catch-all on purpose: this runs in a background thread with no
            # caller to propagate to. An uncaught exception here would be
            # swallowed and the job would sit at "running" forever. OOM included
            # — it lands here, gets recorded, and the job ends "failed" with the
            # message intact rather than hanging.
            _fail(db, job, exc)
    finally:
        # A trainer should free its own VRAM when train() returns, but on a 4 GB
        # card "should" isn't good enough — hand torch's cached blocks back to
        # the driver so the next job (or an annotate run) has room. Also drop any
        # annotator that somehow survived. One crashed job holding VRAM would
        # otherwise OOM every job after it until a restart.
        annotator_registry.release()
        empty_cache()
        db.close()


def _run(db: Session, job: TrainingJob) -> None:
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now()
    db.commit()

    # Classes define the model's output vocabulary and their ORDER fixes the
    # channel-to-class mapping baked into the checkpoint. Ordered by id, the same
    # order the exporters assign indices — so class_names[N] here matches label N
    # in the exported data and in the trained weights.
    categories = list(
        db.scalars(
            select(Category)
            .where(Category.project_id == job.project_id)
            .order_by(Category.id)
        ).all()
    )
    if not categories:
        raise ValueError(
            "This project has no classes. Add at least one class before training."
        )
    class_names = [c.name for c in categories]
    job.num_classes = len(class_names)

    # Finetuning: resolve the source run's checkpoint to start from. Re-checked
    # here (not just at the route) because this runs later and the file could
    # have gone — a stale id must fail loudly, not silently train from scratch.
    init_weights: Path | None = None
    if job.init_from_job_id is not None:
        source = db.get(TrainingJob, job.init_from_job_id)
        if source is None or not source.checkpoint_path:
            raise ValueError(
                f"Cannot continue from run {job.init_from_job_id}: it has no saved "
                "checkpoint."
            )
        init_weights = Path(source.checkpoint_path)
        if not init_weights.exists():
            raise ValueError(
                f"The checkpoint for run {job.init_from_job_id} is missing on disk "
                f"({init_weights}). Train a fresh model instead."
            )

    # A run learns from ACCEPTED boxes in the TRAIN split. Validate that some
    # exist before paying to export and spin up a framework — a run over zero
    # labels trains a model that predicts nothing, slowly, and looks like a bug.
    train_boxes = _accepted_box_count(db, job.project_id, Split.TRAIN)
    if train_boxes == 0:
        raise ValueError(
            "No accepted boxes in the train split. Accept some proposals or draw "
            "boxes and assign images to 'train' on the Dataset page, then train."
        )

    job.train_images = _image_count(db, job.project_id, Split.TRAIN)
    job.val_images = _image_count(db, job.project_id, Split.VAL)
    db.commit()

    trainer = trainer_registry.get_class(job.trainer_key)()

    # Evict any annotator BEFORE exporting/loading, so the peak-memory window
    # (training) never overlaps a resident annotation model.
    annotator_registry.release()
    empty_cache()

    # Everything this run produces lives under storage/runs/<job_id>/ — the
    # exported dataset it trained on (kept, so "what did it see?" is answerable)
    # and the framework's own output. gitignored, like all of storage/.
    run_dir = settings.runs_dir / str(job.id)
    dataset_dir = run_dir / "dataset"
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export in the trainer's format, honouring the per-image split. Proposals
    # are excluded by the exporter itself — only proposed=False rows are written.
    exporter = exporters.get(trainer.export_format)
    split_map = {
        row_id: row_split
        for row_id, row_split in db.execute(
            select(Image.id, Image.split).where(Image.project_id == job.project_id)
        ).all()
    }
    exporter.export(
        db,
        exporters.ExportRequest(
            project_id=job.project_id,
            out_dir=dataset_dir,
            include_unreviewed=True,
            split=split_map,
            copy_images=True,
        ),
    )

    config = TrainConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=job.epochs,
        batch_size=job.batch_size,
        image_size=job.image_size,
        learning_rate=job.learning_rate,
        num_classes=len(class_names),
        class_names=class_names,
        device=get_device(),
        init_weights=init_weights,
    )

    history: list[dict] = []

    def on_epoch(m: EpochMetrics) -> None:
        """Persist one epoch's metrics as the framework reports them.

        Commits per epoch, for the same reason the annotation runner commits per
        image: it's what makes the progress bar actually move, and a crash at
        epoch 40 keeps the record of the first 39 instead of losing the lot.
        """
        job.current_epoch = m.epoch
        job.total_epochs = m.total_epochs
        job.train_loss = m.train_loss
        job.val_map = m.val_map
        if m.val_map is not None:
            # Best is tracked live because mAP is noisy — the last epoch is
            # rarely the best, and it's the best we keep and report.
            job.best_map = m.val_map if job.best_map is None else max(job.best_map, m.val_map)
        history.append(
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_map": m.val_map,
                "val_map50": m.val_map50,
            }
        )
        job.metrics_json = json.dumps(history)
        db.commit()

    result = trainer.train(config, on_epoch)

    # Authoritative end-of-run values from the trainer override the live
    # estimates: best_map is the true best across all epochs, and the checkpoint
    # path is what Phase 5 will load to evaluate and serve.
    if result.best_checkpoint_path is not None:
        job.checkpoint_path = str(result.best_checkpoint_path)
    if result.best_map is not None:
        job.best_map = result.best_map
    job.current_epoch = result.epochs_completed

    job.status = JobStatus.DONE
    job.finished_at = datetime.now()
    db.commit()
    logger.info(
        "Training job %s done: %d epochs, best mAP %s",
        job.id,
        result.epochs_completed,
        result.best_map,
    )


def _accepted_box_count(db: Session, project_id: int, split: str) -> int:
    """Accepted (non-proposed) boxes on images in one split of a project."""
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
    """Total images in one split of a project (negatives included — they're
    legitimate training data)."""
    return db.scalar(
        select(func.count(Image.id)).where(
            Image.project_id == project_id, Image.split == split
        )
    ) or 0
