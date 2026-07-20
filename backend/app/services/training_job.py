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
import shutil
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
from app.services.dataset_snapshot import build_snapshot

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

    # THE DATASET THIS RUN TRAINS ON IS A SAVED VERSION, not the live rows.
    # Everything below — classes, counts, validation, the export — is derived
    # from that snapshot, so the run is reproducible and its recorded provenance
    # ("trained on dataset v3") stays true however the project changes later.
    snapshot = _resolve_snapshot(db, job)

    # Classes define the model's output vocabulary and their ORDER fixes the
    # channel-to-class mapping baked into the checkpoint — the same order the
    # exporter assigns indices, so class_names[N] means label N in the exported
    # data and in the trained weights.
    if not snapshot.categories:
        raise ValueError(
            "That dataset version has no classes. Add at least one class, save "
            "the dataset, and train again."
        )
    class_names = [c.name for c in snapshot.categories]
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

    # A run learns from the boxes in the TRAIN split. Validate that some exist
    # before paying to export and spin up a framework — a run over zero labels
    # trains a model that predicts nothing, slowly, and looks like a bug.
    if snapshot.box_count_for_split(Split.TRAIN) == 0:
        raise ValueError(
            "That dataset version has no boxes in its train split. Annotate some "
            "images, save the dataset, and train the new version."
        )

    split_counts = snapshot.split_counts()
    job.train_images = split_counts.get(Split.TRAIN, 0)
    job.val_images = split_counts.get(Split.VAL, 0)
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
    # Each snapshot image carries its own split, so the exporter needs no map.
    exporter = exporters.get(trainer.export_format)
    exporter.export(
        snapshot,
        exporters.ExportRequest(
            out_dir=dataset_dir,
            include_unreviewed=True,
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

    _reclaim_space(run_dir, job)

    logger.info(
        "Training job %s done: %d epochs, best mAP %s",
        job.id,
        result.epochs_completed,
        result.best_map,
    )


def _reclaim_space(run_dir: Path, job: TrainingJob) -> None:
    """Drop the run's bulky, redundant artifacts once it has succeeded.

    WHY: a run exports the whole dataset — IMAGE FILES INCLUDED — into its own
    directory. Keeping that forever means every training run permanently
    duplicates the entire dataset on disk. On a real project (gigabytes of
    photos) ten runs is ten copies, which fills a disk and slows the machine
    down. It was measured at 11 MB/run on a 130-image toy set, and that scales
    linearly with dataset size.

    It's safe to drop precisely because dataset VERSIONS exist now: the run
    records which version it trained, and that snapshot is the durable,
    reproducible answer to "what went into this model". The copy was the only
    record before versions; now it's redundant.

    `last.pt` goes too — ultralytics writes it for resuming, but we continue
    training by loading `best.pt` into a fresh run, so it's dead weight at
    ~5 MB a time.

    Only on SUCCESS. A failed run keeps everything, because when an export or a
    trainer misbehaves those files are the evidence.
    """
    try:
        shutil.rmtree(run_dir / "dataset", ignore_errors=True)
        if job.checkpoint_path:
            last = Path(job.checkpoint_path).with_name("last.pt")
            last.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        # Housekeeping must never fail a run that actually succeeded.
        logger.warning("Could not reclaim space for run %s", job.id, exc_info=True)


def _resolve_snapshot(db: Session, job: TrainingJob):
    """The dataset this run trains on: its recorded dataset version's snapshot.

    Re-resolved here rather than trusting the route, because this runs later —
    the version's file could have gone missing in between, and a run that
    silently fell back to the live rows while claiming to be version 3 would be
    the worst kind of wrong: plausible and untrue.
    """
    from app.models import DatasetVersion
    from app.services.dataset_version import load_snapshot

    if job.dataset_version_id is None:
        raise ValueError(
            "This run has no dataset version. Save the dataset, then train."
        )
    version = db.get(DatasetVersion, job.dataset_version_id)
    if version is None or version.project_id != job.project_id:
        raise ValueError(
            f"Dataset version {job.dataset_version_id} no longer exists in this project."
        )
    return load_snapshot(version)


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
