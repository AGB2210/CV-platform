"""
Evaluation job runner — score a trained model on a held-out TEST split.

`run_evaluation_job(job_id)` is what FastAPI's BackgroundTasks executes, in the
same shape as the training and annotation runners: takes an id, opens its own
session, leaves a terminal status on every exit path.

WHAT IT DOES
------------
  1. Resolve the model (a finished TrainingJob's checkpoint) and the class list it
     trained on, from the run's dataset-version snapshot.
  2. Resolve the TEST split of the chosen dataset version — the images and their
     ground-truth boxes. Training used train + val; test was untouched, so this
     is the first data the model has met in no role at all.
  3. Run the model over each test image at a LOW confidence floor (COCOeval
     sweeps confidence itself — thresholding here would throw away the low-score
     detections it needs to trace the precision/recall curve).
  4. Feed ground truth and predictions to pycocotools' COCOeval and record the
     headline mAP plus per-class AP.

torch and pycocotools import lazily inside, so importing this module stays free.
"""

from __future__ import annotations

import json
import logging
import traceback

from sqlalchemy.orm import Session

from app.config import from_storage_path
from app.database import SessionLocal
from app.ml import registry as annotator_registry
from app.ml.device import empty_cache
from app.ml.predictors import registry as predictor_registry
from app.models import EvaluationJob, JobStatus, TrainingJob
from app.models.image import Split
from app.services import storage
from app.timestamps import utcnow

logger = logging.getLogger(__name__)

#: Detections below this score are dropped before COCOeval. Deliberately tiny:
#: COCOeval integrates precision over ALL recall levels, so it needs the weak
#: detections too. This is a floor to bound the list size, not the user's dial.
_EVAL_CONF_FLOOR = 0.001


def _fail(db: Session, job: EvaluationJob, exc: Exception) -> None:
    job.status = JobStatus.FAILED
    job.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    job.finished_at = utcnow()
    db.commit()
    logger.exception("Evaluation job %s failed", job.id)


def run_evaluation_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(EvaluationJob, job_id)
        if job is None:
            logger.error("Evaluation job %s vanished before it ran", job_id)
            return
        try:
            _run(db, job)
        except Exception as exc:  # noqa: BLE001 — background thread, nothing to bubble to
            _fail(db, job, exc)
    finally:
        # Hand the card back — an evaluation holds a predictor resident.
        annotator_registry.release()
        predictor_registry.release()
        empty_cache()
        db.close()


def _run(db: Session, job: EvaluationJob) -> None:
    from app.models import DatasetVersion
    from app.services.dataset_version import load_snapshot

    job.status = JobStatus.RUNNING
    job.started_at = utcnow()
    db.commit()

    # 1. The model.
    model_job = db.get(TrainingJob, job.training_job_id)
    if model_job is None or not model_job.checkpoint_path:
        raise ValueError("The model to evaluate has no saved checkpoint.")
    checkpoint = from_storage_path(model_job.checkpoint_path)
    if checkpoint is None or not checkpoint.exists():
        raise ValueError(f"The checkpoint file is missing on disk ({checkpoint}).")

    # 2. The dataset version and its test split. class_names come from the model's
    #    OWN training snapshot so label indices line up with what it learned.
    version = db.get(DatasetVersion, job.dataset_version_id)
    if version is None:
        raise ValueError("That dataset version no longer exists.")
    eval_snapshot = load_snapshot(version)

    train_version = (
        db.get(DatasetVersion, model_job.dataset_version_id)
        if model_job.dataset_version_id is not None
        else None
    )
    class_names = [
        c.name
        for c in (load_snapshot(train_version).categories if train_version else eval_snapshot.categories)
    ]
    # coco category id per class name: 1-based, in class-index order.
    cat_id_of = {name: i + 1 for i, name in enumerate(class_names)}
    # snapshot category id -> name, to read ground-truth labels.
    gt_name_of = {c.id: c.name for c in eval_snapshot.categories}

    test_images = [img for img in eval_snapshot.images if img.split == job.split]
    if not test_images:
        raise ValueError(
            f"Dataset v{version.version} has no {job.split} images. Assign some "
            "images to the test split (or upload a test set) and evaluate again."
        )
    job.num_images = len(test_images)
    db.commit()

    # 3. Build COCO ground truth, and run the model for detections.
    gt_images: list[dict] = []
    gt_annotations: list[dict] = []
    ann_id = 1

    predictor = predictor_registry.acquire(
        model_job.trainer_key, str(checkpoint), class_names
    )
    detections: list[dict] = []
    project_dir = storage.project_dir(job.project_id)

    for image_index, img in enumerate(test_images, start=1):
        gt_images.append({"id": image_index, "width": img.width, "height": img.height})
        for a in img.annotations:
            name = gt_name_of.get(a.category_id)
            cat_id = cat_id_of.get(name) if name else None
            if cat_id is None:
                # A ground-truth class the model never trained on can't be scored
                # against it — skip rather than crash, and it simply counts as a
                # class the model gets no credit for.
                continue
            gt_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_index,
                    "category_id": cat_id,
                    "bbox": [a.x, a.y, a.width, a.height],
                    "area": a.width * a.height,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        path = project_dir / img.filename
        if not path.exists():
            continue  # a missing file scores as zero detections, not a crash
        for b in predictor.predict(str(path), conf_threshold=_EVAL_CONF_FLOOR):
            cat_id = cat_id_of.get(b.label)
            if cat_id is None:
                continue
            detections.append(
                {
                    "image_id": image_index,
                    "category_id": cat_id,
                    "bbox": b.to_coco_bbox(),
                    "score": b.confidence,
                }
            )

    # 4. COCOeval.
    metrics = _coco_evaluate(gt_images, gt_annotations, detections, class_names, cat_id_of)

    job.map_50_95 = metrics["map_50_95"]
    job.map_50 = metrics["map_50"]
    job.map_75 = metrics["map_75"]
    job.per_class_json = json.dumps(metrics["per_class"])
    job.status = JobStatus.DONE
    job.finished_at = utcnow()
    db.commit()
    logger.info(
        "Evaluation job %s done: %d images, test mAP %.4f",
        job.id,
        job.num_images,
        metrics["map_50_95"] or 0.0,
    )


def _coco_evaluate(
    gt_images: list[dict],
    gt_annotations: list[dict],
    detections: list[dict],
    class_names: list[str],
    cat_id_of: dict[str, int],
) -> dict:
    """Run COCOeval and pull out the headline mAP and per-class AP.

    pycocotools imports lazily here. With no detections at all COCOeval would
    still run but every number is zero — which is the correct, honest answer for
    a model that found nothing, so it is not special-cased.
    """
    import contextlib
    import io

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    categories = [{"id": cid, "name": name} for name, cid in cat_id_of.items()]

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": categories,
    }
    # createIndex() and loadRes() print progress; the run is quiet enough that
    # this noise in the server log is not worth it, so swallow their stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections) if detections else COCO()
        if not detections:
            coco_dt.dataset = {"images": gt_images, "annotations": [], "categories": categories}
            coco_dt.createIndex()

        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    stats = ev.stats  # the 12 standard COCO numbers
    # precision: [T iou, R recall, K class, A area, M maxDet]. Per-class AP@[.5:.95]
    # is the mean over IoU and recall for that class at area=all, maxDet=100.
    precision = ev.eval["precision"] if ev.eval else None

    per_class: list[dict] = []
    for name, cid in cat_id_of.items():
        ap = None
        if precision is not None:
            # class order in COCOeval follows sorted category ids, which are our
            # 1-based indices, so k = cid - 1.
            k = cid - 1
            col = precision[:, :, k, 0, -1]
            col = col[col > -1]
            if col.size:
                ap = float(np.mean(col))
        per_class.append({"name": name, "ap": ap})

    def stat(i: int) -> float | None:
        v = float(stats[i]) if stats is not None and len(stats) > i else None
        return v if v is not None and v >= 0 else None

    return {
        "map_50_95": stat(0),
        "map_50": stat(1),
        "map_75": stat(2),
        "per_class": per_class,
    }
