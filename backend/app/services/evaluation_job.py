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

    # Per-image record for the confusion matrix and the worst-images ranking —
    # kept beside the COCO structures because those flatten away the per-image
    # grouping this analysis needs.
    per_image: list[dict] = []

    for image_index, img in enumerate(test_images, start=1):
        gt_images.append({"id": image_index, "width": img.width, "height": img.height})
        record = {
            "image_id": img.id,
            "filename": img.filename,
            "original_filename": img.original_filename,
            "gt": [],
            "preds": [],
        }
        per_image.append(record)
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
            record["gt"].append((cat_id - 1, [a.x, a.y, a.width, a.height]))

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
            record["preds"].append((cat_id - 1, b.to_coco_bbox(), b.confidence))

    # 4. COCOeval, then the diagnostics the headline number hides.
    metrics = _coco_evaluate(gt_images, gt_annotations, detections, class_names, cat_id_of)
    confusion, worst = _confusion_and_worst(per_image, class_names)

    job.map_50_95 = metrics["map_50_95"]
    job.map_50 = metrics["map_50"]
    job.map_75 = metrics["map_75"]
    job.per_class_json = json.dumps(metrics["per_class"])
    job.details_json = json.dumps(
        {
            "pr_curves": metrics["pr_curves"],
            "confusion": confusion,
            "worst": worst,
        }
    )
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
    pr_curves: list[dict] = []
    # COCOeval traces precision at 101 fixed recall thresholds — that IS the
    # PR curve, already computed; extracting it costs nothing extra. IoU 0.50
    # (T index 0): the loosest standard threshold, which is the one people
    # mean when they picture "the PR curve".
    rec_thrs = list(getattr(ev.params, "recThrs", []))

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

            curve = precision[0, :, k, 0, -1]  # IoU=.50, all recall thresholds
            # -1 = "recall level never reached"; the curve honestly ends there.
            pts = [
                (float(r), float(p))
                for r, p in zip(rec_thrs, curve.tolist())
                if p > -1
            ]
            # Every 2nd point: 51 instead of 101 — indistinguishable on a
            # 300px chart, half the JSON.
            pts = pts[::2]
            pr_curves.append(
                {
                    "name": name,
                    "recall": [round(r, 3) for r, _ in pts],
                    "precision": [round(p, 3) for _, p in pts],
                }
            )
        per_class.append({"name": name, "ap": ap})

    def stat(i: int) -> float | None:
        v = float(stats[i]) if stats is not None and len(stats) > i else None
        return v if v is not None and v >= 0 else None

    return {
        "map_50_95": stat(0),
        "map_50": stat(1),
        "map_75": stat(2),
        "per_class": per_class,
        "pr_curves": pr_curves,
    }


# --- Confusion matrix + worst images -----------------------------------------
# COCOeval sweeps every confidence, which is right for mAP and useless for a
# confusion matrix — "what did the model call this?" only means something at
# an OPERATING POINT. These are the conventional ones (YOLO uses the same).

#: Confidence a detection must clear to count as "the model said so".
_CONFUSION_CONF = 0.25
#: Overlap for a prediction and a ground-truth box to be the same object.
_CONFUSION_IOU = 0.45


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two COCO xywh boxes."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = min(ax2, bx2) - max(a[0], b[0])
    ih = min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _confusion_and_worst(
    per_image: list[dict], class_names: list[str]
) -> tuple[dict, list[dict]]:
    """Greedy IoU matching per image -> confusion matrix + worst-image ranking.

    Matrix layout: matrix[predicted][actual], each axis being the classes plus
    a final "background" slot. An unmatched ground-truth box is the model
    MISSING something (background predicted, actual class); an unmatched
    prediction is the model INVENTING something (predicted class, actual
    background). Diagonal = right box, right class.

    Worst images are ranked by misses + inventions at the same operating
    point — the images to LOOK at, which a single mAP number never names.
    """
    n = len(class_names)
    bg = n  # index of the background row/column
    matrix = [[0] * (n + 1) for _ in range(n + 1)]
    scored: list[dict] = []

    for rec in per_image:
        preds = sorted(
            (p for p in rec["preds"] if p[2] >= _CONFUSION_CONF),
            key=lambda p: -p[2],
        )
        gt = list(rec["gt"])
        gt_taken = [False] * len(gt)
        fp = fn = 0

        for p_cls, p_box, _score in preds:
            best_i, best_iou = -1, _CONFUSION_IOU
            for i, (g_cls, g_box) in enumerate(gt):
                if gt_taken[i]:
                    continue
                iou = _iou(p_box, g_box)
                if iou >= best_iou:
                    best_i, best_iou = i, iou
            if best_i >= 0:
                gt_taken[best_i] = True
                g_cls = gt[best_i][0]
                matrix[p_cls][g_cls] += 1
                if p_cls != g_cls:
                    fp += 1  # right place, wrong name — still an error to review
            else:
                matrix[p_cls][bg] += 1
                fp += 1

        for i, (g_cls, _g_box) in enumerate(gt):
            if not gt_taken[i]:
                matrix[bg][g_cls] += 1
                fn += 1

        if fp or fn:
            scored.append(
                {
                    "image_id": rec["image_id"],
                    "filename": rec["filename"],
                    "original_filename": rec["original_filename"],
                    "fp": fp,
                    "fn": fn,
                }
            )

    scored.sort(key=lambda s: -(s["fp"] + s["fn"]))
    confusion = {"classes": [*class_names, "background"], "matrix": matrix}
    return confusion, scored[:12]
