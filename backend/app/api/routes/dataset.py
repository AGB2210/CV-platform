"""
Dataset stats and train/val/test splits.

WHAT USED TO BE HERE
--------------------
A staging -> dataset commit step (append/merge/replace), mirroring Roboflow.
It's gone. The proposal model already prevents unreviewed model output from
becoming an annotation, so staging was a SECOND gate that only ever blocked you
from using work you had already accepted — and the commit dialog asked a
question whose answer was always "yes, add it". Accepting is the commit.

Every image is now a dataset image. `split` is just a property you set, and the
control for it lives on the Dataset page next to the grid.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Category, DatasetVersion, Image
from app.models.image import Split
from app.schemas.dataset import (
    BulkDeleteVersions,
    BulkSplitRequest,
    DatasetStats,
    DatasetVersionCreate,
    DatasetVersionRead,
    DeleteResult,
    RestoreResult,
    SplitCounts,
    SplitRequest,
    VersionRename,
)
from app.services import dataset_version as versions
from app.services.version_naming import DuplicateNameError, clean_name, ensure_unique

router = APIRouter(tags=["dataset"])


# --- Versions ---------------------------------------------------------------
# "Save dataset" is the only thing that creates a version, and it's the gate into
# training: you train a SAVED dataset, so every run points at something
# reproducible rather than at whatever the rows happened to be that afternoon.


@router.post(
    "/projects/{project_id}/dataset/versions",
    response_model=DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def save_dataset_version(
    project_id: int,
    payload: DatasetVersionCreate,
    db: Session = Depends(get_db),
) -> DatasetVersion:
    """Save the current dataset as a new, restorable version."""
    get_project_or_404(project_id, db)
    try:
        return versions.save_version(db, project_id, note=payload.note)
    except versions.DatasetVersionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None


@router.get(
    "/projects/{project_id}/dataset/versions",
    response_model=list[DatasetVersionRead],
)
def list_dataset_versions(
    project_id: int, db: Session = Depends(get_db)
) -> list[DatasetVersion]:
    """Newest first — the version history for this project.

    Each row is tagged with whether it's the one the LIVE dataset currently
    matches. That is not always the newest: restore an older version and the
    dataset on screen IS that older one, while a newer save point still exists.
    Without saying so, the list gives no way to tell which state you're in.
    """
    get_project_or_404(project_id, db)
    rows = list(
        db.scalars(
            select(DatasetVersion)
            .where(DatasetVersion.project_id == project_id)
            .order_by(DatasetVersion.version.desc())
        ).all()
    )
    current = versions.current_version(db, project_id)
    for row in rows:
        # Transient attribute, not a column — computed per request from the live
        # data, so it can never go stale in the way a stored flag would.
        row.is_current = current is not None and row.id == current.id
    return rows


@router.patch(
    "/projects/{project_id}/dataset/versions/{version_id}",
    response_model=DatasetVersionRead,
)
def rename_dataset_version(
    project_id: int,
    version_id: int,
    payload: VersionRename,
    db: Session = Depends(get_db),
) -> DatasetVersion:
    """Rename a version. Blank clears the name, reverting to "v{n}"."""
    get_project_or_404(project_id, db)
    version = _get_version_or_404(db, project_id, version_id)

    name = clean_name(payload.name)
    others = [
        (v.name, v.version)
        for v in db.scalars(
            select(DatasetVersion).where(
                DatasetVersion.project_id == project_id,
                DatasetVersion.id != version_id,
            )
        ).all()
    ]
    try:
        ensure_unique(name, version.version, others)
    except DuplicateNameError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None

    version.name = name
    db.commit()
    db.refresh(version)
    return version


@router.delete(
    "/projects/{project_id}/dataset/versions/{version_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_dataset_version(
    project_id: int, version_id: int, db: Session = Depends(get_db)
) -> None:
    """Delete one version. Its snapshot file goes too — so that version can no
    longer be restored or trained. The UI states that before the click."""
    get_project_or_404(project_id, db)
    version = _get_version_or_404(db, project_id, version_id)
    versions.delete_version(db, version)


@router.post(
    "/projects/{project_id}/dataset/versions/bulk-delete",
    response_model=DeleteResult,
)
def bulk_delete_dataset_versions(
    project_id: int, payload: BulkDeleteVersions, db: Session = Depends(get_db)
) -> DeleteResult:
    """Delete several versions — also how "delete all" is sent, so selecting
    everything and deleting is the same path as deleting one."""
    get_project_or_404(project_id, db)
    found = list(
        db.scalars(
            select(DatasetVersion).where(
                DatasetVersion.project_id == project_id,
                DatasetVersion.id.in_(payload.version_ids),
            )
        ).all()
    )
    for version in found:
        versions.delete_version(db, version)
    found_ids = {v.id for v in found}
    return DeleteResult(
        deleted=len(found),
        not_found=sorted(set(payload.version_ids) - found_ids),
    )


@router.post(
    "/projects/{project_id}/dataset/versions/{version_id}/restore",
    response_model=RestoreResult,
)
def restore_dataset_version(
    project_id: int, version_id: int, db: Session = Depends(get_db)
) -> RestoreResult:
    """Rewind the live dataset to a saved version.

    Nothing is auto-saved first — only an explicit "Save dataset" makes a
    version — so this DISCARDS unsaved changes. The UI warns before the click
    when the dataset matches no save point.
    """
    get_project_or_404(project_id, db)
    version = _get_version_or_404(db, project_id, version_id)
    try:
        result = versions.restore_version(db, project_id, version)
    except versions.DatasetVersionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from None

    return RestoreResult(
        restored_version=version.version,
        images_restored=result.images_restored,
        boxes_restored=result.boxes_restored,
        images_removed=result.images_removed,
        missing_files=result.missing_files,
        classes_removed=result.classes_removed,
    )


def _get_version_or_404(db: Session, project_id: int, version_id: int) -> DatasetVersion:
    version = db.get(DatasetVersion, version_id)
    if version is None or version.project_id != project_id:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Dataset version {version_id} not found in this project.",
        )
    return version


@router.get("/projects/{project_id}/dataset/stats", response_model=DatasetStats)
def dataset_stats(project_id: int, db: Session = Depends(get_db)) -> DatasetStats:
    """Counts for the Dataset page header."""
    get_project_or_404(project_id, db)

    images = list(db.scalars(select(Image).where(Image.project_id == project_id)).all())

    annotated_ids = {
        row[0]
        for row in db.execute(
            select(Annotation.image_id)
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.project_id == project_id, Annotation.proposed.is_(False))
            .distinct()
        ).all()
    }

    boxes = db.execute(
        select(
            # Accepted boxes only. Proposals get their own figure rather than
            # being folded in — "you have 90 boxes" is a lie if 60 are
            # suggestions nobody has looked at.
            func.sum(case((Annotation.proposed.is_(False), 1), else_=0)),
            func.sum(case((Annotation.proposed.is_(True), 1), else_=0)),
            func.count(
                func.distinct(case((Annotation.proposed.is_(True), Annotation.image_id)))
            ),
        )
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
    ).one()

    counts = SplitCounts()
    for image in images:
        setattr(counts, image.split, getattr(counts, image.split, 0) + 1)

    return DatasetStats(
        total_images=len(images),
        annotated_images=len(annotated_ids),
        unannotated_images=len(images) - len(annotated_ids),
        splits=counts,
        total_boxes=boxes[0] or 0,
        proposed_boxes=boxes[1] or 0,
        proposed_images=boxes[2] or 0,
    )


#: Relative-size histogram bins: sqrt(box_area / image_area) in [0, 1].
#: sqrt makes the number read as "fraction of image WIDTH the box spans",
#: which humans reason about far better than raw area ratios.
_HEALTH_BINS = 10

#: A box whose sqrt-relative-size is below this is genuinely hard for a
#: detector to learn at ordinary training resolutions: 0.03 of a 640px image
#: is ~19px on the long side.
_TINY_RELATIVE = 0.03


@router.get("/projects/{project_id}/dataset/health")
def dataset_health(project_id: int, db: Session = Depends(get_db)) -> dict:
    """The dataset's SHAPE — the answer to "why is my mAP low?".

    Disappointing training results are usually the data, not the model: one
    class with 40x the boxes of another, half the boxes too tiny to learn at
    training resolution, a class defined but never labelled. None of that is
    visible from counts alone, so this endpoint computes the distributions and
    turns the pathological ones into named warnings.

    Accepted boxes only, live rows (not a saved version) — health is a
    property of what you're editing right now.
    """
    get_project_or_404(project_id, db)

    categories = list(
        db.scalars(select(Category).where(Category.project_id == project_id)).all()
    )
    total_images = db.scalar(
        select(func.count(Image.id)).where(Image.project_id == project_id)
    ) or 0

    # One pull, computed in Python: even a large local project is a few
    # thousand boxes, and the relative-size maths needs the image dims anyway.
    rows = db.execute(
        select(
            Annotation.category_id,
            Annotation.image_id,
            Annotation.width,
            Annotation.height,
            Image.width,
            Image.height,
        )
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Annotation.proposed.is_(False))
    ).all()

    by_class_boxes: dict[int, int] = {}
    by_class_images: dict[int, set[int]] = {}
    hist = [0] * _HEALTH_BINS
    small = medium = large = tiny = 0
    annotated_images: set[int] = set()

    for cat_id, image_id, bw, bh, iw, ih in rows:
        by_class_boxes[cat_id] = by_class_boxes.get(cat_id, 0) + 1
        by_class_images.setdefault(cat_id, set()).add(image_id)
        annotated_images.add(image_id)

        # COCO's absolute-area buckets, for cross-paper comparability.
        area = bw * bh
        if area < 32 * 32:
            small += 1
        elif area < 96 * 96:
            medium += 1
        else:
            large += 1

        # Scale-independent size: fraction of the image's linear extent.
        rel = (area / (iw * ih)) ** 0.5 if iw and ih else 0.0
        hist[min(int(rel * _HEALTH_BINS), _HEALTH_BINS - 1)] += 1
        if rel < _TINY_RELATIVE:
            tiny += 1

    total_boxes = len(rows)

    classes = [
        {
            "id": c.id,
            "name": c.name,
            "color": c.color,
            "boxes": by_class_boxes.get(c.id, 0),
            "images": len(by_class_images.get(c.id, set())),
        }
        for c in categories
    ]

    # --- Named warnings, each one actionable -------------------------------
    warnings: list[str] = []

    unused = [c["name"] for c in classes if c["boxes"] == 0]
    if unused and total_boxes:
        warnings.append(
            f"{len(unused)} class(es) have no boxes at all ({', '.join(unused[:4])}"
            f"{'…' if len(unused) > 4 else ''}). A model can't learn a class it "
            "never sees — label some, or delete the class."
        )

    labelled = [c for c in classes if c["boxes"] > 0]
    if len(labelled) >= 2:
        top = max(labelled, key=lambda c: c["boxes"])
        bottom = min(labelled, key=lambda c: c["boxes"])
        if bottom["boxes"] and top["boxes"] / bottom["boxes"] > 10:
            warnings.append(
                f"Severe class imbalance: '{top['name']}' has {top['boxes']} boxes, "
                f"'{bottom['name']}' only {bottom['boxes']} ({top['boxes'] // bottom['boxes']}x). "
                f"Expect weak recall on '{bottom['name']}' — add examples of it "
                "before adding more of anything else."
            )

    if total_boxes and tiny / total_boxes > 0.2:
        warnings.append(
            f"{tiny} of {total_boxes} boxes ({tiny * 100 // total_boxes}%) span less "
            f"than ~3% of their image's width — around 19px at 640px training "
            "resolution. Boxes that small are genuinely hard to learn; consider a "
            "larger image size, or check whether they're annotation noise."
        )

    unannotated = total_images - len(annotated_images)
    if total_images and unannotated / total_images > 0.5 and total_boxes:
        warnings.append(
            f"{unannotated} of {total_images} images have no boxes. If they truly "
            "contain nothing, that's useful negative data — but if they're just "
            "unfinished, the model is training on half a dataset."
        )

    return {
        "total_images": total_images,
        "annotated_images": len(annotated_images),
        "total_boxes": total_boxes,
        "classes": classes,
        "box_sizes": {
            "small": small,
            "medium": medium,
            "large": large,
            "tiny": tiny,
            "relative_hist": hist,
        },
        "warnings": warnings,
    }


def _assign_splits(
    images: list[Image], train_pct: float, val_pct: float, test_pct: float
) -> None:
    """Randomly partition images by percentage.

    Shuffled with a FIXED seed. Two reasons that beats system entropy:
      1. Reproducibility — the same dataset splits the same way every run, so a
         model comparison isn't confounded by a different split.
      2. Debuggability — "why is image 47 in val?" has an answer.

    Shuffling at all matters: datasets arrive ordered (by capture time, by
    class, by folder). Slicing an ordered list puts every late-session image in
    val, and the resulting metric measures the wrong thing.
    """
    shuffled = list(images)
    random.Random(42).shuffle(shuffled)
    n = len(shuffled)

    # round(), NOT int(). int() truncates, which silently annihilates a small
    # split: 80/20 on 3 images gives int(3*0.2) == 0 val images — an empty
    # validation set, which is the exact failure this app warns about elsewhere.
    n_val = round(n * val_pct)
    n_test = round(n * test_pct)

    # Even round() gives 0 when n * pct < 0.5 (e.g. 20% of 2). If a split was
    # explicitly asked for and there are enough images to afford it, it gets at
    # least one — an empty val set is never what "20%" meant.
    if val_pct > 0 and n_val == 0 and n >= 2:
        n_val = 1
    if test_pct > 0 and n_test == 0 and n >= 3:
        n_test = 1

    # Train takes the remainder, so the three always sum to exactly n and no
    # image is left unassigned by a rounding artifact.
    n_train = max(0, n - n_val - n_test)

    for i, image in enumerate(shuffled):
        if i < n_train:
            image.split = Split.TRAIN
        elif i < n_train + n_val:
            image.split = Split.VAL
        else:
            image.split = Split.TEST


def _split_counts(db: Session, project_id: int) -> SplitCounts:
    counts = SplitCounts()
    for image in db.scalars(select(Image).where(Image.project_id == project_id)).all():
        setattr(counts, image.split, getattr(counts, image.split, 0) + 1)
    return counts


@router.post("/projects/{project_id}/dataset/split", response_model=SplitCounts)
def resplit(
    project_id: int, payload: SplitRequest, db: Session = Depends(get_db)
) -> SplitCounts:
    """Reassign train/val/test across the project by percentage.

    `only_train=True` handles the common import case: a dataset arrives with
    train/ and test/ but no valid/, so a validation set has to be carved out of
    train without touching test.
    """
    get_project_or_404(project_id, db)

    total = payload.train_pct + payload.val_pct + payload.test_pct
    if abs(total - 1.0) > 0.001:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Split percentages must sum to 1.0, got {total:.2f}",
        )

    query = select(Image).where(Image.project_id == project_id)
    if payload.only_train:
        query = query.where(Image.split == Split.TRAIN)

    images = list(db.scalars(query).all())
    if not images:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No images to split.")

    _assign_splits(images, payload.train_pct, payload.val_pct, payload.test_pct)
    db.commit()
    return _split_counts(db, project_id)


@router.patch("/images/{image_id}/split")
def set_split(image_id: int, split: str, db: Session = Depends(get_db)) -> dict:
    """Move one image to a different split."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")
    if split not in Split.ALL:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Invalid split {split!r}. Must be one of {Split.ALL}.",
        )
    image.split = split
    db.commit()
    return {"id": image.id, "split": image.split}


@router.post("/projects/{project_id}/dataset/split-selected", response_model=SplitCounts)
def set_split_bulk(
    project_id: int, payload: BulkSplitRequest, db: Session = Depends(get_db)
) -> SplitCounts:
    """Move a chosen set of images to one split.

    The manual counterpart to the percentage control: sometimes you know these
    specific twelve images belong in val, and a random shuffle is exactly the
    wrong tool.
    """
    get_project_or_404(project_id, db)
    if payload.split not in Split.ALL:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Invalid split {payload.split!r}. Must be one of {Split.ALL}.",
        )
    if not payload.image_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No images selected.")

    # Scoped to this project: an id from another project must not be moved just
    # because it was in the request body.
    images = list(
        db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.id.in_(payload.image_ids)
            )
        ).all()
    )
    for image in images:
        image.split = payload.split
    db.commit()
    return _split_counts(db, project_id)
