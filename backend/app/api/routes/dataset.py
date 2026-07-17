"""
Staging -> dataset lifecycle, bulk approval, and train/val/test splits.

The two-stage model (see models/image.py) exists so that half-annotated images
can't drift into a training run. This module owns the transition.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image
from app.models.image import Split
from app.schemas.dataset import (
    CommitMode,
    CommitPreview,
    DatasetCommit,
    DatasetStats,
    SplitCounts,
    SplitRequest,
)

router = APIRouter(tags=["dataset"])


def _approved_image_ids(db: Session, project_id: int, staged_only: bool = True) -> set[int]:
    """Images whose every box is reviewed, and which have at least one box.

    "At least one" matters: an image with zero annotations trivially satisfies
    "all boxes reviewed", and quietly promoting empty images into the dataset
    would train the model that these scenes contain nothing. That may be true —
    negative examples are legitimate — but it must be a deliberate choice, not
    an accident of vacuous truth.
    """
    query = (
        select(
            Annotation.image_id,
            func.count(Annotation.id).label("n"),
            func.sum(func.cast(Annotation.reviewed, __import__("sqlalchemy").Integer)).label("r"),
        )
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
        .group_by(Annotation.image_id)
    )
    if staged_only:
        query = query.where(Image.in_dataset.is_(False))

    return {row[0] for row in db.execute(query).all() if row[1] > 0 and row[1] == row[2]}


@router.get("/projects/{project_id}/dataset/stats", response_model=DatasetStats)
def dataset_stats(project_id: int, db: Session = Depends(get_db)) -> DatasetStats:
    """Counts for the Dataset page header."""
    get_project_or_404(project_id, db)

    images = list(db.scalars(select(Image).where(Image.project_id == project_id)).all())
    staging = [i for i in images if not i.in_dataset]
    committed = [i for i in images if i.in_dataset]

    annotated_ids = {
        row[0]
        for row in db.execute(
            select(Annotation.image_id)
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.project_id == project_id)
            .distinct()
        ).all()
    }
    approved = _approved_image_ids(db, project_id)

    boxes = db.execute(
        select(
            func.count(Annotation.id),
            func.sum(func.cast(Annotation.reviewed, __import__("sqlalchemy").Integer)),
        )
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id)
    ).one()

    counts = SplitCounts()
    for image in committed:
        setattr(counts, image.split, getattr(counts, image.split, 0) + 1)

    return DatasetStats(
        staging_total=len(staging),
        staging_annotated=sum(1 for i in staging if i.id in annotated_ids),
        staging_approved=len(approved),
        dataset_total=len(committed),
        splits=counts,
        total_boxes=boxes[0] or 0,
        reviewed_boxes=boxes[1] or 0,
    )


@router.post("/projects/{project_id}/annotations/approve-all")
def approve_all(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Mark every box in the project reviewed.

    The bulk escape hatch. Reviewing 500 images one Enter at a time is the
    correct default, but when you have already eyeballed a grid and the model
    plainly nailed it, making you press Enter 500 times is theatre rather than
    diligence.

    A single UPDATE ... WHERE rather than loading rows and setting a field:
    at 50k boxes the ORM round-trip is the difference between instant and a
    coffee break.
    """
    get_project_or_404(project_id, db)

    image_ids = select(Image.id).where(Image.project_id == project_id)
    result = db.execute(
        Annotation.__table__.update()
        .where(Annotation.image_id.in_(image_ids))
        .where(Annotation.reviewed.is_(False))
        .values(reviewed=True)
    )
    db.commit()
    return {"approved": result.rowcount or 0}


@router.get("/projects/{project_id}/dataset/preview", response_model=CommitPreview)
def commit_preview(
    project_id: int, mode: str = CommitMode.APPEND, db: Session = Depends(get_db)
) -> CommitPreview:
    """What a commit in this mode would do. Read-only.

    Exists because `replace` deletes data. Showing the numbers before the click
    is the difference between an informed decision and a bug report.
    """
    get_project_or_404(project_id, db)

    staged = list(
        db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.in_dataset.is_(False)
            )
        ).all()
    )
    current = db.scalar(
        select(func.count(Image.id)).where(
            Image.project_id == project_id, Image.in_dataset.is_(True)
        )
    ) or 0

    approved = _approved_image_ids(db, project_id)
    would_add = sum(1 for i in staged if i.id in approved)
    would_remove = current if mode == CommitMode.REPLACE else 0

    return CommitPreview(
        staged_total=len(staged),
        staged_approved=would_add,
        staged_unapproved=len(staged) - would_add,
        dataset_current=current,
        would_add=would_add,
        would_remove=would_remove,
        dataset_after=current - would_remove + would_add,
    )


@router.post("/projects/{project_id}/dataset/commit")
def commit_to_dataset(
    project_id: int, payload: DatasetCommit, db: Session = Depends(get_db)
) -> dict:
    """Move approved staging images into the trainable dataset.

    Modes:
      append  — add them; existing dataset untouched.
      merge   — same, but a staged image whose filename already exists in the
                dataset folds its boxes into that image and the duplicate row is
                dropped. This is the "I re-uploaded a corrected batch" case.
      replace — the staged images become the whole dataset; images currently in
                it are DELETED. Destructive, hence /preview.

    Only APPROVED images move. An image with an unreviewed box is, by
    definition, not something you've said is correct — and the entire point of
    the two-stage model is that "not yet checked" can't leak into training.
    """
    get_project_or_404(project_id, db)

    total = payload.train_pct + payload.val_pct + payload.test_pct
    if payload.assign_splits and abs(total - 1.0) > 0.001:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Split percentages must sum to 1.0, got {total:.2f}",
        )

    approved = _approved_image_ids(db, project_id)
    staged = [
        i
        for i in db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.in_dataset.is_(False)
            )
        ).all()
        if i.id in approved
    ]

    if not staged:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No approved images to commit. Review and approve images first.",
        )

    removed = 0
    merged = 0

    if payload.mode == CommitMode.REPLACE:
        for image in db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.in_dataset.is_(True)
            )
        ).all():
            db.delete(image)
            removed += 1

    elif payload.mode == CommitMode.MERGE:
        by_name = {
            i.original_filename: i
            for i in db.scalars(
                select(Image).where(
                    Image.project_id == project_id, Image.in_dataset.is_(True)
                )
            ).all()
        }
        survivors = []
        for image in staged:
            existing = by_name.get(image.original_filename)
            if existing is None:
                survivors.append(image)
                continue
            # Re-point this image's boxes at the dataset copy, then drop the
            # duplicate row. Cheaper and safer than copying box values, and it
            # keeps annotation ids stable.
            for ann in list(image.annotations):
                ann.image_id = existing.id
            db.flush()
            db.delete(image)
            merged += 1
        staged = survivors

    if payload.assign_splits and staged:
        _assign_splits(staged, payload.train_pct, payload.val_pct, payload.test_pct)

    for image in staged:
        image.in_dataset = True

    db.commit()
    return {
        "committed": len(staged),
        "merged": merged,
        "removed": removed,
        "mode": payload.mode,
    }


def _assign_splits(
    images: list[Image], train_pct: float, val_pct: float, test_pct: float
) -> None:
    """Randomly partition images by percentage.

    Shuffled with a FIXED seed. Two reasons that's the right call over
    `random.shuffle` with system entropy:
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
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    # Any rounding remainder goes to train — the split with the most to gain
    # from an extra sample, and never leaving val/test accidentally empty.
    for i, image in enumerate(shuffled):
        if i < n_train:
            image.split = Split.TRAIN
        elif i < n_train + n_val:
            image.split = Split.VAL
        elif test_pct > 0:
            image.split = Split.TEST
        else:
            image.split = Split.TRAIN


@router.post("/projects/{project_id}/dataset/split", response_model=SplitCounts)
def resplit(
    project_id: int, payload: SplitRequest, db: Session = Depends(get_db)
) -> SplitCounts:
    """Reassign train/val/test across the dataset by percentage.

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

    query = select(Image).where(
        Image.project_id == project_id, Image.in_dataset.is_(True)
    )
    if payload.only_train:
        query = query.where(Image.split == Split.TRAIN)

    images = list(db.scalars(query).all())
    if not images:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "No dataset images to split."
        )

    _assign_splits(images, payload.train_pct, payload.val_pct, payload.test_pct)
    db.commit()

    counts = SplitCounts()
    for image in db.scalars(
        select(Image).where(Image.project_id == project_id, Image.in_dataset.is_(True))
    ).all():
        setattr(counts, image.split, getattr(counts, image.split, 0) + 1)
    return counts


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
