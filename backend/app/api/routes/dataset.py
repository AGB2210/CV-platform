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
from app.models import Annotation, Image
from app.models.image import Split
from app.schemas.dataset import (
    BulkSplitRequest,
    DatasetStats,
    SplitCounts,
    SplitRequest,
)

router = APIRouter(tags=["dataset"])


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
