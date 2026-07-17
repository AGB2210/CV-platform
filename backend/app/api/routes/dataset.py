"""
Staging -> dataset lifecycle, bulk approval, and train/val/test splits.

The two-stage model (see models/image.py) exists so that half-annotated images
can't drift into a training run. This module owns the transition.
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
from app.services import storage
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
        # Proposals don't count toward approval either way: they can't make an
        # image approved, and an un-actioned proposal must not BLOCK an image
        # whose real boxes are all fine. Pending suggestions are a separate
        # decision from "are these annotations correct".
        .where(Annotation.proposed.is_(False))
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
            .where(Image.project_id == project_id, Annotation.proposed.is_(False))
            .distinct()
        ).all()
    }
    approved = _approved_image_ids(db, project_id)

    Integer = __import__("sqlalchemy").Integer
    boxes = db.execute(
        select(
            # Every count here is of ACCEPTED boxes. Proposals get their own
            # figure rather than being folded in — "you have 90 boxes" is a lie
            # if 60 of them are suggestions nobody has looked at.
            func.sum(case((Annotation.proposed.is_(False), 1), else_=0)),
            func.sum(
                case(
                    ((Annotation.reviewed.is_(True)) & (Annotation.proposed.is_(False)), 1),
                    else_=0,
                )
            ),
            func.sum(case((Annotation.proposed.is_(True), 1), else_=0)),
            func.count(func.distinct(case((Annotation.proposed.is_(True), Annotation.image_id)))),
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
        proposed_boxes=boxes[2] or 0,
        proposed_images=boxes[3] or 0,
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
        # Never silently accept pending proposals. "Approve all" means "the
        # annotations I have are correct", not "take everything the model
        # guessed" — that's a different decision with its own dialog.
        .where(Annotation.proposed.is_(False))
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
    incoming = [i for i in staged if i.id in approved]
    would_add = len(incoming)

    if mode == CommitMode.REPLACE:
        would_remove = current
    elif mode == CommitMode.MERGE:
        # Merge deletes too — just only the images it supersedes by name. That
        # number has to be on screen before the click, or "merge" reads as
        # harmless and quietly removes data.
        dataset_names = {
            i.original_filename
            for i in db.scalars(
                select(Image).where(
                    Image.project_id == project_id, Image.in_dataset.is_(True)
                )
            ).all()
        }
        would_remove = sum(
            1 for i in incoming if i.original_filename in dataset_names
        )
    else:
        would_remove = 0

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
      append  — add them; existing dataset untouched. A duplicate filename
                simply means two images with the same name.
      merge   — upsert by filename. A staged image whose name already exists in
                the dataset SUPERSEDES that image: the old one and its boxes are
                deleted. Names that don't collide are appended. This is the
                "I re-uploaded a corrected batch" case.
      replace — the staged images become the whole dataset; everything currently
                in it is DELETED, collision or not.

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
        # Upsert by filename: the incoming version WINS, the old one goes.
        #
        # A filename collision means the same picture arrived twice, and the one
        # you just reviewed and approved is the one you meant. Keeping the old
        # copy and folding the new boxes into it (the previous behaviour) left
        # you with BOTH sets of boxes on one image — the stale ones you were
        # trying to replace, plus the new ones. That's not a merge, it's a pile.
        #
        # Scoped to name collisions, which is the whole difference from
        # `replace`: dataset images that this batch doesn't mention are left
        # completely alone.
        by_name = {
            i.original_filename: i
            for i in db.scalars(
                select(Image).where(
                    Image.project_id == project_id, Image.in_dataset.is_(True)
                )
            ).all()
        }
        for image in staged:
            superseded = by_name.get(image.original_filename)
            if superseded is None:
                continue  # no collision — plain append
            # Cascade takes the old image's annotations with it, and
            # delete_image_file removes the bytes from disk so a superseded
            # copy doesn't sit there forever.
            project_id_, filename_ = superseded.project_id, superseded.filename
            db.delete(superseded)
            db.flush()
            storage.delete_image_file(project_id_, filename_)
            merged += 1

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

    # round(), NOT int().
    #
    # int() truncates, which silently annihilates a small split: asking for
    # 80/20 on 3 images gives int(3*0.2) == 0 val images — an empty validation
    # set, which is the exact failure this app warns users about elsewhere. It
    # only shows up on small datasets, so it would survive every test on a
    # realistic one and then bite someone trying the tool with 5 images.
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
