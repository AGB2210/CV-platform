"""
Model proposals: accept, reject, and batch-apply.

Auto-annotation writes PROPOSALS, not annotations. They are the model's
suggestions and are invisible to exports, training, and every count until a
human accepts them. This module is where that decision gets made.

The model proposes; the human disposes. Nothing you drew is ever destroyed to
make room for a suggestion — the old behaviour deleted your boxes and wrote the
model's in their place, which meant running the model cost you whatever was
already there.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image
from app.schemas.annotation import AnnotationRead

router = APIRouter(tags=["proposals"])


class ApplyMode:
    """How a proposal batch combines with your existing boxes, per image."""

    APPEND = "append"
    MERGE = "merge"
    REPLACE = "replace"


class ApplyProposals(BaseModel):
    mode: str = Field(default=ApplyMode.APPEND, pattern="^(append|merge|replace)$")


class ProposalPreview(BaseModel):
    """What applying the batch would do. Read-only.

    Split deliberately into mode-INDEPENDENT facts and mode-DEPENDENT outcomes.
    The UI describes all three modes at once while only one is selected, so it
    needs numbers that don't shift under it — a blurb that reads
    "would_delete_existing" from the *selected* mode's preview tells you Replace
    deletes nothing while Merge happens to be ticked.
    """

    # --- mode-independent ---
    proposed_boxes: int
    proposed_images: int
    existing_boxes: int
    #: Images that have BOTH proposals and existing boxes — the only images the
    #: three modes actually treat differently.
    conflicting_images: int
    #: Your boxes sitting on images this batch covers. This is exactly what
    #: Replace would delete, regardless of which mode is currently selected.
    existing_on_proposed_images: int

    # --- outcome for the requested mode ---
    would_accept: int
    would_discard: int
    would_delete_existing: int


def _proposal_rows(db: Session, project_id: int) -> list[Annotation]:
    return list(
        db.scalars(
            select(Annotation)
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.project_id == project_id, Annotation.proposed.is_(True))
        ).all()
    )


def _existing_by_image(db: Session, project_id: int) -> dict[int, list[Annotation]]:
    out: dict[int, list[Annotation]] = {}
    for ann in db.scalars(
        select(Annotation)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Annotation.proposed.is_(False))
    ).all():
        out.setdefault(ann.image_id, []).append(ann)
    return out


@router.get("/projects/{project_id}/proposals/preview", response_model=ProposalPreview)
def preview(
    project_id: int, mode: str = ApplyMode.APPEND, db: Session = Depends(get_db)
) -> ProposalPreview:
    """Numbers for the batch bar, before anything is applied."""
    get_project_or_404(project_id, db)

    proposals = _proposal_rows(db, project_id)
    existing = _existing_by_image(db, project_id)

    proposal_images = {a.image_id for a in proposals}
    conflicting = {i for i in proposal_images if existing.get(i)}
    existing_on_covered = sum(len(existing.get(i, [])) for i in proposal_images)

    if mode == ApplyMode.APPEND:
        would_accept = len(proposals)
        would_discard = 0
        would_delete = 0
    elif mode == ApplyMode.MERGE:
        # Per-image, not per-box: an image that already has boxes keeps them and
        # the proposals for it are dropped. No IoU maths, no surprises — you can
        # predict the outcome by looking at the grid.
        would_accept = sum(1 for a in proposals if a.image_id not in conflicting)
        would_discard = len(proposals) - would_accept
        would_delete = 0
    else:  # REPLACE
        would_accept = len(proposals)
        would_discard = 0
        # Only on images the batch actually covers. An image the model never
        # looked at keeps its boxes — otherwise "replace" would silently wipe
        # work on images that aren't even part of this run.
        would_delete = existing_on_covered

    return ProposalPreview(
        proposed_boxes=len(proposals),
        proposed_images=len(proposal_images),
        existing_boxes=sum(len(v) for v in existing.values()),
        conflicting_images=len(conflicting),
        existing_on_proposed_images=existing_on_covered,
        would_accept=would_accept,
        would_discard=would_discard,
        would_delete_existing=would_delete,
    )


@router.post("/projects/{project_id}/proposals/apply")
def apply_proposals(
    project_id: int, payload: ApplyProposals, db: Session = Depends(get_db)
) -> dict:
    """Apply the whole proposal batch.

      append   accept every proposal alongside your boxes.
      merge    per image: if it already has boxes, keep yours and drop the
               proposals; if it has none, accept them.
      replace  proposals win on the images they cover; your boxes on those
               images are deleted. Images the batch didn't touch are untouched.
    """
    get_project_or_404(project_id, db)

    proposals = _proposal_rows(db, project_id)
    if not proposals:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "There are no pending proposals to apply."
        )

    existing = _existing_by_image(db, project_id)
    proposal_images = {a.image_id for a in proposals}
    conflicting = {i for i in proposal_images if existing.get(i)}

    accepted = discarded = deleted = 0

    if payload.mode == ApplyMode.REPLACE:
        for image_id in proposal_images:
            for ann in existing.get(image_id, []):
                db.delete(ann)
                deleted += 1

    for ann in proposals:
        if payload.mode == ApplyMode.MERGE and ann.image_id in conflicting:
            db.delete(ann)
            discarded += 1
            continue
        # Accepting is a state change, not a copy: the row stays put and simply
        # stops being a proposal. Its id, geometry and confidence are unchanged,
        # so nothing that already references it breaks.
        ann.proposed = False
        # A human said yes. That's a review — the whole point of the gesture.
        ann.reviewed = True
        accepted += 1

    db.commit()
    return {
        "mode": payload.mode,
        "accepted": accepted,
        "discarded": discarded,
        "deleted_existing": deleted,
    }


@router.delete("/projects/{project_id}/proposals", status_code=status.HTTP_204_NO_CONTENT)
def discard_all(project_id: int, db: Session = Depends(get_db)) -> None:
    """Throw the batch away. Your own boxes are untouched by definition."""
    get_project_or_404(project_id, db)
    for ann in _proposal_rows(db, project_id):
        db.delete(ann)
    db.commit()


@router.post("/annotations/{annotation_id}/accept", response_model=AnnotationRead)
def accept_one(annotation_id: int, db: Session = Depends(get_db)) -> Annotation:
    """Accept a single proposal — the per-box gesture in the canvas."""
    ann = db.get(Annotation, annotation_id)
    if ann is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Annotation {annotation_id} not found"
        )
    if not ann.proposed:
        # Not an error worth failing on: the box is already accepted, which is
        # the state the caller wanted. Idempotent beats pedantic.
        return ann
    ann.proposed = False
    ann.reviewed = True
    db.commit()
    db.refresh(ann)
    return ann


@router.post("/images/{image_id}/proposals/accept", response_model=list[AnnotationRead])
def accept_image(image_id: int, db: Session = Depends(get_db)) -> list[Annotation]:
    """Accept every proposal on one image — the per-image bulk gesture.

    The common case by far: the model got this image right, and clicking each of
    its six boxes to say so is busywork.
    """
    if db.get(Image, image_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    rows = list(
        db.scalars(
            select(Annotation).where(
                Annotation.image_id == image_id, Annotation.proposed.is_(True)
            )
        ).all()
    )
    for ann in rows:
        ann.proposed = False
        ann.reviewed = True
    db.commit()
    for ann in rows:
        db.refresh(ann)
    return rows


@router.delete(
    "/images/{image_id}/proposals", status_code=status.HTTP_204_NO_CONTENT
)
def reject_image(image_id: int, db: Session = Depends(get_db)) -> None:
    """Reject every proposal on one image."""
    if db.get(Image, image_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")
    for ann in db.scalars(
        select(Annotation).where(
            Annotation.image_id == image_id, Annotation.proposed.is_(True)
        )
    ).all():
        db.delete(ann)
    db.commit()


@router.get("/projects/{project_id}/proposals/count")
def count(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Cheap poll for "is there a pending batch?" without pulling every row."""
    get_project_or_404(project_id, db)
    n = db.scalar(
        select(func.count(Annotation.id))
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Annotation.proposed.is_(True))
    ) or 0
    return {"proposed_boxes": n}
