"""
Model proposals: accept or reject.

Auto-annotation writes PROPOSALS, not annotations. They are excluded from
exports, training, and every count until a decision is made:

    Accept  the model's boxes become your annotations. Your previous boxes ON
            THE IMAGES THIS RUN COVERED are deleted — you asked for the model's
            output, so that's what you get. Images the run never looked at keep
            their annotations.
    Reject  the proposals are thrown away and your boxes stay exactly as they
            were.

There are deliberately no other options. Anything that kept BOTH sets would
accumulate duplicates across runs, and "keep mine" is what reject already means.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image

router = APIRouter(tags=["proposals"])


class ProposalPreview(BaseModel):
    """What accepting would do. Read-only."""

    proposed_boxes: int
    proposed_images: int
    #: Your boxes on the images this run covered — exactly what Accept deletes.
    existing_on_proposed_images: int
    #: Your boxes on images the run never touched. Accept leaves these alone.
    existing_elsewhere: int


def _proposals(db: Session, project_id: int) -> list[Annotation]:
    return list(
        db.scalars(
            select(Annotation)
            .join(Image, Image.id == Annotation.image_id)
            .where(Image.project_id == project_id, Annotation.proposed.is_(True))
        ).all()
    )


def _accepted_by_image(db: Session, project_id: int) -> dict[int, list[Annotation]]:
    out: dict[int, list[Annotation]] = {}
    for ann in db.scalars(
        select(Annotation)
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Annotation.proposed.is_(False))
    ).all():
        out.setdefault(ann.image_id, []).append(ann)
    return out


@router.get("/projects/{project_id}/proposals/preview", response_model=ProposalPreview)
def preview(project_id: int, db: Session = Depends(get_db)) -> ProposalPreview:
    """Numbers for the bar. Accepting deletes boxes, so they belong on screen
    before the click, not after."""
    get_project_or_404(project_id, db)

    proposals = _proposals(db, project_id)
    accepted = _accepted_by_image(db, project_id)
    covered = {a.image_id for a in proposals}

    return ProposalPreview(
        proposed_boxes=len(proposals),
        proposed_images=len(covered),
        existing_on_proposed_images=sum(len(accepted.get(i, [])) for i in covered),
        existing_elsewhere=sum(
            len(v) for k, v in accepted.items() if k not in covered
        ),
    )


def _accept(db: Session, proposals: list[Annotation], accepted_by_image: dict) -> dict:
    """Replace the covered images' boxes with the proposals. Shared by both
    accept endpoints so project-wide and per-image can't drift apart."""
    covered = {a.image_id for a in proposals}

    deleted = 0
    for image_id in covered:
        for ann in accepted_by_image.get(image_id, []):
            db.delete(ann)
            deleted += 1

    for ann in proposals:
        # A state change, not a copy: the row stays put and stops being a
        # proposal. Its id, geometry and confidence are untouched.
        ann.proposed = False
        ann.reviewed = True

    db.commit()
    return {"accepted": len(proposals), "deleted_existing": deleted}


@router.post("/projects/{project_id}/proposals/accept")
def accept_all(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Accept the batch: the model's boxes replace yours on the images it covered.

    Images the run never looked at keep their annotations — "accept the model's
    output" means for what it actually looked at, not a project-wide wipe.
    """
    get_project_or_404(project_id, db)

    proposals = _proposals(db, project_id)
    if not proposals:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "There are no pending proposals to accept."
        )
    return _accept(db, proposals, _accepted_by_image(db, project_id))


@router.delete("/projects/{project_id}/proposals", status_code=status.HTTP_204_NO_CONTENT)
def reject_all(project_id: int, db: Session = Depends(get_db)) -> None:
    """Reject the batch. Your boxes are untouched — nothing of yours was ever
    modified, so there is nothing to restore."""
    get_project_or_404(project_id, db)
    for ann in _proposals(db, project_id):
        db.delete(ann)
    db.commit()


@router.post("/images/{image_id}/proposals/accept")
def accept_image(image_id: int, db: Session = Depends(get_db)) -> dict:
    """Accept one image's proposals, replacing that image's boxes."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    proposals = list(
        db.scalars(
            select(Annotation).where(
                Annotation.image_id == image_id, Annotation.proposed.is_(True)
            )
        ).all()
    )
    if not proposals:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "This image has no pending proposals."
        )

    accepted = {
        image_id: list(
            db.scalars(
                select(Annotation).where(
                    Annotation.image_id == image_id, Annotation.proposed.is_(False)
                )
            ).all()
        )
    }
    return _accept(db, proposals, accepted)


@router.delete("/images/{image_id}/proposals", status_code=status.HTTP_204_NO_CONTENT)
def reject_image(image_id: int, db: Session = Depends(get_db)) -> None:
    """Reject one image's proposals. Its boxes stay as they were."""
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
    """Cheap poll for "is there a pending batch?"."""
    get_project_or_404(project_id, db)
    n = db.scalar(
        select(func.count(Annotation.id))
        .join(Image, Image.id == Annotation.image_id)
        .where(Image.project_id == project_id, Annotation.proposed.is_(True))
    ) or 0
    return {"proposed_boxes": n}
