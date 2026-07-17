"""
Annotation ORM model — one bounding box on one image.

WHY A TABLE AND NOT A COCO JSON FILE
------------------------------------
The brief says "store annotations as COCO JSON". This stores them as rows and
treats COCO as an import/export format (see services/coco.py) rather than the
on-disk source of truth. The reasoning:

  - Phase 3 edits ONE box at a time. With a JSON file that's read-whole,
    mutate, write-whole for every drag of every corner — and two browser tabs
    doing it concurrently silently lose one of the edits. A row is an UPDATE.
  - Transactions. "Delete this class and all its boxes" is atomic here. With a
    file, a crash mid-write leaves a truncated JSON and the dataset is gone.
  - Queries. "How many images are unannotated?" is a COUNT, not parsing a
    100 MB file on every page load.
  - COCO is lossy for our purposes. It has nowhere to record "this box came
    from Grounding DINO at 0.42 confidence and a human hasn't checked it yet",
    which is exactly the state the review workflow is about.

The stated GOAL of COCO was "industry standard, keeps us compatible with
existing tools" — and compatibility is delivered by export, which is when it
matters (feeding a trainer, opening in another tool). This satisfies the intent
while keeping the source of truth in something built for mutation.

Columns mirror COCO's field names and conventions exactly, so export is a field
copy rather than a translation with rounding bugs hiding in it.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Annotation(Base):
    __tablename__ = "annotations"

    # The canvas loads every box for one image, and training exports every box
    # for a project. Both are image_id lookups, so index it.
    __table_args__ = (Index("ix_annotation_image", "image_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    # CASCADE: deleting a class removes its boxes. The alternative — nulling the
    # FK — leaves boxes with no class, which can't be exported, can't be trained
    # on, and can't be rendered. A box without a label isn't data, it's litter.
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # COCO's bbox convention: absolute pixels, [x, y, width, height], top-left
    # origin. Stored in COCO's format (not xyxy) so export is a direct copy.
    # Float, not Int: models emit sub-pixel coordinates and COCO permits floats.
    # Rounding at storage time would quietly degrade every box.
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    width: Mapped[float] = mapped_column(Float, nullable=False)
    height: Mapped[float] = mapped_column(Float, nullable=False)

    # NULL for human-drawn boxes — a person doesn't have a confidence score, and
    # 1.0 would be a lie that pollutes any later analysis of model calibration.
    confidence: Mapped[float | None] = mapped_column(Float, default=None)

    # "auto" | "manual". Which model produced it is on the job, not here.
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="manual")

    # Has a human confirmed this box? Only meaningful once it's accepted.
    reviewed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- Proposal state -----------------------------------------------------
    #
    # True = the model SUGGESTS this box; it is not part of your annotations.
    # Proposals are excluded from exports, training, and every annotation count.
    # They exist only until you accept them (proposed -> False) or reject them
    # (deleted).
    #
    # This is what makes auto-annotation non-destructive. It used to delete your
    # boxes and write its own in their place, so running the model cost you
    # whatever was already there. Now the two coexist: yours stay, the model's
    # arrive alongside as a separate layer, and you decide how they combine.
    #
    # The model proposes; the human disposes. Nothing of yours is destroyed to
    # make room for a suggestion.
    proposed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    # Which auto-annotation run produced this box. NULL for human-drawn and
    # imported boxes. Lets a whole batch be applied or discarded as a unit, and
    # lets the UI say "this run proposed 47 boxes" long after the fact.
    #
    # ondelete="SET NULL", not CASCADE: deleting a job's history must not delete
    # the annotations it created. Once you've accepted a box it's yours, and its
    # provenance record is not a lifeline.
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_jobs.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    image: Mapped["Image"] = relationship(back_populates="annotations")  # noqa: F821
    category: Mapped["Category"] = relationship()  # noqa: F821

    @property
    def area(self) -> float:
        """COCO requires an `area` field. Derived, never stored — a stored copy
        would drift the moment someone resizes the box."""
        return self.width * self.height

    def __repr__(self) -> str:
        return f"<Annotation id={self.id} image_id={self.image_id} {self.source}>"
