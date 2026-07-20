"""
DatasetVersion ORM model — an immutable, restorable snapshot of a dataset.

WHY VERSIONS EXIST
------------------
The dataset was previously only ever "now": uploading a new set of images or
deleting the wrong ones was unrecoverable. A version is a save point you can go
back to — the safety net for exactly that accident, and the thing that makes
"this model was trained on dataset v3" a fact rather than a guess.

WHEN ONE IS CREATED
-------------------
Only when the user clicks **Save dataset**. Not on every edit: a version per box
drawn would bury the handful that matter under hundreds that don't. That single
explicit gesture is also the gate into training — you train a SAVED dataset, so
every run points at something reproducible.

WHAT IT COSTS
-------------
Almost nothing. The snapshot is metadata only (which images, their splits, their
accepted boxes) written as JSON under storage/versions/. Image BYTES are never
copied. That's only safe because deleting an image now leaves its file on disk
(see services/storage.py::delete_image_file) — otherwise a version could
reference pictures that no longer exist.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 1-based per project — same reasoning as TrainingJob.version: the UI should
    # speak versions, not internal row ids.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    #: Optional user-given name. NULL means the version is shown by its number
    #: ("v3"). Renaming sets this; uniqueness is enforced against every other
    #: version's LABEL in the project — its name, or its "v{n}" default — so two
    #: versions can never present the same way in a list.
    name: Mapped[str | None] = mapped_column(String(120), default=None)

    #: Optional user note ("before removing blurry shots"). Auto-filled for
    #: safety snapshots taken before a restore.
    note: Mapped[str | None] = mapped_column(String(255), default=None)

    #: Path to the JSON snapshot on disk. Bytes-on-disk, path-in-DB — the same
    #: rule images follow; a snapshot can reach megabytes on a big project.
    snapshot_path: Mapped[str] = mapped_column(Text, nullable=False)

    #: Fingerprint of the dataset content (see DatasetSnapshot.content_hash).
    #: Identifies which version the LIVE dataset currently matches — the newest
    #: version is not necessarily the one you're looking at, and after a restore
    #: it usually isn't. NULL on versions saved before this existed.
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None, index=True)

    # Counts denormalised at save time so the version list renders without
    # opening (and parsing) every snapshot file.
    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_boxes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Boxes in the TRAIN split specifically — what a run actually learns from,
    #: so the train route can refuse an unlearnable version without parsing the
    #: snapshot file.
    train_boxes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_classes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship()  # noqa: F821

    def __repr__(self) -> str:
        return f"<DatasetVersion project={self.project_id} v{self.version}>"
