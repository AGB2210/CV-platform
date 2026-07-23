"""
Imported checkpoint files — external weights a training run can start from.

The existing "Initialize from" options were pretrained bases and previous runs
of THIS app. This table adds the third source: a checkpoint trained elsewhere
(another machine, another tool, a colleague's run) uploaded as a file.

Metadata only, like every other table: the bytes live under
storage/imported_weights/<project_id>/, addressed by content hash so uploading
the same file twice stores it once. The DB row is what the UI lists and what a
TrainingJob references — delete semantics and provenance both hang off it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ImportedWeights(Base):
    __tablename__ = "imported_weights"

    id: Mapped[int] = mapped_column(primary_key=True)

    #: Scoped to a project like everything else in the app. Weights are not
    #: intrinsically project-bound, but a global pool would be the only global
    #: thing in the UI — and the file is small enough to upload twice.
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )

    #: The name the file arrived with — what the picker shows. Not used for
    #: identity or storage (two different files may share a name).
    filename: Mapped[str] = mapped_column(String(255))

    #: Where the bytes live, relative to storage/ (see config.to_storage_path) —
    #: relative so the project folder stays movable.
    stored_path: Mapped[str] = mapped_column(String(500))

    size_bytes: Mapped[int] = mapped_column(Integer)

    #: SHA-256 of the bytes — the same dedupe identity images use. Re-uploading
    #: an already-imported file returns the existing row instead of a copy.
    sha256: Mapped[str] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
