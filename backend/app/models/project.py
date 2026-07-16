"""Project ORM model — the top-level container for a dataset."""

from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.enums import TaskType


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)

    # Plain String, not sa.Enum — see the long note in app/enums.py. Short
    # version: SQLite can't ALTER a CHECK constraint, so a DB-level enum would
    # make adding "segmentation" a real migration. Validation lives in Pydantic.
    task_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TaskType.OBJECT_DETECTION.value
    )

    # server_default=func.now() emits SQLite's CURRENT_TIMESTAMP, so the DB
    # stamps the row even if a future caller (a script, a background job)
    # inserts without going through the API.
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    # onupdate is applied by SQLAlchemy on UPDATE, not by the DB.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # cascade="all, delete-orphan" is the ORM half of cascading deletes: when a
    # Project is deleted through a Session, SQLAlchemy deletes its children too.
    # The DB half (ondelete="CASCADE") lives on the child's ForeignKey, and only
    # actually fires because we turn on `PRAGMA foreign_keys` in database.py —
    # SQLite ignores foreign keys entirely by default.
    images: Mapped[list["Image"]] = relationship(  # noqa: F821
        back_populates="project", cascade="all, delete-orphan"
    )
    categories: Mapped[list["Category"]] = relationship(  # noqa: F821
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Category.id",
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r}>"
