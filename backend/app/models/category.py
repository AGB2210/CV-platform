"""
Category ORM model — a class label such as "car" or "person".

NAMING: the UI calls these "classes" (Roboflow's term, and what you'd say out
loud), but the model is `Category` for two reasons:

  1. `class` is a Python keyword — `class Class(Base)` is a syntax error, and
     every workaround (`Klass`, `ClassModel`) is uglier than just using COCO's
     word.
  2. COCO JSON — which this project stores annotations in — calls them
     `categories`. Matching that name means export is a direct field mapping
     rather than a translation layer where mistakes hide.

So: "class" in the UI, `Category` in the code and the DB. This docstring is the
one place that mapping is written down.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Category(Base):
    __tablename__ = "categories"

    # A class name must be unique WITHIN a project, but "car" can obviously
    # exist in many projects. A composite unique constraint says exactly that,
    # and lets the database reject duplicates even if a bug (or a concurrent
    # request) slips past the application's own check.
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_category_project_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    # ondelete="CASCADE" is the DATABASE-level rule: delete the project row and
    # SQLite removes these too. Requires PRAGMA foreign_keys=ON (see database.py)
    # — without it SQLite silently ignores this and you get orphaned rows.
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    name: Mapped[str] = mapped_column(String(80), nullable=False)

    # Hex colour used to draw this class's boxes on the annotation canvas
    # (Phase 3). Stored per class rather than derived from the class's position
    # in a list, so a class keeps its colour when others are deleted — otherwise
    # every box on screen would change colour when you remove an unrelated class.
    color: Mapped[str] = mapped_column(String(9), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="categories")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Category id={self.id} name={self.name!r} project_id={self.project_id}>"
