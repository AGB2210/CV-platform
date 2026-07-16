"""Image ORM model — metadata for one uploaded image. The bytes live on disk."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # The name on disk: a generated UUID + extension, e.g. "3f2b...c1.jpg".
    # NOT the name the user uploaded. Three reasons this matters:
    #   1. Collisions — everyone's phone produces IMG_0001.jpg.
    #   2. Path traversal — a filename like "../../app/main.py" is an attack.
    #      We never trust user-supplied names as paths.
    #   3. Portability — user filenames carry spaces, unicode, and characters
    #      Windows rejects outright (: * ? " < > |).
    filename: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # What the user actually called it. Display only — never used to build a
    # path. Kept so the grid shows something recognisable.
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

    # Cached from the file at upload time. Denormalised deliberately: the
    # annotation canvas needs dimensions for EVERY image in a grid to lay out
    # boxes, and opening hundreds of files per page render would be absurd when
    # the value never changes.
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="images")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Image id={self.id} original={self.original_filename!r}>"
