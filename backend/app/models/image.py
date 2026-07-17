"""Image ORM model — metadata for one uploaded image. The bytes live on disk."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Split:
    """Which partition an image belongs to.

    Plain string constants rather than a DB Enum — same reason as everywhere
    else in this project: SQLite cannot ALTER a CHECK constraint, so a DB-level
    enum would make adding a value a table rebuild.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    ALL = (TRAIN, VAL, TEST)

    # Folder names seen in the wild, mapped to our vocabulary. Roboflow exports
    # "valid"; the COCO convention is "val2017"; plenty of people write "validation".
    # Normalising here means the importer doesn't care which flavour it meets.
    FOLDER_ALIASES = {
        "train": TRAIN,
        "training": TRAIN,
        "train2017": TRAIN,
        "val": VAL,
        "valid": VAL,
        "validation": VAL,
        "val2017": VAL,
        "test": TEST,
        "testing": TEST,
        "test2017": TEST,
    }

    @classmethod
    def from_folder(cls, name: str) -> str | None:
        """Map a directory name to a split, or None if it isn't one."""
        return cls.FOLDER_ALIASES.get(name.strip().lower())


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

    # --- Dataset lifecycle -------------------------------------------------
    #
    # Mirrors Roboflow's two-stage model, and it exists for a real reason:
    # half-annotated images must not silently end up in a training run.
    #
    #   in_dataset=False  "staging"  — uploaded, being annotated/reviewed
    #   in_dataset=True   "dataset"  — approved and trainable; exports read only these
    #
    # An imported COCO/Roboflow dataset skips staging entirely: it is already
    # labelled ground truth, so there is nothing to review.
    in_dataset: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    # train | val | test. Defaults to train so a plain upload is never blocked
    # on a decision the user hasn't been asked to make yet; the split control
    # reassigns later. An import overrides this from the folder name.
    split: Mapped[str] = mapped_column(
        String(16), nullable=False, default=Split.TRAIN, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    project: Mapped["Project"] = relationship(back_populates="images")  # noqa: F821

    # delete-orphan: removing an image takes its boxes with it. An annotation
    # pointing at a deleted image is unexportable and unrenderable.
    annotations: Mapped[list["Annotation"]] = relationship(  # noqa: F821
        back_populates="image", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Image id={self.id} original={self.original_filename!r}>"
