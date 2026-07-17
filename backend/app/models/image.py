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

    # --- Split --------------------------------------------------------------
    #
    # train | val | test. Defaults to train so an upload is never blocked on a
    # decision nobody has been asked to make yet; the split control on the
    # Dataset page reassigns later. An import overrides this from its folder.
    split: Mapped[str] = mapped_column(
        String(16), nullable=False, default=Split.TRAIN, index=True
    )

    # NOTE: `in_dataset` is gone.
    #
    # It carried Roboflow's two-stage model — uploads landed in "staging" and a
    # commit step moved approved images into the trainable dataset, to stop
    # half-annotated work reaching a training run.
    #
    # The proposal model already does that job better: the model's output isn't
    # part of your annotations until you accept it, so nothing unreviewed can
    # leak in regardless. What staging added on top was a second gate that
    # blocked you from using work you'd already accepted, and a commit dialog
    # asking a question (merge/append/replace) whose answer was always "yes,
    # obviously". Accepting IS the commit.
    #
    # The column still exists in the database — SQLite can't drop one without
    # rebuilding the table, and it's harmless: nothing selects it, and its
    # NOT NULL DEFAULT 0 keeps old rows and new inserts valid.
    #
    # Consequence worth knowing: an image with no boxes now exports as a
    # negative example rather than being held back. That's usually correct, but
    # it IS a behaviour change from "staging protects you".

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
