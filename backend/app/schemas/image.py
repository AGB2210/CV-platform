"""Pydantic schemas for images."""

from pydantic import BaseModel, ConfigDict, computed_field

from app.timestamps import UtcDatetime


class ImageRead(BaseModel):
    """Response body for one image."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    filename: str
    original_filename: str
    width: int
    height: int
    size_bytes: int
    created_at: UtcDatetime

    #: train | val | test
    split: str = "train"

    # Populated by the list route via a GROUP BY join, not a per-image COUNT.
    # Lets the grid show annotation state at a glance — which is the difference
    # between "you have 500 images" and "you have 500 images, 340 annotated,
    # 120 of those still unreviewed".
    #: ACCEPTED boxes only. Proposals are excluded — one isn't an annotation
    #: until you accept it, and counting them would make an untouched image
    #: look done.
    annotation_count: int = 0
    reviewed_count: int = 0
    #: Pending model suggestions awaiting accept/reject.
    proposed_count: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """Browser-facing URL for the image bytes.

        Derived here rather than stored, because it's a function of where the
        API chooses to mount static files — a deployment concern that has no
        business being frozen into a database row. If the mount path changes,
        this one line changes and every existing row is still correct.

        Relative on purpose: the frontend is proxied (dev) or same-origin
        (prod), so a bare path works in both without knowing the hostname.
        """
        return f"/static/images/{self.project_id}/{self.filename}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def thumb_url(self) -> str:
        """Small cached JPEG for grids and filmstrips.

        Grids used to render the full originals into 150px cells — the cause
        of the 200-image scroll lag. Anything showing many images at once
        should use this; anything showing ONE image to work on (the canvas,
        a lightbox) keeps `url`.
        """
        return f"/api/thumbs/{self.project_id}/{self.filename}"


class UploadResult(BaseModel):
    """Response body for an upload.

    Uploads are partial-success by nature: dropping 50 files where 2 are corrupt
    should store the 48 and tell you about the 2 — not fail the whole batch, and
    not silently swallow the failures. So the response reports both sides rather
    than being a bare list.
    """

    uploaded: list[ImageRead]
    skipped: list[str]  # "photo.txt: not a recognised image format"

    # --- import summary (populated when a zip turned out to be a dataset) ---
    # The UI needs these to tell you what actually happened. "Uploaded 1,200
    # images" is a much worse message than "Uploaded 1,200 images, imported
    # 8,400 annotations, created 6 classes, split train/valid/test".
    annotations_imported: int = 0
    classes_created: list[str] = []
    #: split name -> image count, e.g. {"train": 700, "val": 200, "test": 100}
    splits: dict[str, int] = {}
    #: True when the archive used train/valid/test folders, so the UI knows the
    #: split was chosen by the user's own data rather than defaulted by us.
    has_split_folders: bool = False
    notes: list[str] = []
    #: Images already in this project byte-for-byte, so not added again.
    #: Re-importing a folder used to silently double the dataset.
    duplicates_skipped: int = 0
    #: Groups every image added by ONE upload, across all the requests a large
    #: folder is split into, so the whole import can be undone as a unit.
    import_id: str | None = None
    #: Boxes written as PROPOSALS because their image was already in the
    #: project. They await Accept/Reject rather than overwriting existing work.
    proposals_created: int = 0
    #: How many existing images received those proposals.
    reannotated_images: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uploaded_count(self) -> int:
        return len(self.uploaded)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_val_split(self) -> bool:
        """True when the import gave us train data but no validation set.

        The UI uses this to prompt for a percentage split instead of silently
        training with no held-out data — which produces a model that looks
        perfect and generalises like a rock.

        Deliberately NOT conditioned on `has_split_folders`. It used to be, so a
        folder with no train/val/test naming — everything landing in train,
        which is exactly the case most in need of a prompt — reported False.
        The question this answers is "is there anything to validate against?",
        and how the images got their splits doesn't change the answer.
        """
        return self.splits.get("train", 0) > 0 and self.splits.get("val", 0) == 0
