"""Pydantic schemas for images."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


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
    created_at: datetime

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


class UploadResult(BaseModel):
    """Response body for an upload.

    Uploads are partial-success by nature: dropping 50 files where 2 are corrupt
    should store the 48 and tell you about the 2 — not fail the whole batch, and
    not silently swallow the failures. So the response reports both sides rather
    than being a bare list.
    """

    uploaded: list[ImageRead]
    skipped: list[str]  # "photo.txt: not a recognised image format"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def uploaded_count(self) -> int:
        return len(self.uploaded)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def skipped_count(self) -> int:
        return len(self.skipped)
