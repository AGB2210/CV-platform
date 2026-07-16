"""Pydantic schemas for annotations, auto-annotation jobs, and models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field


class AnnotatorInfo(BaseModel):
    """One entry in the UI's model dropdown."""

    key: str
    display_name: str
    description: str
    approx_vram_gb: float


class DeviceInfo(BaseModel):
    """Hardware summary, so the UI can warn before a 20-minute CPU run."""

    available: bool
    device: str
    name: str
    total_vram_gb: float | None = None
    compute_capability: str | None = None
    note: str | None = None


class AnnotationRead(BaseModel):
    """One bounding box, as the canvas consumes it."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    image_id: int
    category_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float | None
    source: str
    reviewed: bool


class AnnotationCreate(BaseModel):
    """A box drawn or edited by a human (Phase 3)."""

    category_id: int
    # ge=0 rejects negative coordinates; the canvas can produce them if a drag
    # goes past the top-left corner and the frontend forgets to clamp.
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    # gt=0 rejects zero-area boxes — a click without a drag would otherwise
    # store an invisible, unexportable, untrainable box.
    width: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


class AnnotationJobCreate(BaseModel):
    """Request body to launch an auto-annotation run."""

    model_key: str
    # Thresholds bounded 0..1 — they're probabilities. A typo'd 30 instead of
    # 0.30 would otherwise silently produce zero detections and look like a
    # broken model rather than a bad input.
    box_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Optional {class_name: prompt} overrides, e.g. {"car": "a parked car"}.
    #: Grounding DINO is genuinely sensitive to phrasing.
    prompts: dict[str, str] = Field(default_factory=dict)


class AnnotationJobRead(BaseModel):
    """Job state — polled by the frontend while a run is in flight."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    model_key: str
    status: str
    total_images: int
    processed_images: int
    boxes_created: int
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def progress_pct(self) -> float:
        """Derived, not stored — a stored percentage is just a chance to drift
        out of sync with the counters it's computed from."""
        if not self.total_images:
            return 0.0
        return round(100.0 * self.processed_images / self.total_images, 1)


class ExportFormatInfo(BaseModel):
    key: str
    display_name: str
    description: str
