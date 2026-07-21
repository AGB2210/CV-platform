"""Pydantic schemas for projects."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.timestamps import UtcDatetime

from app.enums import TaskType


class ProjectCreate(BaseModel):
    """Request body for POST /api/projects."""

    # Field(...) means required. The length bounds are enforced before any of
    # our code runs — FastAPI returns 422 with a precise message and the
    # endpoint body is never entered.
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    # THIS is where task types are policed. The DB column accepts any string
    # (see app/enums.py for why); Pydantic is the gate that rejects anything not
    # in the enum. Strict API, permissive schema — which is what makes adding
    # "segmentation" a one-line change instead of a migration.
    task_type: TaskType = TaskType.OBJECT_DETECTION

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        """Reject whitespace-only names, and normalise surrounding spaces.

        min_length=1 alone would happily accept "   ": it counts characters, not
        meaning. Stripping here means the stored value is what gets validated,
        not a variant of it.
        """
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("name cannot be blank")
        return cleaned


class ProjectUpdate(BaseModel):
    """Request body for PATCH /api/projects/{id}.

    Every field optional, because PATCH is a partial update. `None` and "absent"
    mean different things here — see the route, which uses
    `exclude_unset=True` to tell them apart.
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str | None) -> str | None:
        if v is None:
            return v
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("name cannot be blank")
        return cleaned


class ProjectRead(BaseModel):
    """Response body for a single project.

    from_attributes=True lets FastAPI build this straight from a SQLAlchemy
    object by reading attributes, instead of us hand-writing a dict per
    endpoint.

    Note this schema does NOT simply mirror the table. It adds image_count and
    class_count, which live in other tables — a concrete example of why schemas
    and models are separate: the API's shape is driven by what the UI needs to
    render, not by how rows happen to be stored.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    task_type: str
    created_at: UtcDatetime
    updated_at: UtcDatetime

    # Computed per-request by the route with a COUNT query, not stored on the
    # projects table. Keeping a counter column would mean keeping it correct on
    # every insert/delete — a classic source of drift for a number that a
    # database can compute exactly.
    image_count: int = 0
    class_count: int = 0

    #: When anything in this project last changed — an upload, a saved dataset
    #: version, a training run, or an edit to the project itself.
    #:
    #: NOT the same as `updated_at`, which SQLAlchemy only touches when the
    #: projects ROW is updated. Uploading 500 images or training a model leaves
    #: `updated_at` untouched, so sorting by it would order projects by when
    #: somebody last renamed one — almost never what "last modified" means to
    #: the person reading the list.
    last_activity_at: UtcDatetime | None = None
