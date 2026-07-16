"""Pydantic schemas for categories (called "classes" in the UI)."""

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


class CategoryCreate(BaseModel):
    """Request body for POST /api/projects/{id}/classes."""

    name: str = Field(..., min_length=1, max_length=80)

    # Optional: the route assigns the next colour from CLASS_COLORS when the
    # client doesn't care, which is the common path. The UI shouldn't have to
    # know about the palette just to add a class.
    color: str | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("class name cannot be blank")
        return cleaned

    @field_validator("color")
    @classmethod
    def valid_hex(cls, v: str | None) -> str | None:
        """Validate the colour is a real hex code.

        This value gets interpolated into SVG/canvas fill attributes on the
        frontend. Validating the shape here keeps arbitrary strings from
        reaching the DOM, and catches typos at the API boundary rather than as a
        silently invisible box in the annotation canvas three phases from now.
        """
        if v is None:
            return v
        if not HEX_COLOR.match(v):
            raise ValueError("color must be a hex code like #2563eb")
        return v.lower()


class CategoryUpdate(BaseModel):
    """Request body for PATCH /api/classes/{id}. Partial update."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    color: str | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, v: str | None) -> str | None:
        if v is None:
            return v
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("class name cannot be blank")
        return cleaned

    @field_validator("color")
    @classmethod
    def valid_hex(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not HEX_COLOR.match(v):
            raise ValueError("color must be a hex code like #2563eb")
        return v.lower()


class CategoryRead(BaseModel):
    """Response body for a class."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    name: str
    color: str
    created_at: datetime
