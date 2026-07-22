"""Pydantic schemas for evaluation jobs."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.timestamps import UtcDatetime


class EvaluationCreate(BaseModel):
    """Request to score a model on a dataset version's test split."""

    training_job_id: int
    dataset_version_id: int
    split: str = "test"


class PerClassAP(BaseModel):
    name: str
    ap: float | None


class EvaluationJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    training_job_id: int
    dataset_version_id: int
    split: str
    status: str
    num_images: int
    map_50_95: float | None
    map_50: float | None
    map_75: float | None
    error: str | None
    created_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None

    #: per_class_json unpacked, so the client gets a list, not a string to parse.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def per_class(self) -> list[PerClassAP]:
        raw = getattr(self, "per_class_json", None)
        if not raw:
            return []
        try:
            return [PerClassAP(**c) for c in json.loads(raw)]
        except (ValueError, TypeError):
            return []

    # Carried through from the ORM object so the computed field above can read
    # it; excluded from the response body itself.
    per_class_json: str | None = Field(default=None, exclude=True)
