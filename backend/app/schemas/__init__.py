"""
Pydantic schemas package — the API's request/response contracts.

Kept deliberately separate from `app.models` (SQLAlchemy). They describe
different things and change for different reasons:

  - models  = how data is stored (columns, indexes, foreign keys)
  - schemas = what the API accepts and returns (validation, field exposure)

ProjectRead is the concrete argument for the split: it carries `image_count`,
which is not a column on `projects` at all. The API's shape follows what the UI
needs to render; the table's shape follows what's correct to store.
"""

from app.schemas.category import CategoryCreate, CategoryRead, CategoryUpdate
from app.schemas.image import ImageRead, UploadResult
from app.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate

__all__ = [
    "CategoryCreate",
    "CategoryRead",
    "CategoryUpdate",
    "ImageRead",
    "UploadResult",
    "ProjectCreate",
    "ProjectRead",
    "ProjectUpdate",
]
