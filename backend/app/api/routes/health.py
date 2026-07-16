"""Health-check endpoint.

Serves two purposes: it proves the API process is alive, and — because it opens
a real database session — it proves the DB file is reachable and writable. A
health check that only returns `{"status": "ok"}` without touching dependencies
will happily report green while the database is gone.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db

# Routers let each feature area own its endpoints in its own file; main.py just
# mounts them. Phase 1 adds `projects.py` and `images.py` alongside this.
router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)) -> dict:
    """Report API liveness plus database connectivity."""
    try:
        # Cheapest possible round-trip that still proves the connection works.
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as exc:  # noqa: BLE001 - health checks report, never raise
        db_status = f"error: {exc}"

    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "database": db_status,
        "storage_dir": str(settings.STORAGE_DIR),
    }
