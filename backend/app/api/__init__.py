"""
API layer: the aggregate router that main.py mounts.

Collecting sub-routers here means `main.py` never grows a line per feature — it
includes one router, and new feature areas are registered in this file.
"""

from fastapi import APIRouter

from app.api.routes import health

api_router = APIRouter()
api_router.include_router(health.router)

# Phase 1 will add, e.g.:
#   from app.api.routes import projects
#   api_router.include_router(projects.router, prefix="/projects")
