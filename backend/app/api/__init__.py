"""
API layer: the aggregate router that main.py mounts.

Collecting sub-routers here means `main.py` never grows a line per feature — it
includes one router, and new feature areas are registered in this file.
"""

from fastapi import APIRouter

from app.api.routes import categories, health, images, projects

api_router = APIRouter()

api_router.include_router(health.router)

# Projects own a /projects prefix. Images and classes define their own full
# paths, because they're addressed two ways: nested under a project for
# collection operations (POST /projects/1/images) and flat by their own id for
# item operations (DELETE /images/5). Flat item paths avoid a pointless
# /projects/1/images/5 where the 1 is redundant — the image id already
# determines the project.
api_router.include_router(projects.router, prefix="/projects")
api_router.include_router(images.router)
api_router.include_router(categories.router)
