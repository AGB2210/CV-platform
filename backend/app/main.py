"""
FastAPI application entrypoint.

Run from the `backend/` directory with:
    uvicorn app.main:app --reload --port 8000

Interactive API docs are then at http://localhost:8000/docs — FastAPI generates
them from the type hints and Pydantic schemas, so they stay accurate for free.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import api_router
from app.config import settings
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks.

    The `lifespan` context manager is the modern replacement for the deprecated
    `@app.on_event("startup")`. Code before `yield` runs once at startup; code
    after runs at shutdown. Later phases will use the shutdown half to release
    GPU memory held by loaded models.
    """
    # --- startup ---
    settings.ensure_dirs()  # storage/ dirs are gitignored, so create on demand
    init_db()  # create tables that don't exist yet
    yield
    # --- shutdown ---
    # (nothing to tear down yet)


app = FastAPI(
    title=settings.APP_NAME,
    description="Local, self-hosted computer vision platform. MVP scope: object detection.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS must be added before the app starts serving. The browser sends a
# preflight OPTIONS request for anything beyond a simple GET; without this
# middleware the frontend's fetch() calls fail with an opaque CORS error
# rather than a useful HTTP status.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All API endpoints live under /api. Namespacing from day one means we can later
# serve the built frontend from "/" on the same origin without route collisions.
app.include_router(api_router, prefix="/api")

# Serve uploaded images straight from disk as static files. This is why images
# are stored on the filesystem: the browser can fetch them with a plain <img
# src="...">, and Starlette handles range requests and caching headers for us.
settings.ensure_dirs()
app.mount("/static/images", StaticFiles(directory=settings.images_dir), name="images")
