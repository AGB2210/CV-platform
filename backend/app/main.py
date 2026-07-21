"""
FastAPI application entrypoint.

Run from the `backend/` directory with:
    uvicorn app.main:app --reload --port 8000

Interactive API docs are then at http://localhost:8000/docs — FastAPI generates
them from the type hints and Pydantic schemas, so they stay accurate for free.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import api_router
from app.config import REPO_ROOT, settings
from app.database import init_db

logger = logging.getLogger(__name__)


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


# --- Production: serve the built frontend from this same process -------------
#
# In development this does nothing: Vite serves the UI on :5173 and proxies
# /api here, and `frontend/dist` doesn't exist. In a release build the frontend
# is compiled to static files and there is no Node on the machine at all, so
# something has to serve them — and serving them from the SAME ORIGIN is what
# makes the app's relative `/api/...` calls work with no configuration, no
# hardcoded hostname, and no CORS involved.
#
# Mounted LAST, and only at "/", so it cannot shadow /api or /static/images:
# Starlette matches routes in registration order.

_FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


class SpaStaticFiles(StaticFiles):
    """Static files, but unknown paths fall back to index.html.

    THE ROUTER LIVES IN THE BROWSER. react-router owns /projects/3/train, and
    that URL exists only once the JS has booted. Ask the server for it directly
    — by refreshing the page, or pasting a link — and a plain static handler
    returns 404 for a route the app genuinely has.

    So a miss serves index.html and lets the client router resolve it.

    BUT NOT FOR EVERYTHING. This mount sits at "/", so it also catches any
    /api path the router didn't match — and answering a mistyped endpoint with
    "200 OK" and a page is a genuinely hostile way to fail. A client asking for
    JSON gets HTML and a success code, and the mistake surfaces somewhere far
    away as a parse error. Those paths keep their 404.
    """

    #: Prefixes that must never be answered with the SPA shell. These belong to
    #: the server, so a miss under them is a real 404, not a client-side route.
    API_PREFIXES = ("/api", "/static", "/docs", "/redoc", "/openapi.json")

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # StaticFiles RAISES for a missing file rather than returning a 404
            # response, so checking `response.status_code` never fires and the
            # fallback silently does nothing. Measured: /projects/3/train
            # returned 404 with that version.
            if exc.status_code != 404:
                raise
            request_path = scope.get("path", "")
            if request_path.startswith(self.API_PREFIXES):
                raise
            return await super().get_response("index.html", scope)


if _FRONTEND_DIST.is_dir():
    app.mount("/", SpaStaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info("Serving built frontend from %s", _FRONTEND_DIST)
else:
    # Expected in development. Saying so once at startup beats someone
    # wondering why http://localhost:8000 shows a bare JSON root.
    logger.info(
        "No built frontend at %s — API only. Run the frontend dev server, or "
        "`npm run build` to serve it from here.",
        _FRONTEND_DIST,
    )
