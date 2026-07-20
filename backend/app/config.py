"""
Application configuration.

Everything that might differ between machines (paths, ports, DB location) lives
here rather than being scattered as literals through the codebase. Values can be
overridden with environment variables or a `.env` file, which is what lets the
same code run on your laptop and (later) on a GPU box without edits.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to the repo root rather than the current working
# directory. Without this, running `uvicorn` from a different folder would
# silently create the database and storage dirs in the wrong place.
#   config.py -> app/ -> backend/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Typed application settings, loaded from env vars / .env with defaults."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    APP_NAME: str = "Local CV Platform"
    DEBUG: bool = True

    # --- Storage ---
    # Large binary artifacts (images, model weights, training runs) live on the
    # filesystem, NOT in the database. SQLite would technically accept image
    # blobs, but that makes the DB huge, slow to back up, and awkward to serve
    # from. The standard pattern is: bytes on disk, metadata + path in the DB.
    STORAGE_DIR: Path = REPO_ROOT / "storage"

    # --- Database ---
    # SQLite is a single file. Perfect for a local single-user tool: no server
    # to run, trivial to back up or delete and start over. If this ever became
    # multi-user with concurrent writers, this is the line that would change to
    # a Postgres URL — SQLAlchemy hides the rest of the difference.
    DATABASE_URL: str = f"sqlite:///{(REPO_ROOT / 'backend' / 'cvplatform.db').as_posix()}"

    # --- CORS ---
    # The Vite dev server runs on a different origin (port 5173) from the API
    # (port 8000). Browsers block cross-origin requests unless the server opts
    # in, so we explicitly allow the dev frontend.
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @property
    def images_dir(self) -> Path:
        return self.STORAGE_DIR / "images"

    @property
    def weights_dir(self) -> Path:
        return self.STORAGE_DIR / "weights"

    @property
    def runs_dir(self) -> Path:
        return self.STORAGE_DIR / "runs"

    @property
    def versions_dir(self) -> Path:
        """Saved dataset-version snapshots (JSON, one per version)."""
        return self.STORAGE_DIR / "versions"

    def ensure_dirs(self) -> None:
        """Create storage directories if missing. Called once on startup."""
        for path in (
            self.STORAGE_DIR,
            self.images_dir,
            self.weights_dir,
            self.runs_dir,
            self.versions_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# A single shared instance, imported everywhere else as `from app.config import settings`.
settings = Settings()
