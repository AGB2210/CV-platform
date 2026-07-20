"""
Pytest fixtures: an isolated app on a throwaway database and storage dir.

WHY IT'S BUILT THIS WAY
-----------------------
Every test gets a fresh temp SQLite file and a fresh temp storage dir, so tests
can't see each other's data and none of them touch the real cvplatform.db or
storage/. The app's own endpoints run against it through FastAPI's dependency
override — the same code path a real request takes, not a mock.

These tests deliberately do NOT run the ML model. Auto-annotation needs a GPU
and ~700 MB of weights, and none of the logic worth testing here (proposal
accept/reject, splits, import, provenance) depends on what the model actually
predicted — only on the annotation ROWS. So the `proposals` fixture inserts
`proposed=True` rows directly, exactly as a finished job would have, and the
tests exercise the real endpoints against them. Fast, deterministic, no GPU.
"""

from __future__ import annotations

import io
import itertools

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient wired to a throwaway DB and storage dir."""
    # Point storage at a temp dir BEFORE anything imports/uses it, so
    # save_image() writes there instead of the real storage/. images_dir is a
    # computed property, so setting the root is enough.
    from app.config import settings

    monkeypatch.setattr(settings, "STORAGE_DIR", tmp_path / "storage")
    settings.ensure_dirs()

    # A fresh SQLite file per test. check_same_thread=False mirrors the app's
    # engine; the FK-enforcement PRAGMA is attached to the generic Engine class
    # in app.database, so it fires for this engine too and cascades work.
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )

    from app.database import Base, get_db
    import app.models  # noqa: F401  — registers every model on Base.metadata

    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.main import app

    app.dependency_overrides[get_db] = override_get_db

    # Stash the session factory on the client so tests that need direct DB
    # access (inserting proposal rows) use the SAME database the endpoints see.
    test_client = TestClient(app)
    test_client.SessionLocal = TestSession  # type: ignore[attr-defined]

    yield test_client

    app.dependency_overrides.clear()


#: Makes each png_bytes() call produce different bytes. See below.
_png_counter = itertools.count()


def png_bytes(w: int = 64, h: int = 48, colour=None) -> bytes:
    """A tiny valid PNG for upload tests. DISTINCT bytes on every call.

    Upload deduplicates by content hash, so a helper that returned the same
    image every time would have every file after the first recognised as a
    re-upload and skipped — and dozens of tests would fail for a reason with
    nothing to do with what they were testing. Real datasets contain distinct
    pictures, so distinct-by-default is also the realistic fixture.

    Pass an explicit `colour` to get deterministic, repeatable bytes — that's
    how the duplicate-detection tests ask for the *same* image twice.
    """
    if colour is None:
        n = next(_png_counter)
        # Walk the colour cube so consecutive calls differ visibly as well as
        # byte-wise, which helps when a failure dumps an image.
        colour = (30 + (n * 37) % 220, 60 + (n * 53) % 190, 90 + (n * 71) % 160)
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), colour).save(buf, "PNG")
    return buf.getvalue()


def make_project(client, name="Test", classes=("car", "person")) -> int:
    """Create a project with classes; return its id."""
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    for c in classes:
        client.post(f"/api/projects/{pid}/classes", json={"name": c})
    return pid


def upload_images(client, pid: int, names: list[str]) -> list[dict]:
    """Upload N DISTINCT images, return the created image rows.

    Each gets its own colour, so each has its own bytes. That matters since
    upload deduplicates by content hash: a helper handing the same PNG to every
    name would have the second one onwards recognised as re-uploads and skipped,
    and every test built on it would fail for a reason that has nothing to do
    with what it was testing.

    Real datasets have distinct images, so this is the realistic fixture — the
    identical-bytes case belongs in the tests that are actually about duplicates.
    """
    files = [
        ("files", (n, png_bytes(), "image/png")) for n in names
    ]
    r = client.post(f"/api/projects/{pid}/images", files=files)
    assert r.status_code == 201, r.text
    return client.get(f"/api/projects/{pid}/images?limit=500").json()


def add_proposals(client, image_id: int, category_id: int, n: int = 1) -> list[int]:
    """Insert n proposal rows on an image, as a finished job would have.

    Goes through the test's own session so the rows land in the same DB the
    endpoints read. Returns the new annotation ids.
    """
    from app.models import Annotation

    db = client.SessionLocal()  # type: ignore[attr-defined]
    ids = []
    try:
        for i in range(n):
            a = Annotation(
                image_id=image_id,
                category_id=category_id,
                x=10 + i * 5,
                y=10,
                width=30,
                height=30,
                confidence=0.5,
                source="auto",
                reviewed=False,
                proposed=True,
            )
            db.add(a)
            db.flush()
            ids.append(a.id)
        db.commit()
    finally:
        db.close()
    return ids
