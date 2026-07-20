"""
Things that break for real users but not in a happy-path demo.

Each of these was found by auditing rather than by a bug report, and each one
fails silently or leaves a state nothing can recover from.
"""

from __future__ import annotations

import io

from PIL import Image as PILImage
from PIL import ImageOps

from tests.conftest import make_project, upload_images


# --- EXIF orientation -------------------------------------------------------


def _rotated_jpeg(w: int, h: int, orientation: int = 6) -> bytes:
    """A landscape JPEG tagged "rotate 90" — what every phone produces when you
    hold it upright. The pixels are NOT rotated; the tag says how to show them."""
    img = PILImage.new("RGB", (w, h), (200, 50, 50))
    buf = io.BytesIO()
    exif = img.getexif()
    exif[0x0112] = orientation
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_rotated_photos_are_stored_upright(client):
    """THE bug this guards: we recorded the RAW dimensions while the browser
    honoured the EXIF tag and rendered them swapped.

    The annotation canvas uses the stored dimensions as its SVG viewBox, so the
    coordinate space was transposed relative to the picture being drawn on and
    every box on a rotated photo landed somewhere else. Nothing errors, the
    boxes look right while you draw them, and the dataset is wrong.
    """
    pid = make_project(client, "Rotated", classes=("car",))
    data = _rotated_jpeg(200, 100)

    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("phone.jpg", data, "image/jpeg"))],
    )
    assert r.status_code == 201, r.text

    stored = client.get(f"/api/projects/{pid}/images").json()[0]
    # What a browser would display, which is what the user draws on.
    displayed = ImageOps.exif_transpose(PILImage.open(io.BytesIO(data)))
    assert (stored["width"], stored["height"]) == (displayed.width, displayed.height)
    assert (stored["width"], stored["height"]) == (100, 200), "portrait, as shown"


def test_the_stored_file_carries_no_orientation_tag(client):
    """The rotation is baked into the pixels, so nothing downstream can apply
    it a second time."""
    from app.services import storage

    pid = make_project(client, "NoTag", classes=("car",))
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("phone.jpg", _rotated_jpeg(200, 100), "image/jpeg"))],
    )
    row = client.get(f"/api/projects/{pid}/images").json()[0]
    with PILImage.open(storage.project_dir(pid) / row["filename"]) as img:
        assert (img.getexif() or {}).get(0x0112) in (None, 1)


def test_untagged_images_are_stored_byte_for_byte(client):
    """The large majority of uploads. They must not be re-encoded — that would
    lose quality for no reason at all."""
    from tests.conftest import png_bytes
    from app.services import storage

    pid = make_project(client, "Untouched", classes=("car",))
    original = png_bytes(64, 48, colour=(7, 8, 9))
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("plain.png", original, "image/png"))],
    )
    row = client.get(f"/api/projects/{pid}/images").json()[0]
    assert (storage.project_dir(pid) / row["filename"]).read_bytes() == original


# --- interrupted jobs -------------------------------------------------------


def test_jobs_left_running_by_a_kill_are_failed_at_startup(client):
    """A job runs in a background thread of THIS process. Kill it — Ctrl+C, a
    crash, the launcher's reaper — and the row says "running" forever, because
    the code that would have set a terminal status died with the thread.

    The UI then polls something that will never move, and the project refuses
    new training because the one-GPU-job guard reads exactly that status.
    """
    from app.database import _fail_interrupted_jobs
    from app.models import JobStatus, TrainingJob

    pid = make_project(client, "Interrupted", classes=("car",))
    db = client.SessionLocal()  # type: ignore[attr-defined]
    db.add(
        TrainingJob(
            project_id=pid, trainer_key="yolo", version=1, status=JobStatus.RUNNING
        )
    )
    db.commit()
    db.close()

    assert client.get(f"/api/projects/{pid}/training-jobs").json()[0]["status"] == "running"

    # Startup is the one moment nothing can legitimately be running.
    import app.database as database

    original = database.SessionLocal
    database.SessionLocal = client.SessionLocal  # type: ignore[attr-defined]
    try:
        _fail_interrupted_jobs()
    finally:
        database.SessionLocal = original

    job = client.get(f"/api/projects/{pid}/training-jobs").json()[0]
    assert job["status"] == "failed"
    assert "interrupted" in (job["error"] or "").lower()


# --- class deletion ---------------------------------------------------------


def test_class_list_reports_how_many_boxes_use_each_class(client):
    """So the confirm dialog can say what deleting destroys. Deleting a class
    cascades to every box using it, and that was a one-click, unconfirmed,
    unrecoverable action."""
    pid = make_project(client, "Counted", classes=("car", "person"))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    car, person = [c["id"] for c in client.get(f"/api/projects/{pid}/classes").json()]

    for _ in range(3):
        client.post(
            f"/api/images/{imgs[0]['id']}/annotations",
            json={"category_id": car, "x": 1, "y": 1, "width": 10, "height": 10},
        )

    counts = {c["name"]: c["annotation_count"] for c in client.get(f"/api/projects/{pid}/classes").json()}
    assert counts == {"car": 3, "person": 0}
    assert person  # referenced, keeps the linter quiet


# --- portable paths ---------------------------------------------------------


def test_stored_paths_are_relative_to_storage(client):
    """Absolute paths bake one machine's layout into the data: rename the
    folder or clone the repo and every version and checkpoint breaks, while the
    rows still look healthy."""
    from pathlib import Path

    from app.models import DatasetVersion

    pid = make_project(client, "Portable", classes=("car",))
    upload_images(client, pid, ["a.png"])
    client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        stored = db.scalars(
            __import__("sqlalchemy").select(DatasetVersion)
        ).all()[0].snapshot_path
    finally:
        db.close()

    assert not Path(stored).is_absolute(), f"still absolute: {stored}"
    assert "versions" in stored
