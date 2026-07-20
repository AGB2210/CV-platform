"""
Disk housekeeping: what can be freed, and what only LOOKS like it can.

The dangerous mistake this suite exists to prevent is treating a file with no
live image row as garbage. Once a project has versions, deleting an image keeps
its bytes precisely so a restore can bring it back — so the obvious
implementation (diff the directory against the images table) would delete
exactly the data versions exist to protect.
"""

from __future__ import annotations

from app.config import from_storage_path

from tests.conftest import make_project, png_bytes, upload_images


def _report(client, pid) -> dict:
    return client.get(f"/api/projects/{pid}/storage").json()


def _save(client, pid):
    r = client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})
    assert r.status_code == 201, r.text
    return r.json()


def _files_on_disk(pid) -> set[str]:
    from app.services import storage

    return {p.name for p in storage.project_dir(pid).iterdir() if p.is_file()}


# --- unsaved images ---------------------------------------------------------


def test_uploaded_images_are_reported_as_unsaved_until_a_version_exists(client):
    pid = make_project(client, "Unsaved", classes=("car",))
    upload_images(client, pid, ["a.png", "b.png", "c.png"])

    assert _report(client, pid)["unsaved_images"] == 3

    _save(client, pid)
    assert _report(client, pid)["unsaved_images"] == 0, "the save point covers them"

    upload_images(client, pid, ["d.png"])
    assert _report(client, pid)["unsaved_images"] == 1, "only the new one"


def test_discarding_unsaved_images_removes_rows_and_files(client):
    """The user's explicit cleanup for an upload they didn't want."""
    pid = make_project(client, "Discard", classes=("car",))
    saved_imgs = upload_images(client, pid, ["keep1.png", "keep2.png"])
    _save(client, pid)
    upload_images(client, pid, ["junk1.png", "junk2.png", "junk3.png"])

    r = client.post(f"/api/projects/{pid}/storage/discard-unsaved").json()
    assert r["deleted"] == 3
    assert r["bytes_freed"] > 0

    remaining = client.get(f"/api/projects/{pid}/images").json()
    assert {i["original_filename"] for i in remaining} == {"keep1.png", "keep2.png"}
    # And their bytes are gone — nothing could have restored them anyway.
    assert _files_on_disk(pid) == {i["filename"] for i in saved_imgs}


def test_discard_is_a_no_op_when_everything_is_saved(client):
    pid = make_project(client, "AllSaved", classes=("car",))
    upload_images(client, pid, ["a.png"])
    _save(client, pid)
    assert client.post(f"/api/projects/{pid}/storage/discard-unsaved").json()["deleted"] == 0


# --- orphans vs retained ----------------------------------------------------


def test_a_file_kept_for_a_version_is_retained_not_orphaned(client):
    """THE distinction that matters. No live row, but a version needs it."""
    pid = make_project(client, "Retained", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    _save(client, pid)

    client.delete(f"/api/images/{imgs[0]['id']}")  # row goes, bytes stay

    report = _report(client, pid)
    assert report["retained_files"] == 1, "a version depends on it"
    assert report["orphan_files"] == 0, "so it is NOT waste"

    # ...and reclaim must not touch it.
    client.post(f"/api/projects/{pid}/storage/reclaim")
    assert imgs[0]["filename"] in _files_on_disk(pid)

    # The proof that it was worth keeping: restore brings the image back.
    version = client.get(f"/api/projects/{pid}/dataset/versions").json()[0]
    client.post(f"/api/projects/{pid}/dataset/versions/{version['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert "a.png" in names


def test_a_file_becomes_orphaned_once_every_version_holding_it_is_gone(client):
    """The genuine leak: deleted image, and then the version that referenced it
    is deleted too. Nothing can reach the bytes any more."""
    pid = make_project(client, "Orphan", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    version = _save(client, pid)

    client.delete(f"/api/images/{imgs[0]['id']}")
    assert _report(client, pid)["retained_files"] == 1

    client.delete(f"/api/projects/{pid}/dataset/versions/{version['id']}")

    report = _report(client, pid)
    assert report["orphan_files"] == 1, "now genuinely unreachable"
    assert report["retained_files"] == 0

    r = client.post(f"/api/projects/{pid}/storage/reclaim").json()
    assert r["files_removed"] == 1
    assert r["bytes_freed"] > 0
    assert imgs[0]["filename"] not in _files_on_disk(pid)
    assert imgs[1]["filename"] in _files_on_disk(pid), "the live image is untouched"


def test_reclaim_refuses_when_a_snapshot_cannot_be_read(client):
    """Without knowing what a version depends on, "unreferenced" is a guess —
    and guessing here deletes irreplaceable data."""
    from pathlib import Path

    pid = make_project(client, "Broken", classes=("car",))
    upload_images(client, pid, ["a.png"])
    version = _save(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    from app.models import DatasetVersion

    row = db.get(DatasetVersion, version["id"])
    from_storage_path(row.snapshot_path).write_text("{ this is not valid json")
    db.close()

    report = _report(client, pid)
    assert report["unreadable_versions"] == ["v1"]

    r = client.post(f"/api/projects/{pid}/storage/reclaim")
    assert r.status_code == 409
    assert "could not be read" in r.json()["detail"]


# --- undoing an import ------------------------------------------------------


def test_undo_removes_every_image_from_one_import(client):
    """A large folder arrives as many requests sharing one import id, so a
    partly-failed upload comes out as a unit."""
    pid = make_project(client, "Undo", classes=("car",))

    # Two requests, one import — exactly what a batched folder upload does.
    for names in (["a.png", "b.png"], ["c.png"]):
        files = [("files", (n, png_bytes(), "image/png")) for n in names]
        files.append(("import_id", (None, "imp123")))
        assert client.post(f"/api/projects/{pid}/images", files=files).status_code == 201

    # ...plus an unrelated upload that must survive.
    upload_images(client, pid, ["unrelated.png"])
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 4

    r = client.post(f"/api/projects/{pid}/imports/imp123/undo").json()
    assert r["deleted"] == 3, "both batches, one action"
    assert r["kept_in_versions"] == 0

    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert names == {"unrelated.png"}


def test_undo_keeps_images_a_version_has_since_captured(client):
    """Once a save point depends on an image, removing it would break that
    version — and by saving, the user said they want it."""
    pid = make_project(client, "UndoSaved", classes=("car",))
    files = [("files", (n, png_bytes(), "image/png")) for n in ["a.png", "b.png"]]
    files.append(("import_id", (None, "imp999")))
    client.post(f"/api/projects/{pid}/images", files=files)

    _save(client, pid)

    r = client.post(f"/api/projects/{pid}/imports/imp999/undo").json()
    assert r["deleted"] == 0
    assert r["kept_in_versions"] == 2
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 2


def test_undo_of_an_unknown_import_is_404(client):
    pid = make_project(client, "NoImport", classes=("car",))
    assert client.post(f"/api/projects/{pid}/imports/nope/undo").status_code == 404
