"""
Dataset versions — save points for the dataset, and the accident recovery they
exist to provide.

The headline test here is `test_restore_brings_back_a_deleted_image`: the whole
point of the feature is that deleting the wrong images stops being permanent. If
that ever breaks, versions are a promise the app can't keep, so it's asserted
end-to-end (row gone, file kept, restore recreates both the image and its boxes).
"""

from __future__ import annotations

from tests.conftest import make_project, upload_images


def _class_id(client, pid) -> int:
    return client.get(f"/api/projects/{pid}/classes").json()[0]["id"]


def _add_box(client, image_id: int, category_id: int) -> None:
    r = client.post(
        f"/api/images/{image_id}/annotations",
        json={"category_id": category_id, "x": 5, "y": 5, "width": 10, "height": 10},
    )
    assert r.status_code == 201, r.text


def _save(client, pid, note=None):
    r = client.post(
        f"/api/projects/{pid}/dataset/versions", json={"note": note}
    )
    assert r.status_code == 201, r.text
    return r.json()


def _versions(client, pid) -> list[dict]:
    return client.get(f"/api/projects/{pid}/dataset/versions").json()


def _image_file(pid: int, filename: str):
    from app.services import storage

    return storage.project_dir(pid) / filename


def _setup(client, name="Ver"):
    """A project with 4 images, boxes on the first two, and a val split."""
    pid = make_project(client, name, classes=("car",))
    imgs = upload_images(client, pid, [f"i{i}.png" for i in range(4)])
    car = _class_id(client, pid)
    for img in imgs[:2]:
        _add_box(client, img["id"], car)
    client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": [imgs[3]["id"]], "split": "val"},
    )
    return pid, imgs


# --- saving -----------------------------------------------------------------


def test_save_captures_counts_and_numbers_from_one(client):
    pid, _ = _setup(client)
    v1 = _save(client, pid, note="first cut")

    assert v1["version"] == 1, "versions start at 1 per project"
    assert v1["note"] == "first cut"
    assert v1["total_images"] == 4
    assert v1["train_images"] == 3 and v1["val_images"] == 1
    assert v1["total_boxes"] == 2
    assert v1["num_classes"] == 1

    v2 = _save(client, pid)
    assert v2["version"] == 2
    assert [v["version"] for v in _versions(client, pid)] == [2, 1], "newest first"


def test_save_rejects_empty_dataset(client):
    """Nothing to version — say so rather than writing an empty save point."""
    pid = make_project(client, "Empty", classes=("car",))
    r = client.post(f"/api/projects/{pid}/dataset/versions", json={})
    assert r.status_code == 400


def test_versions_are_scoped_to_their_project(client):
    a, _ = _setup(client, "A")
    b, _ = _setup(client, "B")
    _save(client, a)
    _save(client, a)
    _save(client, b)
    assert [v["version"] for v in _versions(client, a)] == [2, 1]
    assert [v["version"] for v in _versions(client, b)] == [1], "B numbers from 1"


# --- the accident this feature exists for -----------------------------------


def test_delete_keeps_the_file_once_a_version_exists(client):
    """Versions store metadata, not bytes — so the bytes must survive a delete or
    no version could ever restore the image."""
    pid, imgs = _setup(client)
    victim = imgs[0]
    _save(client, pid)

    client.delete(f"/api/images/{victim['id']}")
    assert _image_file(pid, victim["filename"]).exists(), (
        "the file must be kept so a restore can bring the image back"
    )


def test_delete_removes_the_file_when_nothing_could_restore_it(client):
    """Before the first save there's no version referencing it, so keeping the
    file would just accumulate orphans."""
    pid, imgs = _setup(client)
    victim = imgs[0]
    client.delete(f"/api/images/{victim['id']}")
    assert not _image_file(pid, victim["filename"]).exists()


def test_restore_brings_back_a_deleted_image(client):
    """THE test: delete images by accident, restore, get them (and their boxes)
    back."""
    pid, imgs = _setup(client)
    car = _class_id(client, pid)
    v1 = _save(client, pid)

    # Accident: delete two images, one of which had a box.
    client.delete(f"/api/images/{imgs[0]['id']}")
    client.delete(f"/api/images/{imgs[2]['id']}")
    after = client.get(f"/api/projects/{pid}/images").json()
    assert len(after) == 2, "precondition: they really are gone"

    r = client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["restored_version"] == 1
    assert result["missing_files"] == [], "nothing should be unrecoverable"

    restored = client.get(f"/api/projects/{pid}/images").json()
    assert len(restored) == 4, "both deleted images are back"
    by_name = {i["original_filename"]: i for i in restored}
    assert set(by_name) == {f"i{i}.png" for i in range(4)}
    # The box that lived on the deleted image came back with it.
    revived = by_name["i0.png"]
    boxes = client.get(f"/api/images/{revived['id']}/annotations").json()
    assert len(boxes) == 1 and boxes[0]["category_id"] == car


def test_restore_removes_images_added_after_the_version(client):
    """Restoring means 'make it look like then' — including undoing an upload."""
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    upload_images(client, pid, ["late1.png", "late2.png"])
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 6

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert names == {f"i{i}.png" for i in range(4)}, "the later upload is rolled back"


def test_restore_between_two_saved_versions_is_reversible(client):
    """With both states saved, a restore is undone by restoring the other one.

    Restore no longer auto-saves anything, so reversibility is a consequence of
    the user having saved — not something the app arranges behind their back.
    """
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    upload_images(client, pid, ["later.png"])
    v2 = _save(client, pid)

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert "later.png" not in names

    client.post(f"/api/projects/{pid}/dataset/versions/{v2['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert "later.png" in names, "restoring v2 undid the restore of v1"


# --- only an explicit save creates a version --------------------------------


def test_restore_never_creates_a_version(client):
    """"Save dataset" is the ONLY thing that makes a version.

    Restore used to mint an automatic backup of the pre-restore state. It made
    the list grow on its own — often with near-identical entries — and that is
    not what the user asked the app to keep.
    """
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    assert len(_versions(client, pid)) == 1

    # Genuinely unsaved divergence: the old auto-backup would have kept this.
    client.delete(f"/api/images/{imgs[0]['id']}")
    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")

    assert len(_versions(client, pid)) == 1, "restore created no version"


def test_restore_discards_unsaved_changes(client):
    """The cost of not auto-saving, asserted so it stays a deliberate choice."""
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    upload_images(client, pid, ["unsaved.png"])

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert "unsaved.png" not in names, "unsaved work is gone — the UI warns first"


def test_restore_rewinds_the_class_list(client):
    """A class added after the version is removed by restoring it.

    THE BUG THIS GUARDS: classes are dataset content — they're in the snapshot,
    they're in the content fingerprint, and they fix the model's output
    vocabulary. Restore rewound the boxes but kept a later class, so the
    fingerprint never matched again: the app reported "unsaved changes"
    immediately after a restore, no version showed as current, and an
    unqualified Train fell back to the newest version — the exact state the
    user had just rolled away from.
    """
    pid, imgs = _setup(client)
    v1 = _save(client, pid)

    client.post(f"/api/projects/{pid}/classes", json={"name": "truck"})
    assert len(client.get(f"/api/projects/{pid}/classes").json()) == 2

    r = client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore").json()
    assert r["classes_removed"] == ["truck"], "reported, not silent"

    names = {c["name"] for c in client.get(f"/api/projects/{pid}/classes").json()}
    assert names == {"car"}, "the later class is gone"

    by_version = {v["version"]: v for v in _versions(client, pid)}
    assert by_version[1]["is_current"] is True, "the restored version IS what's on screen"


def test_restore_still_brings_back_a_deleted_class(client):
    """The other direction: a class deleted after the version comes back."""
    pid, imgs = _setup(client)
    car = _class_id(client, pid)
    _add_box(client, imgs[0]["id"], car)
    v1 = _save(client, pid)

    client.delete(f"/api/classes/{car}")
    assert client.get(f"/api/projects/{pid}/classes").json() == []

    r = client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore").json()
    assert r["classes_removed"] == []
    names = {c["name"] for c in client.get(f"/api/projects/{pid}/classes").json()}
    assert names == {"car"}, "the deleted class is restored"


def test_current_marks_the_version_on_screen_not_the_newest(client):
    """After restoring an older version, THAT one is current — the newest save
    point still exists but is not what you're looking at."""
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    client.delete(f"/api/images/{imgs[0]['id']}")
    v2 = _save(client, pid)  # newest, 3 images

    assert [v["is_current"] for v in _versions(client, pid)] == [True, False], "v2 is live"

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    by_version = {v["version"]: v for v in _versions(client, pid)}
    assert by_version[1]["is_current"] is True, "the restored version is what's on screen"
    assert by_version[2]["is_current"] is False, "newest is no longer current"
    assert v2["version"] == 2


def test_training_defaults_to_the_current_version_not_the_newest(client):
    """The bug this guards: after restoring v1, 'just train it' must train v1 —
    not the newer save point holding data the user rolled away from."""
    pid, imgs = _setup(client)
    car = _class_id(client, pid)
    for img in imgs[1:]:
        _add_box(client, img["id"], car)
    v1 = _save(client, pid)
    client.delete(f"/api/images/{imgs[0]['id']}")
    _save(client, pid)  # v2, newest

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["current_version"] == 1
    assert p["latest_version"] == 2, "the newer save point still exists"
    assert p["has_unsaved_changes"] is False


def test_unsaved_changes_are_reported(client):
    pid, imgs = _setup(client)
    _save(client, pid)
    client.delete(f"/api/images/{imgs[0]['id']}")
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["current_version"] is None
    assert p["has_unsaved_changes"] is True
    assert any("changed since it was last saved" in w for w in p["warnings"])


# --- renaming ---------------------------------------------------------------


def _rename(client, pid, version_id, name):
    return client.patch(
        f"/api/projects/{pid}/dataset/versions/{version_id}", json={"name": name}
    )


def test_rename_and_clear(client):
    pid, _ = _setup(client)
    v1 = _save(client, pid)
    assert v1["name"] is None, "unnamed versions display as v{n}"

    r = _rename(client, pid, v1["id"], "  baseline  ")
    assert r.status_code == 200
    assert r.json()["name"] == "baseline", "whitespace trimmed"

    # Blank clears it, reverting to the numeric label — how a rename is undone.
    assert _rename(client, pid, v1["id"], "   ").json()["name"] is None


def test_rename_rejects_duplicate_name(client):
    pid, _ = _setup(client)
    v1 = _save(client, pid)
    v2 = _save(client, pid)
    _rename(client, pid, v1["id"], "baseline")

    r = _rename(client, pid, v2["id"], "baseline")
    assert r.status_code == 409
    assert "already used" in r.json()["detail"]

    # Case-insensitive: "BASELINE" would read as the same row in a list.
    assert _rename(client, pid, v2["id"], "BASELINE").status_code == 409
    # Renaming a version to its OWN name is fine (no-op, not a clash).
    assert _rename(client, pid, v1["id"], "baseline").status_code == 200


def test_rename_rejects_clash_with_a_numeric_label(client):
    """An unnamed version still occupies a label. Naming another one "v1" would
    put two rows reading "v1" in the same list."""
    pid, _ = _setup(client)
    _save(client, pid)  # v1, unnamed -> displays as "v1"
    v2 = _save(client, pid)
    r = _rename(client, pid, v2["id"], "v1")
    assert r.status_code == 409


# --- deleting ---------------------------------------------------------------


def test_delete_version_removes_its_snapshot(client):
    from pathlib import Path

    pid, _ = _setup(client)
    v1 = _save(client, pid)
    v2 = _save(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import DatasetVersion

        snapshot = Path(db.get(DatasetVersion, v1["id"]).snapshot_path)
    finally:
        db.close()
    assert snapshot.exists()

    assert client.delete(f"/api/projects/{pid}/dataset/versions/{v1['id']}").status_code == 204
    assert not snapshot.exists(), "the snapshot file goes with the row"
    assert [v["version"] for v in _versions(client, pid)] == [v2["version"]]


def test_delete_does_not_touch_image_files(client):
    """Version snapshots are metadata. Deleting one must not remove pictures the
    live dataset (and other versions) still use."""
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    client.delete(f"/api/projects/{pid}/dataset/versions/{v1['id']}")
    assert _image_file(pid, imgs[0]["filename"]).exists()
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 4


def test_bulk_delete_including_all(client):
    pid, _ = _setup(client)
    ids = [_save(client, pid)["id"] for _ in range(3)]

    r = client.post(
        f"/api/projects/{pid}/dataset/versions/bulk-delete",
        json={"version_ids": ids[:2]},
    )
    assert r.status_code == 200 and r.json()["deleted"] == 2
    assert len(_versions(client, pid)) == 1

    # "Delete all" is the same path with every id selected.
    r = client.post(
        f"/api/projects/{pid}/dataset/versions/bulk-delete",
        json={"version_ids": [ids[2]]},
    )
    assert r.json()["deleted"] == 1
    assert _versions(client, pid) == []


def test_bulk_delete_reports_unknown_ids(client):
    pid, _ = _setup(client)
    v1 = _save(client, pid)
    r = client.post(
        f"/api/projects/{pid}/dataset/versions/bulk-delete",
        json={"version_ids": [v1["id"], 9999]},
    ).json()
    assert r["deleted"] == 1 and r["not_found"] == [9999]


def test_versions_of_another_project_are_untouchable(client):
    a, _ = _setup(client, "A")
    b, _ = _setup(client, "B")
    v = _save(client, b)
    assert client.delete(f"/api/projects/{a}/dataset/versions/{v['id']}").status_code == 404
    assert _rename(client, a, v["id"], "x").status_code == 404


def test_restore_reinstates_split_and_boxes(client):
    """Splits and box edits made after the version are rolled back too."""
    pid, imgs = _setup(client)
    car = _class_id(client, pid)
    v1 = _save(client, pid)

    # Churn: move an image to test, and add a box to a previously-empty image.
    client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": [imgs[0]["id"]], "split": "test"},
    )
    _add_box(client, imgs[2]["id"], car)

    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")

    images = {i["original_filename"]: i for i in client.get(f"/api/projects/{pid}/images").json()}
    assert images["i0.png"]["split"] == "train", "the split change was rolled back"
    assert images["i3.png"]["split"] == "val", "the original val split is intact"
    assert images["i2.png"]["annotation_count"] == 0, "the later box is gone"
    assert images["i0.png"]["annotation_count"] == 1, "the original box is back"
