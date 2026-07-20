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


def test_restore_is_itself_undoable(client):
    """A mistaken restore is one more restore away from being fixed: the current
    state is saved first."""
    pid, imgs = _setup(client)
    v1 = _save(client, pid)
    upload_images(client, pid, ["later.png"])

    r = client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore").json()
    backup = r["backup_version"]
    assert backup == 2, "the pre-restore state was saved as the next version"

    # Undo the restore by restoring the backup — the later upload returns.
    versions = _versions(client, pid)
    backup_row = next(v for v in versions if v["version"] == backup)
    client.post(f"/api/projects/{pid}/dataset/versions/{backup_row['id']}/restore")
    names = {i["original_filename"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert "later.png" in names, "restoring the backup undid the restore"


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
