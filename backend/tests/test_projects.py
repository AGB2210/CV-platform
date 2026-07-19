"""Project CRUD and bulk delete."""

from tests.conftest import make_project, upload_images


def test_create_list_delete(client):
    pid = make_project(client, "P1")
    assert any(p["id"] == pid for p in client.get("/api/projects").json())

    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_project_counts(client):
    pid = make_project(client, "Counts", classes=("car", "person", "bike"))
    upload_images(client, pid, ["a.png", "b.png"])
    p = next(p for p in client.get("/api/projects").json() if p["id"] == pid)
    assert p["image_count"] == 2
    assert p["class_count"] == 3


def test_bulk_delete(client):
    ids = [make_project(client, f"B{i}") for i in range(4)]
    r = client.post("/api/projects/bulk-delete", json={"project_ids": ids[:3]})
    assert r.status_code == 200
    assert r.json()["deleted"] == 3
    remaining = {p["id"] for p in client.get("/api/projects").json()}
    assert ids[3] in remaining
    assert not (set(ids[:3]) & remaining)


def test_bulk_delete_tolerates_stale_ids(client):
    """A missing id must not fail the whole batch — the caller wanted it gone,
    and it is."""
    pid = make_project(client, "Real")
    r = client.post("/api/projects/bulk-delete", json={"project_ids": [pid, 999999]})
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == 1
    assert body["not_found"] == [999999]


def test_delete_cascades_to_images_and_classes(client):
    """FK cascade removes an image row when its project goes."""
    from app.models import Annotation, Image

    pid = make_project(client, "Cascade")
    imgs = upload_images(client, pid, ["a.png"])
    img_id = imgs[0]["id"]

    client.delete(f"/api/projects/{pid}")

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        assert db.get(Image, img_id) is None
        assert db.query(Annotation).count() == 0
    finally:
        db.close()
