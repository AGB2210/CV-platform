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


# --- name uniqueness --------------------------------------------------------
# One rule everywhere: names are compared stripped and case-folded. These had
# drifted into three different answers — see services/naming.py.


def test_duplicate_project_name_rejected(client):
    assert client.post("/api/projects", json={"name": "Street Scenes"}).status_code == 201
    r = client.post("/api/projects", json={"name": "Street Scenes"})
    assert r.status_code == 409


def test_project_name_duplicate_check_ignores_case_and_space(client):
    client.post("/api/projects", json={"name": "Street Scenes"})
    for variant in ("street scenes", "STREET SCENES", "  Street Scenes  "):
        r = client.post("/api/projects", json={"name": variant})
        assert r.status_code == 409, f"{variant!r} should collide"


def test_renaming_a_project_to_an_existing_name_is_rejected(client):
    a = client.post("/api/projects", json={"name": "Alpha"}).json()
    client.post("/api/projects", json={"name": "Beta"})
    assert client.patch(f"/api/projects/{a['id']}", json={"name": "beta"}).status_code == 409
    # ...but renaming itself, including just its capitalisation, is fine.
    assert client.patch(f"/api/projects/{a['id']}", json={"name": "ALPHA"}).status_code == 200


def test_duplicate_class_name_rejected_case_insensitively(client):
    """THE bug this guards: the DB unique constraint is case-SENSITIVE in
    SQLite, so "car" and "Car" both existed — and then export as two separate
    classes, teaching the model to split one concept in half."""
    pid = client.post("/api/projects", json={"name": "Classy"}).json()["id"]
    assert client.post(f"/api/projects/{pid}/classes", json={"name": "car"}).status_code == 201
    for variant in ("Car", "CAR", " car "):
        r = client.post(f"/api/projects/{pid}/classes", json={"name": variant})
        assert r.status_code == 409, f"{variant!r} should collide"
    assert len(client.get(f"/api/projects/{pid}/classes").json()) == 1


def test_same_class_name_in_a_different_project_is_fine(client):
    """Class names are scoped to their project — "car" belongs in many."""
    a = client.post("/api/projects", json={"name": "A"}).json()["id"]
    b = client.post("/api/projects", json={"name": "B"}).json()["id"]
    assert client.post(f"/api/projects/{a}/classes", json={"name": "car"}).status_code == 201
    assert client.post(f"/api/projects/{b}/classes", json={"name": "car"}).status_code == 201


def test_renaming_a_class_to_fix_its_own_capitalisation(client):
    pid = client.post("/api/projects", json={"name": "Fixup"}).json()["id"]
    cid = client.post(f"/api/projects/{pid}/classes", json={"name": "car"}).json()["id"]
    r = client.patch(f"/api/classes/{cid}", json={"name": "Car"})
    assert r.status_code == 200, "a class does not collide with itself"
    assert r.json()["name"] == "Car"
