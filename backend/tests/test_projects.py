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


# --- deleting a project must not leak its bytes -----------------------------


def test_deleting_a_project_removes_all_of_its_files(client, tmp_path):
    """A project owns bytes in THREE places, and delete cleaned up one.

    THE LEAK THIS GUARDS: images/, versions/ and runs/ all belong to a project.
    The rows for all three cascade away with it, so nothing dangles in the
    database and the leak is invisible from inside the app — but the files
    stayed forever. A project with ten training runs left ~50 MB of checkpoints
    behind, and the only way to notice was to look at the disk.
    """
    from app.config import settings
    from app.models import TrainingJob
    from tests.conftest import make_project, upload_images

    pid = make_project(client, "Leaky", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    client.post(
        f"/api/images/{imgs[0]['id']}/annotations",
        json={"category_id": client.get(f"/api/projects/{pid}/classes").json()[0]["id"],
              "x": 1, "y": 1, "width": 10, "height": 10},
    )
    client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})

    # A finished training run, with the run directory one would leave on disk.
    db = client.SessionLocal()  # type: ignore[attr-defined]
    job = TrainingJob(project_id=pid, trainer_key="yolo", version=1, status="done")
    db.add(job)
    db.commit()
    job_id = job.id
    db.close()

    run_dir = settings.runs_dir / str(job_id)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "best.pt").write_text("fake weights")

    images_dir = settings.images_dir / str(pid)
    versions_dir = settings.versions_dir / str(pid)
    assert images_dir.exists() and versions_dir.exists() and run_dir.exists()

    assert client.delete(f"/api/projects/{pid}").status_code == 204

    assert not images_dir.exists(), "uploads left behind"
    assert not versions_dir.exists(), "dataset snapshots left behind"
    assert not run_dir.exists(), "training checkpoints left behind"


def test_bulk_delete_also_removes_every_projects_files(client):
    """Same cleanup on the bulk path — it's how "delete all" is sent."""
    from app.config import settings
    from tests.conftest import make_project, upload_images

    a = make_project(client, "One", classes=("car",))
    b = make_project(client, "Two", classes=("car",))
    for pid in (a, b):
        upload_images(client, pid, ["x.png"])
        client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})

    r = client.post("/api/projects/bulk-delete", json={"project_ids": [a, b]})
    assert r.json()["deleted"] == 2

    for pid in (a, b):
        assert not (settings.images_dir / str(pid)).exists()
        assert not (settings.versions_dir / str(pid)).exists()


def test_delete_project_removes_imported_weights_files(client):
    """Every storage location a project owns goes with it — including the
    imported-checkpoints folder added in 1.1.0. Found by deleting a real
    project and finding the uploaded .pt still on disk: the DB row cascaded,
    the file leaked, and nothing inside the app could ever show it."""
    from app.config import settings
    from tests.conftest import make_project

    pid = make_project(client, "WeightsLeak", classes=("car",))
    r = client.post(
        f"/api/projects/{pid}/weights", files={"file": ("ext.pt", b"WEIGHTBYTES")}
    )
    assert r.status_code == 201, r.text

    weights_dir = settings.STORAGE_DIR / "imported_weights" / str(pid)
    assert weights_dir.exists() and any(weights_dir.iterdir())

    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert not weights_dir.exists(), "the uploaded checkpoint must not outlive its project"
