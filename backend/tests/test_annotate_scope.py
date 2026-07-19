"""
Auto-annotate scoping: which images a run covers.

These stub out the actual job runner — the model needs a GPU and 700 MB of
weights, and the logic under test is entirely in the ENDPOINT: it validates the
selection, counts the images, and records what the job will process. The job
itself running is a separate concern, covered by hand when driving the app.
"""

import pytest

from tests.conftest import make_project, upload_images


@pytest.fixture()
def no_run(monkeypatch):
    """Replace the background job with a no-op so queuing a run doesn't try to
    load the model. The row is still created, so we can assert on it."""
    monkeypatch.setattr("app.api.routes.annotate.run_annotation_job", lambda job_id: None)


def test_selection_scopes_the_job(client, no_run):
    pid = make_project(client, "Sel", classes=("car",))
    imgs = upload_images(client, pid, [f"i{i}.png" for i in range(5)])
    chosen = [imgs[1]["id"], imgs[3]["id"]]

    r = client.post(
        f"/api/projects/{pid}/annotate",
        json={"model_key": "grounding_dino", "image_ids": chosen},
    )
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["total_images"] == 2, "the run is scoped to exactly the selection"


def test_foreign_image_id_rejected(client, no_run):
    pid = make_project(client, "A", classes=("car",))
    upload_images(client, pid, ["a.png"])
    other = make_project(client, "B", classes=("car",))
    foreign = upload_images(client, other, ["x.png"])[0]["id"]

    r = client.post(
        f"/api/projects/{pid}/annotate",
        json={"model_key": "grounding_dino", "image_ids": [foreign]},
    )
    assert r.status_code == 400, "an id from another project can't be annotated here"


def test_unannotated_scope_counts_only_gaps(client, no_run):
    pid = make_project(client, "Gaps", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png", "c.png"])
    car = next(c for c in client.get(f"/api/projects/{pid}/classes").json())
    # Annotate one image, leaving two gaps.
    client.post(
        f"/api/images/{imgs[0]['id']}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    )
    pre = client.get(f"/api/projects/{pid}/annotate/preview").json()
    assert pre["scope_counts"]["unannotated"] == 2
    assert pre["scope_counts"]["all"] == 3


def test_unknown_model_key_rejected(client, no_run):
    pid = make_project(client, "BadModel", classes=("car",))
    upload_images(client, pid, ["a.png"])
    r = client.post(
        f"/api/projects/{pid}/annotate",
        json={"model_key": "does_not_exist", "scope": "all"},
    )
    assert r.status_code == 400


def test_empty_project_rejected(client, no_run):
    pid = make_project(client, "NoImages", classes=("car",))
    r = client.post(
        f"/api/projects/{pid}/annotate",
        json={"model_key": "grounding_dino", "scope": "all"},
    )
    assert r.status_code == 400
