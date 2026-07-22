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


def test_cancel_discards_run_proposals_and_row(client, monkeypatch):
    """Cancel means the run never happened: its proposals AND its row go.

    The runner sees the flag before loading any model, so a cancel-while-queued
    costs nothing. Proposals from other sources must survive.
    """
    from app.models import AnnotationJob, Annotation, JobStatus
    from app.services import annotation_job as service
    from tests.conftest import add_proposals, make_project, upload_images

    monkeypatch.setattr(service, "SessionLocal", client.SessionLocal)

    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    car = client.get(f"/api/projects/{pid}/classes").json()[0]

    # A proposal from an unrelated source (no job_id) that must survive.
    add_proposals(client, images[0]["id"], car["id"], n=1)

    db = client.SessionLocal()
    try:
        job = AnnotationJob(project_id=pid, model_key="grounding_dino", status=JobStatus.QUEUED)
        db.add(job)
        db.commit()
        job_id = job.id
        # A proposal THIS run produced.
        db.add(
            Annotation(
                image_id=images[0]["id"], category_id=car["id"],
                x=1, y=1, width=5, height=5,
                source="auto", proposed=True, reviewed=False, job_id=job_id,
            )
        )
        db.commit()
    finally:
        db.close()

    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 202

    service.run_annotation_job(job_id)

    db = client.SessionLocal()
    try:
        assert db.get(AnnotationJob, job_id) is None, "cancelled row must be gone"
        remaining = db.query(Annotation).filter(Annotation.proposed.is_(True)).all()
        assert len(remaining) == 1 and remaining[0].job_id is None, (
            "only the run's own proposals are discarded"
        )
    finally:
        db.close()


def test_annotator_roster_covers_five_models_in_four_families(client):
    """Five annotators across four families, each carrying the grouping
    metadata the picker needs. All keys resolve to classes without importing
    any heavy framework (this test would hang for minutes if they did)."""
    from app.ml import registry

    annotators = client.get("/api/annotators").json()
    by_key = {a["key"]: a for a in annotators}

    expected = {
        "grounding_dino": "Grounding DINO",
        "grounding_dino_base": "Grounding DINO",
        "yolo_world_s": "YOLO-World",
        "owlv2_base": "OWLv2",
        "florence2_base": "Florence-2",
    }
    for key, family in expected.items():
        assert key in by_key, f"missing annotator {key}"
        assert by_key[key]["family"] == family
        assert by_key[key]["variant"], f"{key} needs a variant for the picker"
        assert by_key[key]["approx_vram_gb"] > 0
        assert registry.get_class(key)

    assert len({a["family"] for a in annotators}) == 4
