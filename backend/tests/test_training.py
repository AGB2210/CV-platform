"""
Training pipeline: readiness preview, launch guards, and the job runner.

Like the annotate-scope tests, the ENDPOINT logic (validation, the one-GPU-job
guard, the readiness preview) is tested with the background runner stubbed out.
The runner itself is then exercised end-to-end against the test DB using a FAKE
trainer — one that asserts the dataset was really exported, emits a couple of
epochs of metrics through the callback, and writes a dummy checkpoint. That
covers everything worth testing (export wiring, per-epoch persistence, the
best-checkpoint record) with no GPU and no heavy deps, exactly the way the suite
stubs the annotation model.
"""

from __future__ import annotations

import pytest

from tests.conftest import make_project, upload_images


# --- helpers ----------------------------------------------------------------


def _class_ids(client, pid) -> list[int]:
    return [c["id"] for c in client.get(f"/api/projects/{pid}/classes").json()]


def _add_box(client, image_id: int, category_id: int) -> None:
    """A human-drawn (accepted, non-proposed) box — real training data."""
    r = client.post(
        f"/api/images/{image_id}/annotations",
        json={"category_id": category_id, "x": 5, "y": 5, "width": 10, "height": 10},
    )
    assert r.status_code == 201, r.text


def _assign_split(client, pid, image_ids, split) -> None:
    r = client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": image_ids, "split": split},
    )
    assert r.status_code == 200, r.text


def make_trainable_project(client, name="Trainable"):
    """A project with classes, 4 train + 2 val images, and boxes on the train set.

    Returns (project_id, images). Images default to the 'train' split, so only
    the val ones need reassigning.
    """
    pid = make_project(client, name, classes=("car", "person"))
    imgs = upload_images(client, pid, [f"i{i}.png" for i in range(6)])
    car = _class_ids(client, pid)[0]
    # Val = last two images; the rest stay 'train'.
    _assign_split(client, pid, [imgs[4]["id"], imgs[5]["id"]], "val")
    # Boxes on the four train images.
    for img in imgs[:4]:
        _add_box(client, img["id"], car)
    return pid, imgs


@pytest.fixture()
def no_train_run(monkeypatch):
    """Stub the background runner so queuing a job doesn't touch the real DB or
    GPU. The row is still created, so route behaviour is assertable."""
    monkeypatch.setattr("app.api.routes.train.run_training_job", lambda job_id: None)


@pytest.fixture()
def fake_trainer():
    """Register a dependency-free trainer for the duration of one test.

    It stands in for a real backend (YOLO, RF-DETR): it verifies the runner
    handed it a real exported dataset, reports synthetic epoch metrics through
    the callback, and writes a placeholder checkpoint. Removed on teardown so it
    never leaks into other tests' view of /api/trainers.
    """
    from app.ml.trainers import registry
    from app.ml.trainers.base import EpochMetrics, TrainConfig, TrainResult, Trainer

    received: list[TrainConfig] = []

    class FakeTrainer(Trainer):
        key = "fake"
        display_name = "Fake Trainer"
        description = "Test double — emits metrics, writes no real weights."
        approx_vram_gb = 0.0
        export_format = "yolo"
        default_epochs = 2
        default_batch_size = 2
        default_image_size = 64

        def train(self, config: TrainConfig, on_epoch) -> TrainResult:
            received.append(config)
            # The runner must have exported a real dataset before calling us.
            assert (config.dataset_dir / "data.yaml").exists(), "dataset not exported"
            for e in range(1, config.epochs + 1):
                on_epoch(
                    EpochMetrics(
                        epoch=e,
                        total_epochs=config.epochs,
                        train_loss=1.0 / e,
                        val_map=0.1 * e,
                        val_map50=0.2 * e,
                    )
                )
            ckpt = config.output_dir / "best.pt"
            ckpt.write_text("fake weights")
            return TrainResult(
                best_checkpoint_path=ckpt,
                best_map=0.1 * config.epochs,
                epochs_completed=config.epochs,
            )

    # Expose the captured configs on the class so tests can assert on what the
    # runner passed. Set after the class body to avoid the assignment-makes-it-
    # local trap inside it.
    FakeTrainer.received = received  # type: ignore[attr-defined]
    registry.register(FakeTrainer)
    yield FakeTrainer
    registry._REGISTRY.pop("fake", None)


# --- capability + readiness -------------------------------------------------


def test_trainers_lists_registered_backends(client):
    """The dropdown is fetched, not hardcoded: whatever trainers are registered
    show up, each carrying the metadata and form defaults the UI needs. YOLO is
    registered in Phase 4b."""
    r = client.get("/api/trainers")
    assert r.status_code == 200
    keys = {t["key"] for t in r.json()}
    assert "yolo" in keys
    yolo = next(t for t in r.json() if t["key"] == "yolo")
    assert yolo["export_format"] == "yolo"
    assert yolo["default_epochs"] > 0 and yolo["default_image_size"] > 0


def test_preview_reports_split_readiness(client):
    pid, _ = make_trainable_project(client)
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["num_classes"] == 2
    assert p["splits"]["train"] == {"images": 4, "boxes": 4}
    assert p["splits"]["val"] == {"images": 2, "boxes": 0}
    assert p["can_train"] is True


def test_preview_warns_when_no_val_split(client):
    pid = make_project(client, "NoVal", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    _add_box(client, imgs[0]["id"], _class_ids(client, pid)[0])
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["can_train"] is True  # trainable, just not measurable
    assert any("validation" in w.lower() for w in p["warnings"])


def test_preview_warns_on_tiny_train_set(client):
    """A handful of images can't fine-tune a detector — mAP will sit at ~0. The
    preview must say so, so a small-data run doesn't just look broken."""
    pid, _ = make_trainable_project(client)  # 4 train images
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["can_train"] is True  # not blocked, just warned
    assert any("too few" in w.lower() for w in p["warnings"])


def test_preview_no_tiny_warning_on_larger_set(client):
    """The warning must not nag once the set is a reasonable size, or it teaches
    people to ignore warnings."""
    pid = make_project(client, "Big", classes=("car",))
    imgs = upload_images(client, pid, [f"i{i}.png" for i in range(14)])
    car = _class_ids(client, pid)[0]
    _assign_split(client, pid, [imgs[12]["id"], imgs[13]["id"]], "val")
    for img in imgs[:12]:
        _add_box(client, img["id"], car)
    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert not any("too few" in w.lower() for w in p["warnings"])


# --- launch guards ----------------------------------------------------------


def test_train_rejects_empty_train_split(client, no_train_run, fake_trainer):
    """No accepted boxes in train = nothing to learn from = 400, no job row."""
    pid = make_project(client, "Empty", classes=("car",))
    upload_images(client, pid, ["a.png"])  # image, but no boxes
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 400
    assert client.get(f"/api/projects/{pid}/training-jobs").json() == []


def test_train_rejects_unknown_trainer(client, no_train_run):
    pid, _ = make_trainable_project(client)
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "nope"})
    assert r.status_code == 400


def test_train_conflicts_with_active_training(client, no_train_run, fake_trainer):
    pid, _ = make_trainable_project(client)
    first = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert first.status_code == 202
    # A second while the first is queued/running must be refused.
    second = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert second.status_code == 409


def test_train_conflicts_with_active_annotation(client, no_train_run, fake_trainer):
    """Annotation and training both want the whole 4 GB card — refuse to start
    training while an annotate job is in flight for the project."""
    from app.models import AnnotationJob, JobStatus

    pid, _ = make_trainable_project(client)
    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        db.add(
            AnnotationJob(
                project_id=pid,
                model_key="grounding_dino",
                status=JobStatus.RUNNING,
                total_images=1,
            )
        )
        db.commit()
    finally:
        db.close()

    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 409


# --- the runner, end to end -------------------------------------------------


def test_run_exports_records_metrics_and_checkpoint(client, monkeypatch, fake_trainer):
    """Full path: POST queues a job, the background runner exports the dataset,
    drives the (fake) trainer, and persists per-epoch metrics + the checkpoint.

    The runner opens its OWN session from app.database.SessionLocal, which points
    at the real DB — so we repoint it at the test's session factory, the same
    trick the app uses via dependency_overrides but for the background path.
    """
    from pathlib import Path

    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )

    pid, _ = make_trainable_project(client)
    r = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": "fake", "epochs": 2, "batch_size": 2, "image_size": 64},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]

    # TestClient runs the background task synchronously, so by now it's finished.
    job = client.get(f"/api/training-jobs/{job_id}").json()
    assert job["status"] == "done", job.get("error")
    assert job["current_epoch"] == 2
    assert job["train_images"] == 4 and job["val_images"] == 2
    assert job["best_map"] == pytest.approx(0.2)

    # Per-epoch history came through the callback, parsed into an array.
    assert [m["epoch"] for m in job["metrics"]] == [1, 2]
    assert job["metrics"][-1]["val_map"] == pytest.approx(0.2)

    # The checkpoint was recorded and actually exists on disk.
    assert job["checkpoint_path"]
    assert Path(job["checkpoint_path"]).exists()


def test_finetune_rejects_unusable_source(client, no_train_run, fake_trainer):
    """Can't continue from a run that doesn't exist or never produced weights."""
    pid, _ = make_trainable_project(client)
    # Non-existent source.
    r = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": "fake", "init_from_job_id": 999},
    )
    assert r.status_code == 400


def test_finetune_continues_from_prior_checkpoint(client, monkeypatch, fake_trainer):
    """A second run started 'from' the first is handed the first's checkpoint as
    init_weights — building on it instead of the pretrained base."""
    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client)

    first = client.post(
        f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 2}
    ).json()
    assert client.get(f"/api/training-jobs/{first['id']}").json()["status"] == "done"
    ckpt = client.get(f"/api/training-jobs/{first['id']}").json()["checkpoint_path"]

    second = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": "fake", "epochs": 2, "init_from_job_id": first["id"]},
    )
    assert second.status_code == 202, second.text
    second_job = client.get(f"/api/training-jobs/{second.json()['id']}").json()
    assert second_job["status"] == "done"
    assert second_job["init_from_job_id"] == first["id"]

    # The runner passed the first run's checkpoint to the trainer as init_weights.
    last_config = fake_trainer.received[-1]
    assert last_config.init_weights is not None
    assert str(last_config.init_weights) == ckpt
