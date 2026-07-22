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

from app.config import from_storage_path

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


def save_dataset(client, pid, note=None) -> dict:
    """Click "Save dataset" — training runs against a saved version, so tests
    that train must save first, exactly as the UI requires."""
    r = client.post(f"/api/projects/{pid}/dataset/versions", json={"note": note})
    assert r.status_code == 201, r.text
    return r.json()


def make_trainable_project(client, name="Trainable"):
    """A project with classes, 4 train + 2 val images, boxes on the train set,
    and the dataset saved as v1.

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
    save_dataset(client, pid, note="test fixture")
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


def test_no_val_split_blocks_training(client, no_train_run, fake_trainer):
    """A validation split is REQUIRED, not advisory.

    Without held-out images the reported mAP is scored on the data the model
    just trained on. That isn't a poor estimate of generalisation, it's not one
    at all — and it reads HIGH, so a memorised model looks like an excellent
    one. This used to be a warning beside an enabled button.
    """
    pid = make_project(client, "NoVal", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    _add_box(client, imgs[0]["id"], _class_ids(client, pid)[0])
    save_dataset(client, pid)

    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["can_train"] is False
    assert any("validation" in w.lower() for w in p["warnings"])

    # The button is disabled AND the endpoint refuses — a disabled control is a
    # courtesy, not a guarantee.
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 400
    assert "validation" in r.json()["detail"].lower()
    assert client.get(f"/api/projects/{pid}/training-jobs").json() == []


def test_val_split_makes_it_trainable(client):
    """The same project becomes trainable once val images exist."""
    pid = make_project(client, "GetsVal", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    _add_box(client, imgs[0]["id"], _class_ids(client, pid)[0])
    _assign_split(client, pid, [imgs[1]["id"]], "val")
    save_dataset(client, pid)

    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["can_train"] is True
    assert not any("validation" in w.lower() for w in p["warnings"])


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


def test_can_train_describes_the_version_train_would_actually_run(
    client, no_train_run, fake_trainer
):
    """Readiness must be judged on the version an unqualified Train would use.

    THE BUG THIS GUARDS: the preview computed can_train from the NEWEST version
    while the route trained the CURRENT one. Restore an older version with an
    empty train split while a newer one has boxes, and the page enabled a button
    whose request came straight back as a 400.
    """
    pid = make_project(client, "Drift", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])

    # v1: images, no boxes — nothing to learn from.
    v1 = save_dataset(client, pid)

    # v2: a box in the train split.
    _add_box(client, imgs[0]["id"], _class_ids(client, pid)[0])
    save_dataset(client, pid)

    # Roll back to the empty one. The live dataset IS v1 now.
    client.post(f"/api/projects/{pid}/dataset/versions/{v1['id']}/restore")

    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["current_version"] == 1, "the restored version is on screen"
    assert p["latest_version"] == 2, "the newer save point still exists"
    assert p["can_train"] is False, "v1 has nothing to train on"

    # And the two agree: a disabled button matches a rejected request.
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 400


# --- launch guards ----------------------------------------------------------


def test_train_rejects_empty_train_split(client, no_train_run, fake_trainer):
    """No boxes in the saved version's train split = nothing to learn = 400."""
    pid = make_project(client, "Empty", classes=("car",))
    upload_images(client, pid, ["a.png"])  # image, but no boxes
    save_dataset(client, pid)
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 400
    assert client.get(f"/api/projects/{pid}/training-jobs").json() == []


def test_train_requires_a_saved_dataset(client, no_train_run, fake_trainer):
    """The gate: you train a SAVED dataset, so an unsaved project is refused
    with an instruction rather than trained against whatever the rows are."""
    pid = make_project(client, "Unsaved", classes=("car",))
    imgs = upload_images(client, pid, ["a.png"])
    _add_box(client, imgs[0]["id"], _class_ids(client, pid)[0])
    # Deliberately NOT saved.
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert r.status_code == 400
    assert "save the dataset" in r.json()["detail"].lower()

    p = client.get(f"/api/projects/{pid}/train/preview").json()
    assert p["can_train"] is False and p["has_saved_version"] is False


def test_train_rejects_unknown_trainer(client, no_train_run):
    pid, _ = make_trainable_project(client)
    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "nope"})
    assert r.status_code == 400


def test_overlapping_training_queues_instead_of_409(client, no_train_run, fake_trainer):
    """The old contract (409 while another job is in flight) is GONE: a second
    start is accepted and holds in QUEUED, where the runner's GPU-admission
    loop will start it once the card frees up (tests/test_gpu_admission.py
    covers the admission logic itself)."""
    pid, _ = make_trainable_project(client)
    first = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert first.status_code == 202
    second = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"})
    assert second.status_code == 202
    assert second.json()["status"] == "queued"


def test_training_queues_while_annotation_runs(client, no_train_run, fake_trainer):
    """Same across job kinds: an annotate run in flight no longer blocks the
    POST — the training job queues behind it."""
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
    assert r.status_code == 202
    assert r.json()["status"] == "queued"


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
    # Paths are stored relative to storage/ for portability.
    assert from_storage_path(job["checkpoint_path"]).exists()


def test_versions_number_per_project_and_model(client, monkeypatch, fake_trainer):
    """Version is 1-based per (project, trainer): consecutive runs increment, and
    a different project restarts at 1 — not the global row id."""
    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client, "VerA")
    # TestClient runs the (fake) run to completion synchronously, so each POST is
    # free to queue the next without tripping the one-GPU-job guard.
    v1 = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    v2 = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    assert (v1["version"], v2["version"]) == (1, 2)

    other, _ = make_trainable_project(client, "VerB")
    v = client.post(f"/api/projects/{other}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    assert v["version"] == 1, "a different project's versions restart at 1"


def test_training_an_older_version_uses_that_version_data(client, monkeypatch, fake_trainer):
    """Picking dataset v1 trains v1's snapshot, not the live rows.

    This is the whole provenance claim: a run that says "trained on dataset v1"
    must be describing what actually went into it, even though the dataset has
    moved on since.
    """
    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client)  # v1: 4 train + 2 val
    v1 = client.get(f"/api/projects/{pid}/dataset/versions").json()[0]
    assert v1["version"] == 1 and v1["train_images"] == 4

    # The dataset moves on: three more train images, saved as v2.
    later = upload_images(client, pid, ["x1.png", "x2.png", "x3.png"])
    car = _class_ids(client, pid)[0]
    for img in later:
        _add_box(client, img["id"], car)
    v2 = save_dataset(client, pid)
    assert v2["train_images"] == 7

    # Train the OLD version explicitly.
    r = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": "fake", "epochs": 1, "dataset_version_id": v1["id"]},
    )
    assert r.status_code == 202, r.text
    job = client.get(f"/api/training-jobs/{r.json()['id']}").json()
    assert job["status"] == "done", job.get("error")
    assert job["dataset_version_id"] == v1["id"]
    assert job["train_images"] == 4, "trained v1's 4 images, not the live 7"


def test_train_defaults_to_the_latest_saved_version(client, no_train_run, fake_trainer):
    pid, _ = make_trainable_project(client)
    upload_images(client, pid, ["extra.png"])
    v2 = save_dataset(client, pid)
    job = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"}).json()
    assert job["dataset_version_id"] == v2["id"], "no version given => newest save"


def test_metrics_never_record_an_epoch_twice(client, monkeypatch):
    """A framework that re-fires its per-epoch hook must not corrupt the curve.

    Ultralytics does exactly this: one extra call after the loop, by which point
    its epoch counter has advanced — producing a phantom point with the previous
    epoch's loss, so the chart repeated a value and its axis ran past the epochs
    actually trained. The adapter filters those, and the runner keys history by
    epoch so it can't happen regardless of which trainer misbehaves.
    """
    from app.ml.trainers import registry
    from app.ml.trainers.base import EpochMetrics, TrainConfig, TrainResult, Trainer

    class RepeatingTrainer(Trainer):
        key = "repeat"
        display_name = "Repeats epochs"
        description = "Test double that re-reports epochs, as ultralytics does."
        approx_vram_gb = 0.0
        export_format = "yolo"
        default_epochs = 3

        def train(self, config: TrainConfig, on_epoch) -> TrainResult:
            for e in (1, 2, 3):
                on_epoch(EpochMetrics(epoch=e, total_epochs=3, train_loss=1.0 / e, val_map=0.1 * e))
            # The phantom: same epoch again, and one past the schedule.
            on_epoch(EpochMetrics(epoch=3, total_epochs=3, train_loss=99.0, val_map=0.9))
            ckpt = config.output_dir / "best.pt"
            ckpt.write_text("w")
            return TrainResult(best_checkpoint_path=ckpt, best_map=0.3, epochs_completed=3)

    registry.register(RepeatingTrainer)
    try:
        monkeypatch.setattr(
            "app.services.training_job.SessionLocal",
            client.SessionLocal,  # type: ignore[attr-defined]
        )
        pid, _ = make_trainable_project(client)
        r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "repeat", "epochs": 3})
        job = client.get(f"/api/training-jobs/{r.json()['id']}").json()

        epochs = [m["epoch"] for m in job["metrics"]]
        assert epochs == [1, 2, 3], "one point per epoch, no phantom and no duplicate"
        assert len(set(epochs)) == len(epochs)
        # The repeat REPLACED epoch 3 rather than appending beside it.
        assert job["metrics"][-1]["train_loss"] == 99.0
    finally:
        registry._REGISTRY.pop("repeat", None)


# --- stopping and cancelling a run ------------------------------------------


@pytest.fixture()
def controllable_trainer():
    """A trainer that asks the runner, each epoch, whether to keep going — and
    obeys. Stands in for a real backend's early-stop flag."""
    from app.ml.trainers import registry
    from app.ml.trainers.base import EpochMetrics, TrainConfig, TrainResult, Trainer

    class ControllableTrainer(Trainer):
        key = "ctl"
        display_name = "Controllable"
        description = "Test double honouring the stop signal."
        approx_vram_gb = 0.0
        export_format = "yolo"
        default_epochs = 10
        default_batch_size = 2
        default_image_size = 64

        def train(self, config: TrainConfig, on_epoch) -> TrainResult:
            completed = 0
            for e in range(1, config.epochs + 1):
                completed = e
                stop = on_epoch(
                    EpochMetrics(epoch=e, total_epochs=config.epochs, val_map=0.1 * e)
                )
                if stop:
                    break  # finish THIS epoch, then stop — never mid-epoch
            ckpt = config.output_dir / "best.pt"
            ckpt.write_text("weights")
            return TrainResult(
                best_checkpoint_path=ckpt, best_map=0.1 * completed, epochs_completed=completed
            )

    registry.register(ControllableTrainer)
    yield ControllableTrainer
    registry._REGISTRY.pop("ctl", None)


def test_stop_finishes_current_epoch_and_keeps_the_model(client, monkeypatch, controllable_trainer):
    """Stop at epoch 1 of 10: the epoch in flight completes, the run ends there,
    and the version survives with a usable checkpoint."""
    from pathlib import Path

    from app.models import JobControl, TrainingJob

    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client)

    # Pre-set the stop flag by intercepting the queue, so it's already pending
    # when the first epoch reports — the runner should then stop at epoch 1.
    real_runner = None

    def queue(job_id: int):
        db = client.SessionLocal()  # type: ignore[attr-defined]
        try:
            job = db.get(TrainingJob, job_id)
            job.control = JobControl.STOP
            db.commit()
        finally:
            db.close()
        real_runner(job_id)

    from app.services.training_job import run_training_job as _real

    real_runner = _real
    monkeypatch.setattr("app.api.routes.train.run_training_job", queue)

    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "ctl", "epochs": 10})
    job = client.get(f"/api/training-jobs/{r.json()['id']}").json()

    assert job["status"] == "done", job.get("error")
    assert job["stopped_early"] is True
    assert job["current_epoch"] == 1, "stopped after the epoch in flight, not at 10"
    assert job["total_epochs"] == 10, "the schedule it was asked for is still recorded"
    # Paths are stored relative to storage/ for portability.
    assert from_storage_path(job["checkpoint_path"]).exists(), "a stopped run still yields a model"


def test_cancel_discards_the_run_entirely(client, monkeypatch, controllable_trainer):
    """Cancel: no version is kept and the run's directory is removed."""
    from app.models import JobControl, TrainingJob

    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client)

    from app.services.training_job import run_training_job as _real

    def queue(job_id: int):
        db = client.SessionLocal()  # type: ignore[attr-defined]
        try:
            job = db.get(TrainingJob, job_id)
            job.control = JobControl.CANCEL
            db.commit()
        finally:
            db.close()
        _real(job_id)

    monkeypatch.setattr("app.api.routes.train.run_training_job", queue)

    r = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "ctl", "epochs": 10})
    job_id = r.json()["id"]

    assert client.get(f"/api/training-jobs/{job_id}").status_code == 404, "no version kept"
    assert client.get(f"/api/projects/{pid}/training-jobs").json() == []

    from app.config import settings

    assert not (settings.runs_dir / str(job_id)).exists(), "its output was discarded"


def test_cancelled_version_number_is_reused(client, monkeypatch, controllable_trainer, fake_trainer):
    """A cancelled run leaves no gap: numbering counts what exists, so the next
    run takes the number the cancelled one was using."""
    from app.models import JobControl, TrainingJob

    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client)
    from app.services.training_job import run_training_job as _real

    def cancelling_queue(job_id: int):
        db = client.SessionLocal()  # type: ignore[attr-defined]
        try:
            db.get(TrainingJob, job_id).control = JobControl.CANCEL
            db.commit()
        finally:
            db.close()
        _real(job_id)

    monkeypatch.setattr("app.api.routes.train.run_training_job", cancelling_queue)
    first = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "ctl"}).json()
    assert first["version"] == 1

    monkeypatch.setattr("app.api.routes.train.run_training_job", _real)
    second = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "ctl"}).json()
    assert second["version"] == 1, "the cancelled run's number is free again"


def test_stop_rejects_a_finished_run(client, monkeypatch, fake_trainer):
    pid, _ = make_trainable_project(client)
    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    job = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    r = client.post(f"/api/training-jobs/{job['id']}/stop")
    assert r.status_code == 409
    assert "nothing to stop" in r.json()["detail"]


# --- renaming and deleting model versions -----------------------------------


def _make_two_versions(client, monkeypatch, pid):
    """Two completed runs, so rename/delete have something to act on."""
    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    a = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    b = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake", "epochs": 1}).json()
    return a, b


def test_rename_model_version_and_clear(client, monkeypatch, fake_trainer):
    pid, _ = make_trainable_project(client)
    a, _ = _make_two_versions(client, monkeypatch, pid)

    r = client.patch(f"/api/training-jobs/{a['id']}", json={"name": "  baseline "})
    assert r.status_code == 200 and r.json()["name"] == "baseline"
    assert client.patch(f"/api/training-jobs/{a['id']}", json={"name": ""}).json()["name"] is None


def test_rename_model_version_rejects_duplicates(client, monkeypatch, fake_trainer):
    pid, _ = make_trainable_project(client)
    a, b = _make_two_versions(client, monkeypatch, pid)
    client.patch(f"/api/training-jobs/{a['id']}", json={"name": "baseline"})

    assert client.patch(f"/api/training-jobs/{b['id']}", json={"name": "baseline"}).status_code == 409
    assert client.patch(f"/api/training-jobs/{b['id']}", json={"name": "BaseLine"}).status_code == 409
    # Clashing with the other version's numeric label is a duplicate too.
    assert client.patch(f"/api/training-jobs/{b['id']}", json={"name": "v1"}).status_code == 409


def test_delete_model_version_removes_its_run_directory(client, monkeypatch, fake_trainer):
    from pathlib import Path

    pid, _ = make_trainable_project(client)
    a, b = _make_two_versions(client, monkeypatch, pid)
    stored = client.get(f"/api/training-jobs/{a['id']}").json()["checkpoint_path"]
    ckpt = from_storage_path(stored)
    assert ckpt.exists()

    assert client.delete(f"/api/training-jobs/{a['id']}").status_code == 204
    assert not ckpt.exists(), "checkpoints and the exported dataset go with the version"
    remaining = [j["id"] for j in client.get(f"/api/projects/{pid}/training-jobs").json()]
    assert remaining == [b["id"]]


def test_cannot_delete_a_version_still_training(client, no_train_run, fake_trainer):
    """The runner is stubbed, so this job stays queued — deleting it out from
    under a live run would leave the runner writing to a deleted row."""
    pid, _ = make_trainable_project(client)
    job = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"}).json()
    r = client.delete(f"/api/training-jobs/{job['id']}")
    assert r.status_code == 409
    assert "still training" in r.json()["detail"]


def test_bulk_delete_model_versions_skips_in_flight(client, monkeypatch, fake_trainer):
    """Deleting nine finished versions shouldn't be blocked by a tenth that's
    still running — it's skipped and reported, not fatal."""
    pid, _ = make_trainable_project(client)
    a, b = _make_two_versions(client, monkeypatch, pid)
    # A third that never completes: point the runner back at a no-op.
    monkeypatch.setattr("app.api.routes.train.run_training_job", lambda job_id: None)
    live = client.post(f"/api/projects/{pid}/train", json={"trainer_key": "fake"}).json()

    r = client.post(
        f"/api/projects/{pid}/training-jobs/bulk-delete",
        json={"job_ids": [a["id"], b["id"], live["id"], 9999]},
    ).json()
    assert r["deleted"] == 2
    assert r["not_found"] == [9999]
    assert r["skipped"] == {str(live["id"]): "still training"}


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
    # Paths are stored relative to storage/; the runner resolves them back.
    ckpt = from_storage_path(
        client.get(f"/api/training-jobs/{first['id']}").json()["checkpoint_path"]
    )

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
    # The runner resolves the stored relative path back to a real one.
    assert last_config.init_weights == ckpt


# --- live logs ---------------------------------------------------------------


def test_training_logs_capture_and_endpoint(client, monkeypatch, fake_trainer):
    """A run's narration is tailable while it runs and after it finishes;
    framework logger records are captured; a missing job 404s."""
    import logging

    from app.services import training_logs

    monkeypatch.setattr(
        "app.services.training_job.SessionLocal",
        client.SessionLocal,  # type: ignore[attr-defined]
    )
    pid, _ = make_trainable_project(client, "Logs")
    r = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": "fake", "epochs": 2, "batch_size": 2, "image_size": 64},
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]

    lines = client.get(f"/api/training-jobs/{job_id}/logs").json()["lines"]
    text = "\n".join(lines)
    assert "Exporting dataset" in text
    assert "Epoch 1/" in text, f"per-epoch line missing from: {lines}"
    assert "Done:" in text

    # Framework narration: anything logged on the captured logger during a run
    # lands in the buffer. Simulate directly against the context manager.
    with training_logs.capture_framework_logs(999001):
        logging.getLogger("ultralytics").info("train: Scanning labels... 8 found")
    assert any("Scanning labels" in ln for ln in training_logs.tail(999001))
    # After the capture window, the same logger no longer reaches the buffer.
    logging.getLogger("ultralytics").info("stray line after run")
    assert not any("stray line" in ln for ln in training_logs.tail(999001))
    training_logs.discard(999001)
    assert training_logs.tail(999001) == []

    assert client.get("/api/training-jobs/999999/logs").status_code == 404


# --- the model roster --------------------------------------------------------


def test_roster_covers_yolo_family_and_rtdetr(client):
    """Six ultralytics trainers, grouped by family, with 'yolo' keeping its
    historic key (existing job rows reference it)."""
    trainers = client.get("/api/trainers").json()
    by_key = {t["key"]: t for t in trainers}

    for key in ("yolo", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "rtdetr_l"):
        assert key in by_key, f"missing trainer {key}"

    assert by_key["yolo"]["family"] == "YOLO11" and by_key["yolo"]["variant"] == "nano"
    assert by_key["rtdetr_l"]["family"] == "RT-DETR"
    # The ladder is ordered: VRAM demand grows with size within the family.
    yolo_family = [t for t in trainers if t["family"] == "YOLO11"]
    assert len(yolo_family) == 5
    vrams = [by_key[k]["approx_vram_gb"] for k in ("yolo", "yolo11s", "yolo11m", "yolo11l", "yolo11x")]
    assert vrams == sorted(vrams)


def test_oom_failure_gets_actionable_message(client, monkeypatch):
    """A CUDA OOM ends the run FAILED with settings advice up front — not a
    raw allocator traceback, and no silent retry at a smaller batch."""
    from app.ml.trainers import registry
    from app.ml.trainers.base import TrainResult, Trainer

    class OomTrainer(Trainer):
        key = "oomer"
        display_name = "OOM Trainer"
        description = "dies of memory"
        export_format = "yolo"
        default_epochs = 1
        default_batch_size = 4
        default_image_size = 64

        def train(self, config, on_epoch) -> TrainResult:
            # torch re-wraps OOM in RuntimeError often enough that the message
            # is the reliable signal — that's what the guard keys on.
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")

    registry.register(OomTrainer)
    try:
        monkeypatch.setattr(
            "app.services.training_job.SessionLocal",
            client.SessionLocal,  # type: ignore[attr-defined]
        )
        pid, _ = make_trainable_project(client, "OomProj")
        r = client.post(
            f"/api/projects/{pid}/train",
            json={"trainer_key": "oomer", "epochs": 1, "batch_size": 4, "image_size": 64},
        )
        assert r.status_code == 202, r.text
        job = client.get(f"/api/training-jobs/{r.json()['id']}").json()
        assert job["status"] == "failed"
        assert job["error"].startswith("Out of GPU memory"), job["error"]
        assert "batch size (currently 4)" in job["error"]
        assert "original error" in job["error"]
    finally:
        registry._REGISTRY.pop("oomer", None)
