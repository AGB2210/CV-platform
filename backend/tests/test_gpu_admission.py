"""
GPU admission — jobs queue and wait for real resources instead of colliding.

The behaviour under test: starting a second training/annotation (any project)
while one is running no longer errors OR overlaps. The new job holds in QUEUED
with a live "waiting for GPU" reason built from actual free-VRAM numbers, and
starts when the resources exist. FIFO among waiters; CPU mode degrades to
one-job-at-a-time; cancelling a waiting job releases it.

nvidia-smi is never invoked here — free_vram_gb is stubbed per test, which is
exactly the seam the design intends (the rest of the module is pure logic).
"""

from __future__ import annotations

import pytest

from tests.test_training import make_trainable_project


@pytest.fixture()
def no_train_run(monkeypatch):
    """Stub the background runner: POST creates the QUEUED row and nothing
    runs — exactly the state a waiting job holds in."""
    monkeypatch.setattr("app.api.routes.train.run_training_job", lambda job_id: None)


@pytest.fixture()
def fake_trainer():
    """A registered, dependency-free trainer so the route's key check passes."""
    from app.ml.trainers import registry
    from app.ml.trainers.base import TrainResult, Trainer

    class FakeAdmissionTrainer(Trainer):
        key = "fake"
        display_name = "Fake"
        description = "test double"
        approx_vram_gb = 3.0
        export_format = "yolo"
        default_epochs = 1
        default_batch_size = 2
        default_image_size = 64

        def train(self, config, on_epoch) -> TrainResult:  # pragma: no cover
            raise NotImplementedError

    registry.register(FakeAdmissionTrainer)
    yield FakeAdmissionTrainer
    registry._REGISTRY.pop("fake", None)


def _queued_training_job(client, pid, key="fake", **params):
    """POST a training run with the runner stubbed, leaving a QUEUED row."""
    r = client.post(
        f"/api/projects/{pid}/train",
        json={"trainer_key": key, "epochs": 1, "batch_size": 2, "image_size": 64, **params},
    )
    assert r.status_code == 202, r.text
    return r.json()["id"]


def _mark_running(client, job_id: int) -> None:
    """Flip a queued row to RUNNING, as if its runner had started."""
    from app.models import JobStatus, TrainingJob

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        db.get(TrainingJob, job_id).status = JobStatus.RUNNING
        db.commit()
    finally:
        db.close()


def test_check_admits_when_vram_is_plentiful(client, monkeypatch, no_train_run, fake_trainer):
    from app.services import gpu_admission

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: 8.0)
    pid, _ = make_trainable_project(client, "AdmitA")
    job_id = _queued_training_job(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        job = db.get(TrainingJob, job_id)
        admitted, reason = gpu_admission.check(db, "training", job.id, job.created_at, 3.0)
        assert admitted is True and reason is None
    finally:
        db.close()


def test_lone_job_is_admitted_whatever_the_vram_says(
    client, monkeypatch, no_train_run, fake_trainer
):
    """Admission arbitrates between OUR jobs. With nothing else running, a job
    always starts — whether it fits at all is the OOM guard's question, and
    the OS desktop permanently holds VRAM that would otherwise make a
    tight-but-fine job wait forever. (Seen live: a 3 GB training refused on an
    idle 4 GB card because Windows held 600 MB.)"""
    from app.services import gpu_admission

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: 0.2)
    pid, _ = make_trainable_project(client, "AdmitLone")
    job_id = _queued_training_job(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        job = db.get(TrainingJob, job_id)
        admitted, reason = gpu_admission.check(db, "training", job.id, job.created_at, 3.0)
        assert admitted is True and reason is None
    finally:
        db.close()


def test_check_refuses_with_live_numbers(client, monkeypatch, no_train_run, fake_trainer):
    """With another job RUNNING and not enough VRAM left, the refusal carries
    the ACTUAL free/needed numbers — the UI shows it verbatim, so it must
    never be a hardcoded guess."""
    from app.services import gpu_admission

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: 1.2)
    pid, _ = make_trainable_project(client, "AdmitB")
    runner_id = _queued_training_job(client, pid)
    _mark_running(client, runner_id)
    job_id = _queued_training_job(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        job = db.get(TrainingJob, job_id)
        admitted, reason = gpu_admission.check(db, "training", job.id, job.created_at, 3.0)
        assert admitted is False
        assert "1.2 GB free" in reason and "~3.0 GB" in reason
        assert "training v" in reason, "the refusal names what holds the GPU"
    finally:
        db.close()


def test_fifo_younger_waiter_yields(client, monkeypatch, no_train_run, fake_trainer):
    """With two jobs waiting, only the older may take freed resources."""
    from app.services import gpu_admission

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: 8.0)
    pid, _ = make_trainable_project(client, "AdmitC")
    older = _queued_training_job(client, pid)
    younger = _queued_training_job(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        o = db.get(TrainingJob, older)
        y = db.get(TrainingJob, younger)
        admitted_old, _ = gpu_admission.check(db, "training", o.id, o.created_at, 3.0)
        admitted_young, reason = gpu_admission.check(db, "training", y.id, y.created_at, 3.0)
        assert admitted_old is True
        assert admitted_young is False and "ahead in line" in reason
    finally:
        db.close()


def test_cpu_mode_is_one_job_at_a_time(client, monkeypatch, no_train_run, fake_trainer):
    """No GPU to meter -> admitted only when nothing is RUNNING, and the
    refusal names what is."""
    from app.services import gpu_admission

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: None)
    pid, _ = make_trainable_project(client, "AdmitD")
    waiting = _queued_training_job(client, pid)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import JobStatus, TrainingJob

        w = db.get(TrainingJob, waiting)
        admitted, reason = gpu_admission.check(db, "training", w.id, w.created_at, 3.0)
        assert admitted is True, "idle machine admits"

        # Something starts RUNNING -> the same check now refuses, naming it.
        runner_id = _queued_training_job(client, pid)
        r = db.get(TrainingJob, runner_id)
        r.status = JobStatus.RUNNING
        db.commit()

        admitted, reason = gpu_admission.check(db, "training", w.id, w.created_at, 3.0)
        assert admitted is False
        assert "training v" in reason
    finally:
        db.close()


def test_wait_for_gpu_writes_live_reason_then_clears_it(
    client, monkeypatch, no_train_run, fake_trainer
):
    """The wait loop: refusal -> status_detail carries the reason; admission ->
    cleared and True returned. Sleep is stubbed so the test doesn't."""
    from app.services import gpu_admission

    pid, _ = make_trainable_project(client, "AdmitE")
    runner_id = _queued_training_job(client, pid)
    _mark_running(client, runner_id)  # someone holds the card
    job_id = _queued_training_job(client, pid)

    # First poll refuses, second admits (the running job shrank / finished
    # freeing memory).
    frees = iter([1.0, 8.0])
    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: next(frees))

    observed: list[str | None] = []

    def spy_sleep(_s):
        # Runs between the two polls — exactly when the reason must be visible.
        db2 = client.SessionLocal()  # type: ignore[attr-defined]
        try:
            from app.models import TrainingJob

            observed.append(db2.get(TrainingJob, job_id).status_detail)
        finally:
            db2.close()

    monkeypatch.setattr(gpu_admission.time, "sleep", spy_sleep)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        job = db.get(TrainingJob, job_id)
        assert gpu_admission.wait_for_gpu(db, job, "training", 3.0) is True
        assert observed and "1.0 GB free" in (observed[0] or "")
        db.expire(job)
        assert job.status_detail is None, "admission must clear the waiting reason"
    finally:
        db.close()


def test_cancel_while_waiting_releases_the_job(client, monkeypatch, no_train_run, fake_trainer):
    from app.services import gpu_admission

    pid, _ = make_trainable_project(client, "AdmitF")
    runner_id = _queued_training_job(client, pid)
    _mark_running(client, runner_id)  # holds the card for the whole test
    job_id = _queued_training_job(client, pid)

    monkeypatch.setattr(gpu_admission, "free_vram_gb", lambda: 0.1)  # never admits

    def cancel_then_continue(_s):
        db2 = client.SessionLocal()  # type: ignore[attr-defined]
        try:
            from app.models import TrainingJob

            db2.get(TrainingJob, job_id).control = "cancel"
            db2.commit()
        finally:
            db2.close()

    monkeypatch.setattr(gpu_admission.time, "sleep", cancel_then_continue)

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        from app.models import TrainingJob

        job = db.get(TrainingJob, job_id)
        assert gpu_admission.wait_for_gpu(db, job, "training", 3.0) is False
    finally:
        db.close()


def test_second_job_queues_instead_of_409(client, monkeypatch, fake_trainer):
    """The route contract that changed: overlapping starts are ACCEPTED now.

    With the runner stubbed, the first job stays QUEUED — and a second POST for
    the same project (formerly a 409) also lands as a QUEUED row. Cross-project
    likewise. The status_detail field rides the poll response so the UI can say
    why a job is holding.
    """
    monkeypatch.setattr("app.api.routes.train.run_training_job", lambda job_id: None)
    pid, _ = make_trainable_project(client, "QueueA")
    first = _queued_training_job(client, pid)
    second = _queued_training_job(client, pid)
    assert first != second

    other, _ = make_trainable_project(client, "QueueB")
    third = _queued_training_job(client, other)

    for jid in (first, second, third):
        job = client.get(f"/api/training-jobs/{jid}").json()
        assert job["status"] == "queued"
        assert "status_detail" in job


def test_startup_fails_orphaned_jobs_and_clears_wait_note(
    client, monkeypatch, no_train_run, fake_trainer
):
    """Rows left queued/running by a dead server process are failed at startup
    (database._fail_interrupted_jobs) — load-bearing now, because admission
    reads RUNNING rows globally and one orphan would make every future job
    wait for a GPU nothing is using. A stale waiting note is cleared too."""
    import app.database as database

    monkeypatch.setattr(
        database, "SessionLocal", client.SessionLocal  # type: ignore[attr-defined]
    )

    pid, _ = make_trainable_project(client, "Orphans")
    queued = _queued_training_job(client, pid)
    running = _queued_training_job(client, pid)
    _mark_running(client, running)

    # The queued one was mid-wait when the "server died".
    from app.models import TrainingJob

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        db.get(TrainingJob, queued).status_detail = "Waiting for GPU: …"
        db.commit()
    finally:
        db.close()

    database._fail_interrupted_jobs()

    for jid in (queued, running):
        job = client.get(f"/api/training-jobs/{jid}").json()
        assert job["status"] == "failed"
        assert "server stopped" in job["error"]
        assert job["status_detail"] is None
