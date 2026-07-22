"""
Inference playground endpoints: list deployable models, and predict on an upload.

No real model — a fake trainer returns a fake predictor with fixed boxes, and the
"checkpoint" is a real file on disk so the on-disk validation runs. What is
tested is the endpoint contract and the two rules: predictions come back but are
never written to the database, and only a finished run with a present checkpoint
is deployable.
"""

from __future__ import annotations

import pytest

from app.ml.annotators.base import Box
from app.ml.predictors.base import Predictor
from app.ml.trainers.base import Trainer
from tests.conftest import make_project, png_bytes


class _FakePredictor(Predictor):
    key = "fakeinfer"

    def _load_impl(self) -> None:
        pass

    def _unload_impl(self) -> None:
        pass

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        return [
            b
            for b in (
                Box(10, 20, 40, 60, "car", 0.9),
                Box(5, 5, 8, 8, "person", 0.1),  # dropped at the default threshold
            )
            if b.confidence >= conf_threshold
        ]


class _FakeInferTrainer(Trainer):
    key = "fakeinfer"
    export_format = "yolo"

    def train(self, config, on_epoch):  # pragma: no cover
        raise NotImplementedError

    def load_predictor(self, checkpoint_path, class_names):
        return _FakePredictor()


@pytest.fixture()
def infer_setup(client, tmp_path):
    """A project, a finished run whose checkpoint file exists, and the fake trainer."""
    from app.config import settings, to_storage_path
    from app.ml.predictors import registry as predictor_registry
    from app.ml.trainers import registry as trainer_registry
    from app.models import JobStatus, TrainingJob

    trainer_registry.register(_FakeInferTrainer)

    pid = make_project(client, classes=("car", "person"))

    # A real checkpoint file under storage, so the on-disk check passes.
    ckpt = settings.runs_dir / "999" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not really weights")

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        job = TrainingJob(
            project_id=pid,
            trainer_key="fakeinfer",
            version=1,
            status=JobStatus.DONE,
            epochs=1,
            total_epochs=1,
            checkpoint_path=to_storage_path(ckpt),
        )
        db.add(job)
        db.commit()
        job_id = job.id
    finally:
        db.close()

    yield {"project_id": pid, "job_id": job_id}

    trainer_registry._REGISTRY.pop("fakeinfer", None)
    predictor_registry.release()


def test_list_models_shows_finished_run(client, infer_setup):
    models = client.get(f"/api/projects/{infer_setup['project_id']}/models").json()
    assert len(models) == 1
    assert models[0]["job_id"] == infer_setup["job_id"]
    assert models[0]["trainer_key"] == "fakeinfer"


def test_list_models_hides_run_with_missing_checkpoint(client, infer_setup):
    """A finished run whose weights were cleaned up is not deployable."""
    from app.config import settings

    (settings.runs_dir / "999" / "best.pt").unlink()
    models = client.get(f"/api/projects/{infer_setup['project_id']}/models").json()
    assert models == []


def test_predict_returns_boxes(client, infer_setup):
    r = client.post(
        f"/api/models/{infer_setup['job_id']}/predict",
        files={"file": ("x.png", png_bytes(64, 48), "image/png")},
        data={"conf_threshold": "0.25"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_width"] == 64 and body["image_height"] == 48
    # The 0.1-confidence box is below threshold; only the car survives.
    assert len(body["boxes"]) == 1
    assert body["boxes"][0]["label"] == "car"
    # COCO-style x,y,width,height, not xyxy.
    assert body["boxes"][0]["width"] == 30 and body["boxes"][0]["height"] == 40


def test_predict_never_writes_to_the_database(client, infer_setup):
    """The playground is read-only. A prediction must create no annotation rows."""
    from app.models import Annotation

    client.post(
        f"/api/models/{infer_setup['job_id']}/predict",
        files={"file": ("x.png", png_bytes(), "image/png")},
    )
    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        assert db.query(Annotation).count() == 0
    finally:
        db.close()


def test_predict_404_for_unknown_model(client, infer_setup):
    r = client.post(
        "/api/models/999999/predict",
        files={"file": ("x.png", png_bytes(), "image/png")},
    )
    assert r.status_code == 404


def test_predict_400_when_run_unfinished(client, infer_setup):
    """A run that never finished has no usable checkpoint to deploy."""
    from app.models import JobStatus, TrainingJob

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        job = db.get(TrainingJob, infer_setup["job_id"])
        job.status = JobStatus.RUNNING
        db.commit()
    finally:
        db.close()

    r = client.post(
        f"/api/models/{infer_setup['job_id']}/predict",
        files={"file": ("x.png", png_bytes(), "image/png")},
    )
    # RUNNING is both "not done" and "gpu busy"; either way it must refuse.
    assert r.status_code in (400, 409)


def test_download_weights_streams_checkpoint(client, infer_setup):
    """The .pt download is the checkpoint file, byte for byte, under a
    label-carrying name."""
    r = client.get(f"/api/models/{infer_setup['job_id']}/weights")
    assert r.status_code == 200
    assert r.content == b"not really weights"
    dispo = r.headers["content-disposition"]
    assert dispo.endswith('.pt"') and "fakeinfer" in dispo


def test_download_weights_404_and_400(client, infer_setup):
    """No job = 404; a job without a usable checkpoint = 400."""
    assert client.get("/api/models/999999/weights").status_code == 404

    from app.config import from_storage_path
    from app.models import TrainingJob

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        job = db.get(TrainingJob, infer_setup["job_id"])
        from_storage_path(job.checkpoint_path).unlink()  # weights cleaned up
    finally:
        db.close()
    assert client.get(f"/api/models/{infer_setup['job_id']}/weights").status_code == 400
