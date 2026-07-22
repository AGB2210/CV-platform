"""
Evaluation (Phase 5, step 3): the COCO metric, the endpoint guards, and a full
run against a test split with a fake predictor.

The metric itself is exercised against real pycocotools — a wrong number there is
silent and misleading — while the model is faked, so no GPU or weights are needed.
"""

from __future__ import annotations

import pytest

from app.ml.annotators.base import Box
from app.ml.predictors.base import Predictor
from app.ml.trainers.base import Trainer
from tests.conftest import make_project, upload_images


# --- the metric ------------------------------------------------------------


def test_coco_metric_perfect_and_empty():
    """Perfect detections score ~1.0; no detections score 0.0. If these drift,
    every test mAP the app reports is wrong."""
    from app.services.evaluation_job import _coco_evaluate

    cat_id_of = {"car": 1, "person": 2}
    gt_images = [{"id": 1, "width": 100, "height": 100}]
    gt_anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
    ]
    perfect = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "score": 0.99},
    ]
    m = _coco_evaluate(gt_images, gt_anns, perfect, ["car", "person"], cat_id_of)
    assert m["map_50_95"] > 0.99
    assert {c["name"] for c in m["per_class"]} == {"car", "person"}
    assert all(c["ap"] > 0.99 for c in m["per_class"])

    empty = _coco_evaluate(gt_images, gt_anns, [], ["car", "person"], cat_id_of)
    assert empty["map_50_95"] == 0.0


def test_coco_metric_reports_per_class_separately():
    """A model perfect on one class and blind to the other shows exactly that —
    the aggregate would hide it."""
    from app.services.evaluation_job import _coco_evaluate

    cat_id_of = {"car": 1, "person": 2}
    gt_images = [{"id": 1, "width": 100, "height": 100}]
    gt_anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
        {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
    ]
    cars_only = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99}]
    m = _coco_evaluate(gt_images, gt_anns, cars_only, ["car", "person"], cat_id_of)
    ap = {c["name"]: c["ap"] for c in m["per_class"]}
    assert ap["car"] > 0.99
    assert ap["person"] == 0.0


# --- endpoints -------------------------------------------------------------


class _FakePredictor(Predictor):
    key = "fakeeval"

    def __init__(self) -> None:
        super().__init__()

    def _load_impl(self) -> None:
        pass

    def _unload_impl(self) -> None:
        pass

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        # Detects a car exactly where the ground truth is. Box is xyxy, so
        # (10,10)->(40,40) is the 30x30 COCO box the fixture annotates — a clean
        # match (IoU 1.0), giving a high, non-degenerate mAP.
        return [Box(10, 10, 40, 40, "car", 0.95)]


class _FakeEvalTrainer(Trainer):
    key = "fakeeval"
    export_format = "yolo"

    def train(self, config, on_epoch):  # pragma: no cover
        raise NotImplementedError

    def load_predictor(self, checkpoint_path, class_names):
        return _FakePredictor()


@pytest.fixture()
def eval_setup(client, monkeypatch):
    """A project with a saved version that HAS a test split, a finished model with
    a checkpoint, and evaluation wired to the test's own DB session."""
    from app.config import settings, to_storage_path
    from app.ml.predictors import registry as predictor_registry
    from app.ml.trainers import registry as trainer_registry
    from app.models import JobStatus, TrainingJob
    from app.services import evaluation_job

    trainer_registry.register(_FakeEvalTrainer)
    # The runner opens its OWN session; point it at the test database.
    monkeypatch.setattr(evaluation_job, "SessionLocal", client.SessionLocal)  # type: ignore[attr-defined]

    pid = make_project(client, classes=("car", "person"))
    images = upload_images(client, pid, [f"img{i}.png" for i in range(4)])
    car = client.get(f"/api/projects/{pid}/classes").json()[0]

    # Give every image a car box, and put two images in the test split.
    for img in images:
        client.post(
            f"/api/images/{img['id']}/annotations",
            json={"category_id": car["id"], "x": 10, "y": 10, "width": 30, "height": 30},
        )
    for img in images[:2]:
        client.patch(f"/api/images/{img['id']}/split", params={"split": "test"})

    version = client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None}).json()

    # A finished model with a checkpoint file on disk.
    ckpt = settings.runs_dir / "888" / "best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"weights")
    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        job = TrainingJob(
            project_id=pid,
            trainer_key="fakeeval",
            version=1,
            status=JobStatus.DONE,
            epochs=1,
            total_epochs=1,
            checkpoint_path=to_storage_path(ckpt),
            dataset_version_id=version["id"],
        )
        db.add(job)
        db.commit()
        model_id = job.id
    finally:
        db.close()

    yield {"project_id": pid, "model_id": model_id, "version_id": version["id"]}

    trainer_registry._REGISTRY.pop("fakeeval", None)
    predictor_registry.release()


def test_evaluate_runs_and_reports_test_map(client, eval_setup):
    """The happy path: start an evaluation, and it produces a test mAP.

    TestClient runs BackgroundTasks after the response, so by the time we poll the
    job it has run against the test split with the fake predictor.
    """
    r = client.post(
        f"/api/projects/{eval_setup['project_id']}/evaluate",
        json={
            "training_job_id": eval_setup["model_id"],
            "dataset_version_id": eval_setup["version_id"],
            "split": "test",
        },
    )
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]

    job = client.get(f"/api/evaluation-jobs/{job_id}").json()
    assert job["status"] == "done", job.get("error")
    assert job["num_images"] == 2  # two images were put in the test split
    assert job["map_50_95"] is not None
    # The fake predictor matches the ground-truth car exactly, so mAP is high.
    assert job["map_50_95"] > 0.9
    names = {c["name"] for c in job["per_class"]}
    assert names == {"car", "person"}


def test_evaluate_404_for_unknown_model(client, eval_setup):
    r = client.post(
        f"/api/projects/{eval_setup['project_id']}/evaluate",
        json={"training_job_id": 999999, "dataset_version_id": eval_setup["version_id"]},
    )
    assert r.status_code == 404


def test_evaluate_400_when_split_empty(client, eval_setup):
    """No val images were assigned, so evaluating the val split is refused up front."""
    r = client.post(
        f"/api/projects/{eval_setup['project_id']}/evaluate",
        json={
            "training_job_id": eval_setup["model_id"],
            "dataset_version_id": eval_setup["version_id"],
            "split": "val",
        },
    )
    assert r.status_code == 400
    assert "no val images" in r.json()["detail"].lower()


# --- confusion matrix, PR curves, worst images -------------------------------


def test_confusion_and_worst_matching_logic():
    """The pure matching maths, all four outcomes: correct match, class
    confusion, invention (FP vs background), and miss (FN)."""
    from app.services.evaluation_job import _confusion_and_worst

    classes = ["car", "person"]
    per_image = [
        {
            # Image A: one car predicted correctly; one person MISSED.
            "image_id": 10, "filename": "a.png", "original_filename": "a.png",
            "gt": [(0, [10, 10, 20, 20]), (1, [50, 50, 10, 10])],
            "preds": [(0, [11, 11, 20, 20], 0.9)],
        },
        {
            # Image B: the car called a PERSON (confusion), plus a pure
            # invention in empty space; below-threshold pred must be ignored.
            "image_id": 11, "filename": "b.png", "original_filename": "b.png",
            "gt": [(0, [10, 10, 20, 20])],
            "preds": [
                (1, [10, 10, 20, 20], 0.8),
                (0, [80, 80, 10, 10], 0.7),
                (0, [10, 10, 20, 20], 0.05),  # under 0.25 — invisible
            ],
        },
    ]

    confusion, worst = _confusion_and_worst(per_image, classes)
    m = confusion["matrix"]  # [predicted][actual], bg index = 2
    assert confusion["classes"] == ["car", "person", "background"]
    assert m[0][0] == 1, "correct car match"
    assert m[1][0] == 1, "car called person"
    assert m[0][2] == 1, "invented car (background)"
    assert m[2][1] == 1, "missed person"

    # Worst ranking: image B has 2 errors (confusion + invention), A has 1 miss.
    assert [w["image_id"] for w in worst] == [11, 10]
    assert worst[0]["fp"] == 2 and worst[0]["fn"] == 0
    assert worst[1]["fp"] == 0 and worst[1]["fn"] == 1


def test_evaluation_stores_details(client, eval_setup):
    """A finished evaluation carries PR curves, the confusion matrix and the
    worst-image list through the API."""
    r = client.post(
        f"/api/projects/{eval_setup['project_id']}/evaluate",
        json={
            "training_job_id": eval_setup["model_id"],
            "dataset_version_id": eval_setup["version_id"],
            "split": "test",
        },
    )
    assert r.status_code == 202, r.text
    job = client.get(f"/api/evaluation-jobs/{r.json()['id']}").json()
    assert job["status"] == "done", job.get("error")

    d = job["details"]
    assert d is not None
    assert {c["name"] for c in d["pr_curves"]} <= {"car", "person"}
    curve = next(c for c in d["pr_curves"] if c["name"] == "car")
    assert len(curve["recall"]) == len(curve["precision"]) > 0
    assert all(0 <= p <= 1 for p in curve["precision"])

    conf = d["confusion"]
    assert conf["classes"][-1] == "background"
    n = len(conf["classes"])
    assert len(conf["matrix"]) == n and all(len(row) == n for row in conf["matrix"])
    # The fake predictor nails the car on both test images — diagonal hits.
    assert conf["matrix"][0][0] == 2

    assert isinstance(d["worst"], list)
