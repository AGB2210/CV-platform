"""
The predictor seam (Phase 5, step 1): lifecycle, the one-resident registry, and
the VRAM cross-eviction that keeps a predictor and an annotator off the card at
the same time.

No GPU and no real weights — a fake trainer produces a fake predictor, exactly as
the training tests use fake trainers. What is tested is the SEAM (load/unload,
caching, eviction), not any model's output.
"""

from __future__ import annotations

import pytest

from app.ml.annotators.base import Box
from app.ml.predictors.base import Predictor
from app.ml.predictors import registry as predictor_registry
from app.ml.trainers.base import Trainer


class FakePredictor(Predictor):
    """Counts its own load/unload/predict so the registry's behaviour is visible."""

    def __init__(self, key: str, boxes: list[Box] | None = None) -> None:
        super().__init__()
        self.key = key
        self._boxes = boxes or []
        self.load_calls = 0
        self.unload_calls = 0

    def _load_impl(self) -> None:
        self.load_calls += 1

    def _unload_impl(self) -> None:
        self.unload_calls += 1

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> list[Box]:
        return [b for b in self._boxes if b.confidence >= conf_threshold]


class FakeTrainer(Trainer):
    key = "fake"
    export_format = "yolo"

    def train(self, config, on_epoch):  # pragma: no cover - not used here
        raise NotImplementedError

    def load_predictor(self, checkpoint_path, class_names):
        # A distinct predictor per checkpoint, so eviction-vs-cache is observable.
        return FakePredictor(key=f"fake:{checkpoint_path}")


@pytest.fixture()
def fake_trainer():
    from app.ml.trainers import registry as trainer_registry

    trainer_registry.register(FakeTrainer)
    yield
    trainer_registry._REGISTRY.pop("fake", None)
    predictor_registry.release()


def test_lifecycle_is_idempotent():
    p = FakePredictor("x")
    p.load()
    p.load()  # second load is a no-op
    assert p.is_loaded and p.load_calls == 1
    p.unload()
    p.unload()  # second unload is a no-op
    assert not p.is_loaded and p.unload_calls == 1


def test_context_manager_unloads_even_on_error():
    p = FakePredictor("x")
    with pytest.raises(ValueError):
        with p:
            assert p.is_loaded
            raise ValueError("boom")
    assert not p.is_loaded and p.unload_calls == 1


def test_predict_respects_confidence_threshold():
    boxes = [
        Box(0, 0, 10, 10, "car", 0.9),
        Box(0, 0, 10, 10, "car", 0.2),
    ]
    p = FakePredictor("x", boxes)
    assert len(p.predict("img", conf_threshold=0.5)) == 1
    assert len(p.predict("img", conf_threshold=0.1)) == 2


def test_registry_caches_same_checkpoint(fake_trainer):
    """Acquiring the same (trainer, checkpoint) twice reuses the resident model."""
    a = predictor_registry.acquire("fake", "best.pt", ["car"])
    b = predictor_registry.acquire("fake", "best.pt", ["car"])
    assert a is b  # cache hit — not reloaded
    assert a.load_calls == 1


def test_registry_evicts_on_different_checkpoint(fake_trainer):
    """A different checkpoint unloads the old resident before loading the new."""
    a = predictor_registry.acquire("fake", "v1.pt", ["car"])
    b = predictor_registry.acquire("fake", "v2.pt", ["car"])
    assert a is not b
    assert a.unload_calls == 1  # the old one was evicted
    assert b.is_loaded
    assert predictor_registry.resident_id() == ("fake", "v2.pt")


def test_acquiring_a_predictor_evicts_the_resident_annotator(fake_trainer, monkeypatch):
    """The card holds ONE model: loading a predictor drops any annotator.

    This is the VRAM rule that keeps a 4 GB card from OOMing — a predictor and a
    detection model cannot both be resident.
    """
    from app.ml import registry as annotator_registry

    released = {"count": 0}
    monkeypatch.setattr(
        annotator_registry, "release", lambda: released.__setitem__("count", released["count"] + 1)
    )

    predictor_registry.acquire("fake", "best.pt", ["car"])
    assert released["count"] == 1


def test_release_is_safe_when_nothing_resident():
    predictor_registry.release()  # must not raise
    assert predictor_registry.resident_id() is None
