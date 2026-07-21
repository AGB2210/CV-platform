"""
YOLO trainer (Ultralytics) — the first concrete training backend.

Chosen to go first because it is the most forgiving path to a real run on a
modest GPU: the nano model fits comfortably, the install is robust on Windows,
and it consumes the YOLO export we already build and round-trip test. RF-DETR / RT-DETR
(COCO-based) follow behind the same interface.

HOW IT ADAPTS THE FRAMEWORK
---------------------------
Ultralytics owns the whole training loop — we don't step inside it. Two seams let
us report progress without doing so:

  - a per-epoch CALLBACK ("on_fit_epoch_end") it invokes with its own trainer
    object, from which we pull the epoch's loss and validation mAP and forward an
    EpochMetrics to the runner's callback;
  - the return value of model.train(), plus the best.pt it writes under our
    output dir, for the final result.

Everything heavy (ultralytics, torch) imports lazily INSIDE train(), never at
module top level — importing this module must stay free so /api/trainers can be
listed, and so a machine without the deps still renders the page.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ml.trainers.base import EpochCallback, EpochMetrics, TrainConfig, TrainResult, Trainer
from app.ml.trainers.registry import register

logger = logging.getLogger(__name__)


@register
class YoloTrainer(Trainer):
    key = "yolo"
    # Name the ACTUAL variant, not the family. "YOLO11" could be any of n/s/m/l/x
    # — sizes that differ by an order of magnitude in memory and speed — so a
    # user picking from a list deserves to know which one they're getting.
    display_name = "YOLO11n (nano) · Ultralytics"
    description = (
        "Fine-tunes the pretrained YOLO11 nano checkpoint (yolo11n.pt) — the "
        "smallest of the YOLO11 family, so it trains on modest GPUs. Consumes "
        "the YOLO export format."
    )
    # Rough peak VRAM for nano at 640px / batch 8; batch and image size move it.
    # Surfaced as a NUMBER rather than baked into prose, so the UI can compare it
    # against whatever GPU the app is actually running on — this ships to other
    # machines, and hardcoding one developer's card into user-facing text would
    # be wrong everywhere else.
    approx_vram_gb = 3.0
    export_format = "yolo"

    default_epochs = 50
    default_batch_size = 8
    default_image_size = 640

    #: Pretrained checkpoint to fine-tune from. Nano is the safe default: the
    #: larger variants OOM at any useful batch size on a small card. Downloaded
    #: once by ultralytics from its own release assets and then cached.
    base_weights = "yolo11n.pt"

    def train(self, config: TrainConfig, on_epoch: EpochCallback) -> TrainResult:
        # Lazy, and inside a function: importing ultralytics pulls torch and
        # costs seconds + hundreds of MB, which must not happen just to list
        # trainers. A clear error here (rather than at import) is also what lets
        # the page report "backend not installed" instead of crashing.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` in "
                "the backend venv and restart the server."
            ) from exc

        # Turn off ultralytics' anonymous analytics/telemetry — this is a local,
        # no-cloud tool, and a training run shouldn't phone home. Best-effort:
        # the settings API has shifted across versions, so never let it fail the
        # run.
        try:
            from ultralytics import settings

            settings.update({"sync": False})
        except Exception:  # noqa: BLE001
            pass

        data_yaml = config.dataset_dir / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Expected exported dataset at {data_yaml}")

        # Finetune from a prior run's checkpoint when asked, else the pretrained
        # base. Loading a .pt trained on the same classes continues improving it;
        # YOLO reads the architecture + weights (including the trained detection
        # head) straight from the file.
        start_weights = str(config.init_weights) if config.init_weights else self.base_weights
        model = YOLO(start_weights)

        # Ultralytics fires on_fit_epoch_end ONE EXTRA TIME after the training
        # loop ends — its final validation pass — and by then `trainer.epoch`
        # has already advanced. Recorded naively that produces a phantom epoch
        # N+1 carrying the SAME train loss as the real last epoch: a duplicated
        # point on the curve and an axis running to 4 on a 3-epoch run. Both
        # were visible in real runs (2 planned epochs -> 3 points).
        #
        # So the adapter tracks what it has already reported and refuses to
        # report an epoch twice, one beyond the requested schedule, or anything
        # at all once a stop has been signalled (that final validation is
        # exactly when the phantom arrives).
        reported: set[int] = set()
        finished = False

        # Bridge ultralytics' per-epoch signal to our callback. The framework
        # hands us ITS trainer object; we defensively dig the numbers out of it,
        # because the exact metric-dict keys and loss attributes have moved
        # between versions and a KeyError here would abort an otherwise-fine run.
        def _on_fit_epoch_end(yolo_trainer) -> None:  # noqa: ANN001
            nonlocal finished
            if finished:
                return
            epoch = int(getattr(yolo_trainer, "epoch", 0)) + 1  # ultralytics is 0-based
            if epoch in reported or epoch > config.epochs:
                return
            reported.add(epoch)
            metrics = getattr(yolo_trainer, "metrics", None) or {}

            def metric(*keys: str) -> float | None:
                for k in keys:
                    v = metrics.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            return None
                return None

            # mAP averaged over IoU .50:.95, and the looser .50 figure. The "(B)"
            # suffix is ultralytics' notation for box (vs mask) metrics.
            val_map = metric("metrics/mAP50-95(B)", "metrics/mAP50-95")
            val_map50 = metric("metrics/mAP50(B)", "metrics/mAP50")

            train_loss: float | None = None
            tloss = getattr(yolo_trainer, "tloss", None)
            if tloss is not None:
                try:
                    train_loss = float(tloss.sum())
                except Exception:  # noqa: BLE001 — tloss may be a scalar tensor
                    try:
                        train_loss = float(tloss)
                    except (TypeError, ValueError):
                        train_loss = None

            try:
                should_stop = on_epoch(
                    EpochMetrics(
                        epoch=epoch,
                        total_epochs=int(yolo_trainer.epochs),
                        train_loss=train_loss,
                        val_map=val_map,
                        val_map50=val_map50,
                    )
                )
            except Exception:  # noqa: BLE001
                # A DB hiccup writing progress must never kill the training run.
                logger.exception("Failed to record epoch progress")
                return

            if should_stop:
                # Ultralytics' own early-stopping flag: it's checked at the top
                # of each epoch, so setting it here lets the epoch in flight
                # finish and checkpoint before the loop exits. Requesting a stop
                # therefore keeps a usable best.pt rather than truncating mid-
                # epoch — a half-written epoch is worse than one more epoch of
                # waiting.
                #
                # `finished` also closes the door on the post-loop validation
                # callback, which would otherwise land as a phantom epoch.
                logger.info("Stop requested — finishing after epoch %s", epoch)
                finished = True
                yolo_trainer.stop = True

        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        # device: ultralytics wants 0 for the first CUDA GPU, "cpu" otherwise.
        device = 0 if config.device == "cuda" else "cpu"

        kwargs: dict = dict(
            data=str(data_yaml),
            epochs=config.epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            device=device,
            project=str(config.output_dir),
            name="run",
            exist_ok=True,
            # Windows + a background thread: the multiprocessing dataloader can
            # deadlock. Single-process loading is slower but reliable here.
            workers=0,
            # Keep the run self-contained and quiet; we render progress ourselves.
            verbose=False,
            plots=False,
        )
        if config.learning_rate is not None:
            kwargs["lr0"] = config.learning_rate

        results = model.train(**kwargs)

        # best.pt is written under <output>/run/weights/. Prefer the path the
        # trainer reports; fall back to the conventional location.
        best = getattr(getattr(model, "trainer", None), "best", None)
        best_path = Path(best) if best else config.output_dir / "run" / "weights" / "best.pt"

        best_map: float | None = None
        try:
            best_map = float(results.box.map)  # mAP50-95 of the best model on val
        except Exception:  # noqa: BLE001
            pass

        epochs_completed = config.epochs
        done_epoch = getattr(getattr(model, "trainer", None), "epoch", None)
        if done_epoch is not None:
            epochs_completed = int(done_epoch) + 1

        return TrainResult(
            best_checkpoint_path=best_path if best_path.exists() else None,
            best_map=best_map,
            epochs_completed=epochs_completed,
        )
