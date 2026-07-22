"""
Ultralytics trainers — the YOLO11 family and RT-DETR, one adapter.

One shared train() implementation because ultralytics gives every model the
same API (train(data=...), callbacks, best.pt); the variants differ only in
which pretrained checkpoint they start from and how much VRAM they want. Each
is registered as its own key so a job's provenance names the actual
architecture it trained — "yolo11l", not "some YOLO".

THE VARIANTS ARE A LADDER, NOT A MENU OF EQUALS
-----------------------------------------------
nano..xlarge trade accuracy for memory/speed in order; RT-DETR is a different
architecture (a DETR — transformer detection head, NMS-free) that competes
with the upper YOLOs. The UI groups by family so "which architecture" and
"which size" read as the two separate questions they are.

HOW IT ADAPTS THE FRAMEWORK
---------------------------
Ultralytics owns the whole training loop — we don't step inside. Two seams let
us report progress without doing so:

  - a per-epoch CALLBACK ("on_fit_epoch_end") it invokes with its own trainer
    object, from which we pull the epoch's loss and validation mAP and forward
    an EpochMetrics to the runner's callback;
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


class UltralyticsTrainer(Trainer):
    """Shared adapter over ultralytics' Model API. Subclasses only declare
    metadata: which checkpoint to start from and what it costs."""

    export_format = "yolo"
    default_epochs = 50
    default_image_size = 640

    #: Pretrained checkpoint to fine-tune from. Downloaded once by ultralytics
    #: from its own release assets and then cached.
    base_weights = ""
    #: Which ultralytics class loads this architecture: "YOLO" or "RTDETR".
    #: They share the training API but not the inference pipeline — RT-DETR
    #: has its own predictor (no NMS), and loading its checkpoint through the
    #: YOLO class can route postprocessing wrongly. The trainer knows what it
    #: trained, so it says.
    model_class = "YOLO"

    def _load_model(self, weights: str):
        try:
            import ultralytics
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Run `pip install ultralytics` in "
                "the backend venv and restart the server."
            ) from exc
        cls = getattr(ultralytics, self.model_class)
        return cls(weights)

    def load_predictor(self, checkpoint_path, class_names: list[str]):
        """A runnable predictor over one of this trainer's `best.pt` checkpoints."""
        from app.ml.predictors.yolo import YoloPredictor

        return YoloPredictor(checkpoint_path, class_names, model_class=self.model_class)

    def train(self, config: TrainConfig, on_epoch: EpochCallback) -> TrainResult:
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
        # ultralytics reads the architecture + weights (including the trained
        # detection head) straight from the file.
        start_weights = str(config.init_weights) if config.init_weights else self.base_weights
        model = self._load_model(start_weights)

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
            # Verbose so ultralytics narrates through its logger — the runner
            # captures that narration into the job's live log for the UI
            # (services/training_logs.py). The console cost is a few lines in
            # the server log.
            verbose=True,
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


# --- The roster --------------------------------------------------------------
# approx_vram_gb is rough peak at 640px and that variant's default batch —
# batch and image size move it. The numbers are deliberately conservative: the
# UI compares them against the detected GPU to size the default batch, and an
# optimistic figure turns into an OOM mid-run where a cautious one merely
# trains slightly slower.
#
# YOLO11 vs YOLO12: the user asked for whichever is better, not both. YOLO12
# wins on accuracy at every size (ultralytics' own COCO tables: 40.6 vs 39.5
# mAP at nano, and the gap holds up the ladder) at comparable inference cost,
# so YOLO12 is the listed family and YOLO11 is legacy — registered but hidden,
# because checkpoints already trained with these keys must keep deploying.


class _LegacyYolo11(UltralyticsTrainer):
    """Base for the delisted YOLO11 family — see the roster note above."""

    family = "YOLO11"
    listed = False
    description = "Superseded by YOLO12. Existing checkpoints still deploy."


@register
class Yolo11NanoTrainer(_LegacyYolo11):
    # HISTORIC KEY: this was the first (only) trainer, registered as "yolo".
    # Existing job rows and checkpoints reference it, so the key stays even
    # though its siblings follow the "yolo11s" pattern.
    key = "yolo"
    variant = "nano"
    display_name = "YOLO11 nano"
    approx_vram_gb = 3.0
    default_batch_size = 8
    base_weights = "yolo11n.pt"


@register
class Yolo11SmallTrainer(_LegacyYolo11):
    key = "yolo11s"
    variant = "small"
    display_name = "YOLO11 small"
    approx_vram_gb = 4.5
    default_batch_size = 8
    base_weights = "yolo11s.pt"


@register
class Yolo11MediumTrainer(_LegacyYolo11):
    key = "yolo11m"
    variant = "medium"
    display_name = "YOLO11 medium"
    approx_vram_gb = 6.5
    default_batch_size = 4
    base_weights = "yolo11m.pt"


@register
class Yolo11LargeTrainer(_LegacyYolo11):
    key = "yolo11l"
    variant = "large"
    display_name = "YOLO11 large"
    approx_vram_gb = 8.5
    default_batch_size = 4
    base_weights = "yolo11l.pt"


@register
class Yolo11XLargeTrainer(_LegacyYolo11):
    key = "yolo11x"
    variant = "xlarge"
    display_name = "YOLO11 xlarge"
    approx_vram_gb = 11.0
    default_batch_size = 2
    base_weights = "yolo11x.pt"


# --- YOLO12: attention-based, better mAP than YOLO11 at every size ----------
# Slightly hungrier than YOLO11 at the same size (the attention blocks), hence
# the +0.5 GB on each estimate.


@register
class Yolo12NanoTrainer(UltralyticsTrainer):
    key = "yolo12n"
    family = "YOLO12"
    variant = "nano"
    display_name = "YOLO12 nano"
    description = (
        "Attention-based YOLO, better accuracy than YOLO11 at every size. "
        "Nano trains on modest GPUs — the right starting point on most machines."
    )
    approx_vram_gb = 3.5
    default_batch_size = 8
    base_weights = "yolo12n.pt"


@register
class Yolo12SmallTrainer(UltralyticsTrainer):
    key = "yolo12s"
    family = "YOLO12"
    variant = "small"
    display_name = "YOLO12 small"
    description = "A step up in accuracy from nano for roughly half again the memory."
    approx_vram_gb = 5.0
    default_batch_size = 8
    base_weights = "yolo12s.pt"


@register
class Yolo12MediumTrainer(UltralyticsTrainer):
    key = "yolo12m"
    family = "YOLO12"
    variant = "medium"
    display_name = "YOLO12 medium"
    description = "The middle of the family — needs a mid-range GPU (7 GB+)."
    approx_vram_gb = 7.0
    default_batch_size = 4
    base_weights = "yolo12m.pt"


@register
class Yolo12LargeTrainer(UltralyticsTrainer):
    key = "yolo12l"
    family = "YOLO12"
    variant = "large"
    display_name = "YOLO12 large"
    description = "High accuracy, slower — wants 9 GB+ of VRAM."
    approx_vram_gb = 9.0
    default_batch_size = 4
    base_weights = "yolo12l.pt"


@register
class Yolo12XLargeTrainer(UltralyticsTrainer):
    key = "yolo12x"
    family = "YOLO12"
    variant = "xlarge"
    display_name = "YOLO12 xlarge"
    description = "The biggest YOLO12. Best accuracy in the family; wants 12 GB+."
    approx_vram_gb = 12.0
    default_batch_size = 2
    base_weights = "yolo12x.pt"


# --- YOLO26: ultralytics' newest family — NMS-free end-to-end ---------------


@register
class Yolo26NanoTrainer(UltralyticsTrainer):
    key = "yolo26n"
    family = "YOLO26"
    variant = "nano"
    display_name = "YOLO26 nano"
    description = (
        "Ultralytics' newest family: NMS-free end-to-end detection, faster "
        "inference and better small-object accuracy than YOLO11/12. Nano fits "
        "modest GPUs."
    )
    approx_vram_gb = 3.0
    default_batch_size = 8
    base_weights = "yolo26n.pt"


@register
class Yolo26SmallTrainer(UltralyticsTrainer):
    key = "yolo26s"
    family = "YOLO26"
    variant = "small"
    display_name = "YOLO26 small"
    description = "A step up in accuracy from nano for roughly half again the memory."
    approx_vram_gb = 4.5
    default_batch_size = 8
    base_weights = "yolo26s.pt"


@register
class Yolo26MediumTrainer(UltralyticsTrainer):
    key = "yolo26m"
    family = "YOLO26"
    variant = "medium"
    display_name = "YOLO26 medium"
    description = "The middle of the family — needs a mid-range GPU (6 GB+)."
    approx_vram_gb = 6.5
    default_batch_size = 4
    base_weights = "yolo26m.pt"


@register
class Yolo26LargeTrainer(UltralyticsTrainer):
    key = "yolo26l"
    family = "YOLO26"
    variant = "large"
    display_name = "YOLO26 large"
    description = "High accuracy, slower — wants 8 GB+ of VRAM."
    approx_vram_gb = 8.5
    default_batch_size = 4
    base_weights = "yolo26l.pt"


@register
class Yolo26XLargeTrainer(UltralyticsTrainer):
    key = "yolo26x"
    family = "YOLO26"
    variant = "xlarge"
    display_name = "YOLO26 xlarge"
    description = "The biggest YOLO26. Best accuracy in the family; wants 10 GB+."
    approx_vram_gb = 11.0
    default_batch_size = 2
    base_weights = "yolo26x.pt"


# --- RT-DETR: Baidu's real-time detection transformer -----------------------


@register
class RtDetrLTrainer(UltralyticsTrainer):
    key = "rtdetr_l"
    family = "RT-DETR"
    variant = "L"
    display_name = "RT-DETR L"
    description = (
        "Baidu's real-time detection transformer (via ultralytics). NMS-free, "
        "competitive with the larger YOLOs; heavier to train — wants 10 GB+."
    )
    approx_vram_gb = 10.0
    default_batch_size = 4
    base_weights = "rtdetr-l.pt"
    model_class = "RTDETR"


@register
class RtDetrXTrainer(UltralyticsTrainer):
    key = "rtdetr_x"
    family = "RT-DETR"
    variant = "X"
    display_name = "RT-DETR X"
    description = (
        "The bigger RT-DETR — best transformer accuracy in the roster, and the "
        "hungriest to train. Wants 14 GB+."
    )
    approx_vram_gb = 14.0
    default_batch_size = 2
    base_weights = "rtdetr-x.pt"
    model_class = "RTDETR"
