"""
RF-DETR trainers — Roboflow's 2025 real-time detection transformer.

The current state of the art for real-time COCO mAP, and a genuinely different
lineage from everything ultralytics ships: a DETR head over a DINOv2 backbone.
Comes from its own `rfdetr` package (added to requirements-ml with its [train]
extras — the training path runs on PyTorch Lightning).

THREE ADAPTATIONS THIS MODULE OWNS
----------------------------------
1. DATASET LAYOUT. rfdetr reads Roboflow's COCO convention — train/ valid/
   test/ folders with `_annotations.coco.json` and the images DIRECTLY beside
   it. Our exporter writes val/ (not valid/) and nests files under images/.
   `_adapt_layout` renames and flattens the job's own exported copy in place —
   it's scratch space, so mutating it is free.

2. EPOCH PROGRESS. rfdetr 1.7 discards its legacy callbacks dict ("not
   forwarded to PTL"), so the supported seam is a real Lightning Callback on
   the Trainer. We get one in by patching `rfdetr.training.build_trainer` for
   the duration of the run — train() re-imports it per call, so the patch
   takes effect — and APPENDING our bridge to the built trainer's callbacks
   rather than passing callbacks= through trainer_kwargs, which would REPLACE
   rfdetr's whole stack (EMA, best-checkpoint saving, COCO eval).
   The bridge reads the keys rfdetr actually logs (train/loss,
   val/mAP_50_95, val/mAP_50) and honours stop/cancel via
   `trainer.should_stop`, Lightning's graceful end-of-epoch stop.

3. RESOLUTION. Our image_size knob is NOT forwarded: RF-DETR trains at its
   variant's fixed resolution (DINOv2 patch/window divisibility rules), and a
   not-quite-divisible value is a hard ValueError. The trade is documented in
   the UI description rather than silently misbehaving.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.ml.trainers.base import EpochCallback, EpochMetrics, TrainConfig, TrainResult, Trainer
from app.ml.trainers.registry import register

logger = logging.getLogger(__name__)

# build_trainer is patched per-run to inject the epoch bridge; two concurrent
# RF-DETR runs (possible on a large card) must not fight over the patch.
_patch_lock = threading.Lock()


def _adapt_layout(dataset_dir: Path) -> None:
    """Rearrange our COCO export into Roboflow's convention, in place.

    {split}/images/*.jpg + {split}/_annotations.coco.json
      ->  train|valid|test/*.jpg beside the json, with val renamed valid.
    """
    mapping = {"train": "train", "val": "valid", "test": "test"}
    for ours, theirs in mapping.items():
        src = dataset_dir / ours
        if not src.exists():
            continue
        dst = dataset_dir / theirs
        if src != dst:
            src.rename(dst)
        images = dst / "images"
        if images.exists():
            for f in images.iterdir():
                f.rename(dst / f.name)
            images.rmdir()


class RfDetrTrainer(Trainer):
    """Shared adapter; subclasses pick the variant class and its cost."""

    family = "RF-DETR"
    export_format = "coco"
    default_epochs = 50
    # Not used by RF-DETR (fixed-resolution architecture) but the schema wants
    # a number; the UI description says the knob doesn't apply.
    default_image_size = 560

    #: Name of the model class inside the rfdetr package.
    model_class = "RFDETRNano"

    def load_predictor(self, checkpoint_path, class_names: list[str]):
        from app.ml.predictors.rf_detr import RfDetrPredictor

        return RfDetrPredictor(checkpoint_path, class_names)

    def train(self, config: TrainConfig, on_epoch: EpochCallback) -> TrainResult:
        try:
            import rfdetr
            import rfdetr.training as rf_training
        except ImportError as exc:
            raise RuntimeError(
                "rfdetr is not installed. Run `pip install \"rfdetr[train]\"` in "
                "the backend venv and restart the server."
            ) from exc

        _adapt_layout(config.dataset_dir)

        # Continue from a previous run's checkpoint, or the pretrained base.
        # from_checkpoint rebuilds the exact architecture the file was trained
        # with, so a continued run can't silently switch variants.
        if config.init_weights is not None:
            model = rfdetr.from_checkpoint(str(config.init_weights))
        else:
            model = getattr(rfdetr, self.model_class)()

        best_map_seen: float | None = None
        stop_requested = False

        import pytorch_lightning as pl

        class _EpochBridge(pl.Callback):
            """Forward each epoch's metrics to the runner; honour stop/cancel."""

            def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
                nonlocal best_map_seen, stop_requested
                if stop_requested:
                    return
                metrics = trainer.callback_metrics

                def metric(*keys: str) -> float | None:
                    for k in keys:
                        v = metrics.get(k)
                        if v is not None:
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                return None
                    return None

                val_map = metric("val/mAP_50_95")
                val_map50 = metric("val/mAP_50")
                if val_map is not None:
                    best_map_seen = (
                        val_map if best_map_seen is None else max(best_map_seen, val_map)
                    )
                try:
                    should_stop = on_epoch(
                        EpochMetrics(
                            epoch=int(trainer.current_epoch) + 1,  # PTL is 0-based
                            total_epochs=int(trainer.max_epochs or config.epochs),
                            train_loss=metric("train/loss"),
                            val_map=val_map,
                            val_map50=val_map50,
                        )
                    )
                except Exception:  # noqa: BLE001 — a DB hiccup must not kill the run
                    logger.exception("Failed to record epoch progress")
                    return
                if should_stop:
                    logger.info("Stop requested — finishing after this epoch")
                    stop_requested = True
                    # Lightning's graceful stop: finishes the epoch, runs the
                    # checkpoint callbacks, then exits fit().
                    trainer.should_stop = True

        # Inject the bridge by wrapping build_trainer for this run only.
        # APPEND to the built trainer's callbacks — passing callbacks= through
        # trainer_kwargs would replace rfdetr's own stack wholesale.
        with _patch_lock:
            original_build = rf_training.build_trainer

            def build_with_bridge(*args, **kwargs):  # noqa: ANN002, ANN003
                trainer = original_build(*args, **kwargs)
                trainer.callbacks.append(_EpochBridge())
                return trainer

            rf_training.build_trainer = build_with_bridge
        try:
            kwargs: dict = dict(
                dataset_dir=str(config.dataset_dir),
                output_dir=str(config.output_dir),
                epochs=config.epochs,
                batch_size=config.batch_size,
                # Keep the EFFECTIVE batch near rfdetr's recommended 16 however
                # small the per-step batch had to be for the card.
                grad_accum_steps=max(1, 16 // max(1, config.batch_size)),
                device=config.device,
                # No plots/telemetry side-channels; progress goes through the
                # bridge and the captured logger.
                tensorboard=False,
                wandb=False,
                # No console progress bar. We render progress ourselves — and
                # rfdetr's default Rich bar writes unicode box-drawing to
                # stdout, which CRASHES the whole run with UnicodeEncodeError
                # on a Windows cp1252 console (observed live).
                progress_bar=None,
            )
            if config.learning_rate is not None:
                kwargs["lr"] = config.learning_rate

            model.train(**kwargs)
        finally:
            with _patch_lock:
                rf_training.build_trainer = original_build

        # rfdetr's BestModelCallback keeps the winner (regular vs EMA) here.
        out = Path(config.output_dir)
        best = next(
            (
                p
                for p in (
                    out / "checkpoint_best_total.pth",
                    out / "checkpoint_best_regular.pth",
                    out / "checkpoint_best_ema.pth",
                )
                if p.exists()
            ),
            None,
        )

        return TrainResult(
            best_checkpoint_path=best,
            best_map=best_map_seen,
            epochs_completed=config.epochs,
        )


_DESCRIPTION_TAIL = (
    " Trains at the architecture's own fixed resolution — the image-size "
    "setting does not apply."
)


@register
class RfDetrNanoTrainer(RfDetrTrainer):
    key = "rfdetr_nano"
    variant = "nano"
    display_name = "RF-DETR nano"
    description = (
        "Roboflow's 2025 detection transformer — state-of-the-art real-time "
        "accuracy. Nano is the entry point for modest GPUs." + _DESCRIPTION_TAIL
    )
    approx_vram_gb = 4.0
    default_batch_size = 4
    model_class = "RFDETRNano"


@register
class RfDetrSmallTrainer(RfDetrTrainer):
    key = "rfdetr_small"
    variant = "small"
    display_name = "RF-DETR small"
    description = (
        "A step up in accuracy from nano; wants 6 GB+ to train." + _DESCRIPTION_TAIL
    )
    approx_vram_gb = 6.0
    default_batch_size = 2
    model_class = "RFDETRSmall"


@register
class RfDetrMediumTrainer(RfDetrTrainer):
    key = "rfdetr_medium"
    variant = "medium"
    display_name = "RF-DETR medium"
    description = (
        "The strongest RF-DETR here — best transformer accuracy in the "
        "roster; wants 9 GB+ to train." + _DESCRIPTION_TAIL
    )
    approx_vram_gb = 9.0
    default_batch_size = 2
    model_class = "RFDETRMedium"
