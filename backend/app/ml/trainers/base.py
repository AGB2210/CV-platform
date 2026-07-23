"""
The Trainer interface.

The fourth pluggable seam in the project, alongside AutoAnnotator, DatasetExporter
and (Phase 5) the evaluators. Same reasoning as AutoAnnotator: adding a training
backend means writing one class and registering it — the job runner, the API and
the UI never change, because they only ever talk to this interface.

WHY THIS LOOKS DIFFERENT FROM AutoAnnotator
-------------------------------------------
An annotator is a thing WE drive: we load it, call predict() per image, unload it.
So its base class owns a load/unload/predict lifecycle and the registry keeps one
resident across a batch.

A trainer is the opposite. Frameworks like ultralytics and rfdetr own their whole
training loop — they load the model, iterate epochs, checkpoint, and free VRAM
themselves. We don't get to step inside. So a Trainer exposes ONE call, train(),
and reports back through a callback the framework invokes each epoch. There is no
per-step lifecycle for us to manage, so there isn't one on the base class.

What we DO still own is the memory constraint: before a trainer runs, any resident
annotator must be evicted (the job runner does this), and only one trainer may run
at a time (the route guards this). torch is imported lazily inside train(), never
at module top level — importing this module must stay free so the /api/trainers
menu can be rendered with the heavy deps uninstalled.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything a trainer needs to run one job.

    A dataclass rather than a pile of arguments, for the same reason as
    AnnotationRequest: every trainer receives the same shape, so the runner
    builds one object and hands it to whichever backend the user picked without
    knowing which.

    Paths, not data. The dataset has already been exported to disk in this
    trainer's `export_format` (COCO json or the YOLO images/labels layout), and
    `dataset_dir` is its root. The adapter knows its own format's conventions —
    where data.yaml lives, where the COCO json is — and derives them from here.
    """

    #: Root of the exported dataset, in this trainer's export_format.
    dataset_dir: Path
    #: Where the trainer may write checkpoints, logs, and its own run artifacts.
    output_dir: Path

    epochs: int
    batch_size: int
    #: Square training resolution. Memory scales with its square, so it is one
    #: of the first things to lower when a run won't fit.
    image_size: int
    #: None means "use the framework's default schedule", which is usually the
    #: right call — its defaults are tuned per-architecture and we shouldn't
    #: pretend to know better without a reason.
    learning_rate: float | None

    num_classes: int
    #: In the SAME ORDER the exporter assigned class indices, so a checkpoint's
    #: output channel N always means class_names[N]. Off-by-one here trains a
    #: model that confidently calls cars "people" — see the exporter's note.
    class_names: list[str]

    #: "cuda" | "cpu", from app.ml.device.get_device().
    device: str

    #: Weights to START from. None = the trainer's pretrained base (train from
    #: scratch-ish). A path = continue/finetune from a previous run's checkpoint,
    #: so you build on prior training instead of re-learning from zero each time.
    #: The class set must match what that checkpoint was trained on.
    init_weights: Path | None = None

    #: Polled BETWEEN BATCHES: returns True when the user has cancelled the run.
    #: CANCEL only, deliberately — cancel throws the whole run away, so there is
    #: nothing to protect and it should take effect in seconds, not at the end
    #: of an epoch that might be crawling in spilled GPU memory. STOP stays an
    #: epoch-boundary decision (via the on_epoch return value) because stopping
    #: KEEPS the model, and a checkpoint is worth waiting one epoch for.
    #: Implementations should call this cheaply (the runner throttles the
    #: underlying DB read) and abort as soon as it returns True.
    check_cancel: Callable[[], bool] | None = None


@dataclass
class EpochMetrics:
    """One epoch's results, handed to the runner's callback as they land.

    Every field is optional except the epoch counters: not all frameworks
    surface a validation mAP every epoch (some evaluate only periodically), and
    a missing metric must read as "not measured this epoch", not as zero — a
    zero mAP plotted mid-run looks like the model collapsed.
    """

    epoch: int  # 1-based
    total_epochs: int
    train_loss: float | None = None
    #: COCO mAP averaged over IoU .50:.95 — the headline number.
    val_map: float | None = None
    #: mAP at IoU .50 — the looser, more forgiving figure, shown alongside.
    val_map50: float | None = None


@dataclass
class TrainResult:
    """What a finished run leaves behind."""

    #: Best-performing checkpoint on the val set. None if training produced no
    #: usable weights (e.g. it stopped before the first checkpoint).
    best_checkpoint_path: Path | None
    best_map: float | None
    epochs_completed: int


#: The callback a trainer invokes once per epoch. It writes progress to the DB;
#: the trainer must not assume anything about what it does or how long it takes.
#:
#: RETURNS True to mean "stop after this epoch". That's how the user's Stop and
#: Cancel reach a framework that owns its own loop: we can't interrupt training
#: from outside, but we are handed control once per epoch and can decline to
#: continue. A trainer MUST honour it — gracefully, letting the epoch in flight
#: finish and its checkpoint be written, because a half-written epoch is worse
#: than one more epoch of waiting.
#:
#: Both Stop and Cancel look identical here. The difference is what the runner
#: does with the result afterwards, which is none of the trainer's business.
EpochCallback = Callable[[EpochMetrics], bool]


class Trainer(ABC):
    """Base class for training backends.

    A subclass declares its metadata and which export format it consumes, then
    implements train(). Everything else — exporting the dataset, evicting the
    annotator, recording progress, saving the run — is the job runner's job, so
    that a trainer stays a thin adapter over one framework.
    """

    #: Stable identifier used in the registry, the API, and the DB. Changing it
    #: breaks existing job records, so treat it as permanent.
    key: str = ""
    #: Shown in the UI's trainer dropdown.
    display_name: str = ""
    #: Grouping for the UI's picker: models come in FAMILIES (YOLO12, RT-DETR)
    #: whose members differ only in size/speed. A flat list of six entries
    #: hides that structure; family + variant lets the UI offer "which
    #: architecture" and "which size" as the two separate questions they are.
    family: str = ""
    #: The size/flavour within the family, e.g. "nano" or "L".
    variant: str = ""
    description: str = ""
    #: Offered for NEW runs? False = legacy: hidden from the picker, but still
    #: registered so existing checkpoints trained with this key keep loading
    #: (deploy, evaluate, weights download). This is how a family gets
    #: superseded without breaking every model it ever trained.
    listed: bool = True
    #: Rough peak VRAM at the DEFAULT settings, surfaced so the user can tell
    #: what will fit before they wait for an OOM. Batch/image size move it.
    approx_vram_gb: float = 0.0

    #: Which exporter feeds this trainer: "coco" or "yolo". The runner reads this
    #: to pick the exporter, so a trainer never parses the DB itself — it only
    #: ever sees files in the layout it asked for.
    export_format: str = ""

    #: Defaults the UI pre-fills the config form with. Per-trainer because a
    #: sane batch size for YOLO-nano is not a sane one for a DETR.
    default_epochs: int = 50
    default_batch_size: int = 8
    default_image_size: int = 640

    @abstractmethod
    def train(self, config: TrainConfig, on_epoch: EpochCallback) -> TrainResult:
        """Run one training job to completion.

        Must call `on_epoch` once per completed epoch with whatever metrics are
        available, and return a TrainResult pointing at the best checkpoint.

        Converting the framework's own progress signal (ultralytics callbacks,
        rfdetr's logs) into EpochMetrics is the adapter's job — precisely what
        this interface exists to contain. Raising is fine: the runner catches
        it, records the traceback, and marks the job failed. An OOM should be
        allowed to propagate with its original message intact.
        """

    def export_onnx(self, checkpoint_path) -> "Path":
        """Convert one of this trainer's checkpoints to ONNX; return the file.

        Lives on the trainer for the same reason load_predictor does: the
        framework that WROTE the weights is the only thing that knows how to
        read them, and pairing them structurally means they can't drift apart.
        Implementations should run on CPU (an export must not fight a training
        run for the GPU) and write beside the checkpoint so the result is
        cached — converting is slow, serving a file is not.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support ONNX export yet."
        )

    def load_predictor(self, checkpoint_path, class_names: list[str]):
        """Load a checkpoint this trainer produced into a runnable Predictor.

        The trainer owns this — not a separate registry keyed by architecture —
        because the framework that WROTE the weights is the only thing that knows
        how to read them. Pairing them structurally means the two can't drift
        apart. Returns a Predictor (app/ml/predictors/base.py); the predictor
        registry keeps at most one resident.

        Not abstract only so a trainer without inference yet still imports; a
        trainer that can't predict raises here rather than at class definition.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support inference yet."
        )
