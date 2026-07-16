"""
Dataset export interface.

The third pluggable seam in this project, alongside AutoAnnotator and (Phase 4)
Trainer. Same reasoning: the DB is the source of truth, and each format is an
adapter at the edge.

This is not academic. Phase 4 needs BOTH formats simultaneously:

    RF-DETR   -> COCO
    RT-DETR   -> COCO
    YOLO26    -> YOLO txt + data.yaml

So a conversion step is unavoidable regardless of storage choice. The only
question is whether it converts from a good source (queryable rows) or an
awkward one (a JSON file you must parse first). This is the argument for rows.

THE FORMATS DISAGREE ABOUT EVERYTHING
-------------------------------------
                COCO                        YOLO
    coords      absolute pixels             normalised 0..1
    anchor      top-left (x, y, w, h)       centre (cx, cy, w, h)
    layout      one JSON for the dataset    one .txt per image
    classes     categories[] with ids       index into data.yaml names[]
    ids         category_id (1-based, ours) class index (0-based, always)

Every one of those is a silent-failure opportunity. Getting the anchor wrong
doesn't crash — it shifts every box by half its size and you find out when mAP
is inexplicably 0.3. The conversions live here, once, tested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session


@dataclass
class ExportRequest:
    """What to export and where."""

    project_id: int
    out_dir: Path

    #: When False, only boxes a human has confirmed (reviewed=True) are written.
    #: The point of the whole review workflow — train on verified data, or
    #: knowingly train on drafts.
    include_unreviewed: bool = True

    #: image_id -> "train" | "val". None means everything goes to train.
    #: Populated by the Phase 6 train/val split UI.
    split: dict[int, str] | None = None

    #: Copy image files next to the labels. Required by YOLO (it resolves images
    #: by path convention); optional for COCO, which only records file_name.
    copy_images: bool = True

    def split_for(self, image_id: int) -> str:
        return (self.split or {}).get(image_id, "train")


class DatasetExporter(ABC):
    """Base class for dataset format writers."""

    key: str = ""
    display_name: str = ""
    description: str = ""

    @abstractmethod
    def export(self, db: Session, request: ExportRequest) -> Path:
        """Write the dataset. Returns the root directory written."""


# --- Shared coordinate maths ------------------------------------------------
# Defined once, here, rather than inline in each exporter. These four lines are
# where detection pipelines most often go quietly wrong.


def coco_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """COCO absolute top-left xywh -> YOLO normalised centre cxcywh.

    Two transformations at once, which is why it's error-prone:
      1. top-left origin -> centre origin  (add half the extent)
      2. absolute pixels -> 0..1           (divide by image size)

    Clamped to [0, 1] because a box touching the image edge can compute to
    1.0000001 through float error, and some YOLO loaders reject out-of-range
    values outright while others silently train on them.
    """
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def yolo_to_coco(
    cx: float, cy: float, nw: float, nh: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """YOLO normalised centre cxcywh -> COCO absolute top-left xywh.

    The inverse. Needed for importing YOLO datasets (and worth having so the
    round-trip can be tested — a conversion pair you can't round-trip is a
    conversion pair you don't trust).
    """
    w = nw * img_w
    h = nh * img_h
    x = (cx * img_w) - w / 2
    y = (cy * img_h) - h / 2
    return x, y, w, h
