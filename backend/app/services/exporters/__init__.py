"""
Dataset exporters.

Registry of output formats. Same pattern as app/ml/registry.py — a dict keyed by
a string the API accepts, so adding Pascal VOC or TFRecord later is one class
plus one line here, with no route or UI change.

Simpler than the annotator registry because exporters are stateless and cheap:
no lifecycle, no VRAM, nothing to evict. Instantiating one costs nothing, so
there's no lazy-loading dance.
"""

from app.services.exporters.base import (
    DatasetExporter,
    ExportRequest,
    coco_to_yolo,
    yolo_to_coco,
)
from app.services.exporters.coco import CocoExporter
from app.services.exporters.yolo import YoloExporter

_EXPORTERS: dict[str, type[DatasetExporter]] = {
    CocoExporter.key: CocoExporter,
    YoloExporter.key: YoloExporter,
}


def available() -> list[dict]:
    """Metadata for the UI's format dropdown."""
    return [
        {"key": c.key, "display_name": c.display_name, "description": c.description}
        for c in _EXPORTERS.values()
    ]


def get(key: str) -> DatasetExporter:
    try:
        return _EXPORTERS[key]()
    except KeyError:
        raise KeyError(
            f"Unknown export format {key!r}. Available: {sorted(_EXPORTERS)}"
        ) from None


__all__ = [
    "DatasetExporter",
    "ExportRequest",
    "CocoExporter",
    "YoloExporter",
    "available",
    "get",
    "coco_to_yolo",
    "yolo_to_coco",
]
