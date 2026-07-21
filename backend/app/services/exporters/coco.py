"""COCO JSON exporter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from app.services import storage
from app.services.dataset_snapshot import DatasetSnapshot
from app.timestamps import utcnow
from app.services.exporters.base import DatasetExporter, ExportRequest


class CocoExporter(DatasetExporter):
    key = "coco"
    display_name = "COCO JSON"
    description = "Standard COCO detection format. Consumed by RF-DETR, RT-DETR, and most tools."

    def export(self, snapshot: DatasetSnapshot, request: ExportRequest) -> Path:
        root = request.out_dir
        root.mkdir(parents=True, exist_ok=True)

        project_name = snapshot.project_name
        categories = snapshot.categories
        images = snapshot.images

        # COCO category ids are 1-based by convention (id 0 is reserved for
        # background in many tools). Our DB ids happen to start at 1 too, but
        # relying on that would be a latent bug the first time a class is
        # deleted and the ids develop gaps — so we renumber explicitly and keep
        # a map.
        cat_id_map = {c.id: i + 1 for i, c in enumerate(categories)}

        # COCO is one file per SPLIT, not one per dataset — train and val must
        # be separate annotation files, each with its own image list.
        splits: dict[str, dict] = {}

        for image in images:
            split = image.split
            bucket = splits.setdefault(
                split,
                {
                    "info": {
                        "description": project_name,
                        "version": "1.0",
                        "date_created": utcnow().isoformat(),
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": [
                        {
                            "id": cat_id_map[c.id],
                            "name": c.name,
                            "supercategory": "",
                        }
                        for c in categories
                    ],
                },
            )

            bucket["images"].append(
                {
                    "id": image.id,
                    "file_name": image.original_filename,
                    "width": image.width,
                    "height": image.height,
                }
            )

            # Proposals never reach here — a snapshot only ever holds accepted
            # boxes, so exporting suggestions as ground truth is impossible by
            # construction rather than by remembering a filter.
            anns = image.annotations
            if not request.include_unreviewed:
                anns = [a for a in anns if a.reviewed]

            for idx, ann in enumerate(anns):
                bucket["annotations"].append(
                    {
                        # Annotation id, unique across the file. Falls back to a
                        # positional id for snapshots written without one.
                        "id": ann.id if ann.id is not None else idx + 1,
                        "image_id": image.id,
                        "category_id": cat_id_map[ann.category_id],
                        # Direct field copy — no maths. This is the payoff for
                        # storing in COCO's convention (absolute top-left xywh)
                        # rather than xyxy.
                        "bbox": [ann.x, ann.y, ann.width, ann.height],
                        "area": ann.area,
                        # 0 = a single object; 1 = a crowd region encoded as RLE.
                        # We only produce single objects, and pycocotools treats
                        # iscrowd=1 boxes differently during evaluation, so
                        # getting this wrong silently skews mAP.
                        "iscrowd": 0,
                        # Non-standard but widely tolerated. Useful for filtering
                        # low-confidence drafts downstream.
                        "score": ann.confidence,
                    }
                )

            if request.copy_images:
                dest_dir = root / split / "images"
                dest_dir.mkdir(parents=True, exist_ok=True)
                src = storage.project_dir(snapshot.project_id) / image.filename
                if src.exists():
                    # Copy under the ORIGINAL filename, matching file_name above.
                    # Our uuid storage names are an internal detail that would be
                    # meaningless in an exported dataset.
                    shutil.copy2(src, dest_dir / image.original_filename)

        for split, payload in splits.items():
            split_dir = root / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "_annotations.coco.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )

        return root
