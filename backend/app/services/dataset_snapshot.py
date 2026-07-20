"""
A dataset snapshot — the dataset's contents at one moment, detached from the DB.

WHY THIS EXISTS
---------------
Two things need "the dataset": exporting the live project, and exporting a SAVED
VERSION from months ago. Before this, exporters queried the Session directly, so
there was no way to export anything but the current rows — which makes "train on
dataset v3" impossible to honour truthfully.

So exporters now consume a DatasetSnapshot instead of a Session. The live dataset
builds one on demand (`build_snapshot`); a saved version loads one from its JSON
file (`read_snapshot`). One exporter code path serves both, which matters because
the coordinate maths and layout conventions in there are the easiest thing in the
project to get subtly wrong — duplicating them for versions would guarantee drift.

WHAT A SNAPSHOT CONTAINS (and deliberately doesn't)
---------------------------------------------------
Metadata only: which images are in the dataset, each one's split, and every
ACCEPTED box. Image BYTES are never copied — a snapshot is kilobytes, so keeping
dozens of versions costs nothing. That's only safe because deleting an image
leaves its file on disk (see services/storage.py), so any version can still
resolve the pictures it references.

Proposals are excluded, always. A pending model suggestion is not part of the
dataset until accepted, so it cannot be part of a version of it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

#: Bumped if the on-disk shape ever changes incompatibly, so an old file can be
#: recognised rather than mis-parsed into confidently wrong boxes.
SNAPSHOT_FORMAT = 1


@dataclass
class SnapshotAnnotation:
    """One accepted box. Mirrors the Annotation columns an export needs."""

    category_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float | None = None
    source: str = "manual"
    reviewed: bool = True
    #: The original row id, preserved so COCO annotation ids stay stable across
    #: exports of the same version. None for snapshots that predate it.
    id: int | None = None

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class SnapshotImage:
    id: int
    filename: str  # the uuid name on disk
    original_filename: str
    width: int
    height: int
    size_bytes: int
    split: str
    annotations: list[SnapshotAnnotation] = field(default_factory=list)


@dataclass
class SnapshotCategory:
    id: int
    name: str
    color: str


@dataclass
class DatasetSnapshot:
    project_id: int
    project_name: str
    categories: list[SnapshotCategory] = field(default_factory=list)
    images: list[SnapshotImage] = field(default_factory=list)

    # --- derived counts, used for the version list and readiness checks -----

    @property
    def total_boxes(self) -> int:
        return sum(len(i.annotations) for i in self.images)

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for image in self.images:
            counts[image.split] = counts.get(image.split, 0) + 1
        return counts

    def box_count_for_split(self, split: str) -> int:
        return sum(len(i.annotations) for i in self.images if i.split == split)

    def content_hash(self) -> str:
        """Stable fingerprint of what this dataset CONTAINS.

        Lets two dataset states be compared without diffing them by eye, which
        answers two questions the app kept getting wrong:

          - "is the live dataset already saved?" — if some version's hash
            matches, nothing would be lost, so there is nothing to back up.
          - "which version am I actually looking at?" — the one whose hash
            matches the live data, which is NOT necessarily the newest.

        Hashes the meaning, not the bookkeeping: classes and boxes are keyed by
        class NAME (row ids change when a class is deleted and recreated, and a
        restore does exactly that), images by their stored filename, and
        coordinates rounded so float noise can't make identical datasets look
        different. Order is normalised so it depends on content alone.
        """
        by_id = {c.id: c.name for c in self.categories}
        images = sorted(
            (
                img.filename,
                img.split,
                sorted(
                    (
                        by_id.get(a.category_id, "?"),
                        round(a.x, 3),
                        round(a.y, 3),
                        round(a.width, 3),
                        round(a.height, 3),
                    )
                    for a in img.annotations
                ),
            )
            for img in self.images
        )
        payload = {"classes": sorted(by_id.values()), "images": images}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    # --- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"format": SNAPSHOT_FORMAT, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetSnapshot":
        return cls(
            project_id=int(data["project_id"]),
            project_name=str(data.get("project_name", "")),
            categories=[SnapshotCategory(**c) for c in data.get("categories", [])],
            images=[
                SnapshotImage(
                    **{k: v for k, v in img.items() if k != "annotations"},
                    annotations=[
                        SnapshotAnnotation(**a) for a in img.get("annotations", [])
                    ],
                )
                for img in data.get("images", [])
            ],
        )


def build_snapshot(db: Session, project_id: int) -> DatasetSnapshot:
    """Capture the project's CURRENT dataset.

    Reads accepted boxes only (`proposed=False`) — proposals aren't dataset
    content. Ordered by id throughout so the class index assignment an exporter
    derives is deterministic for a given snapshot.
    """
    from app.models import Annotation, Category, Image, Project

    project = db.get(Project, project_id)
    if project is None:
        raise ValueError(f"Project {project_id} not found")

    categories = [
        SnapshotCategory(id=c.id, name=c.name, color=c.color)
        for c in db.scalars(
            select(Category).where(Category.project_id == project_id).order_by(Category.id)
        ).all()
    ]

    rows = db.scalars(
        select(Image).where(Image.project_id == project_id).order_by(Image.id)
    ).all()

    images: list[SnapshotImage] = []
    for image in rows:
        anns = db.scalars(
            select(Annotation)
            .where(Annotation.image_id == image.id, Annotation.proposed.is_(False))
            .order_by(Annotation.id)
        ).all()
        images.append(
            SnapshotImage(
                id=image.id,
                filename=image.filename,
                original_filename=image.original_filename,
                width=image.width,
                height=image.height,
                size_bytes=image.size_bytes,
                split=image.split,
                annotations=[
                    SnapshotAnnotation(
                        id=a.id,
                        category_id=a.category_id,
                        x=a.x,
                        y=a.y,
                        width=a.width,
                        height=a.height,
                        confidence=a.confidence,
                        source=a.source,
                        reviewed=a.reviewed,
                    )
                    for a in anns
                ],
            )
        )

    return DatasetSnapshot(
        project_id=project_id,
        project_name=project.name,
        categories=categories,
        images=images,
    )


def write_snapshot(snapshot: DatasetSnapshot, path: Path) -> None:
    """Persist a snapshot as JSON. On disk, not in the DB: it's a document that
    can reach megabytes on a large project, and this project keeps blobs on the
    filesystem with only the path recorded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_dict()), encoding="utf-8")


def read_snapshot(path: Path) -> DatasetSnapshot:
    """Load a saved snapshot."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetSnapshot.from_dict(data)
