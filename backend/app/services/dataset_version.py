"""
Dataset versions: save a restorable snapshot, and restore one.

The safety net for the dataset. Saving writes an immutable JSON snapshot of what
the dataset looks like right now; restoring rewinds the live rows to match one.

TWO RULES THAT MAKE THIS SAFE
-----------------------------
1. **Restoring is itself undoable.** It takes a snapshot of the CURRENT state
   first ("before restoring vN"), so a mistaken restore is one more restore away
   from being fixed. Nothing this module does is a one-way door.
2. **Image bytes are never deleted.** Deleting an image drops its row but leaves
   the file (services/storage.py), so a restore can recreate the row pointing at
   the same bytes. Without that, versions would be a promise we couldn't keep.

WHAT RESTORE TOUCHES — AND WHAT IT LEAVES ALONE
----------------------------------------------
It rewrites the dataset: which images exist, their splits, and their ACCEPTED
boxes. Pending proposals on surviving images are left untouched — they aren't
dataset content (they're a model's un-actioned suggestions), so a version never
captured them and restoring shouldn't silently throw them away. Proposals on
images the version doesn't contain disappear with the image, by cascade.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.services.dataset_snapshot import (
    DatasetSnapshot,
    build_snapshot,
    read_snapshot,
    write_snapshot,
)

logger = logging.getLogger(__name__)


class DatasetVersionError(Exception):
    """Domain error. Services never raise HTTPException — the route maps it."""


def snapshot_path_for(project_id: int, version: int) -> Path:
    return settings.versions_dir / str(project_id) / f"v{version}.json"


def next_version_number(db: Session, project_id: int) -> int:
    """1-based, per project. Counts every version ever saved, so numbers are
    never reused even if one is removed."""
    from app.models import DatasetVersion

    current = db.scalar(
        select(func.max(DatasetVersion.version)).where(
            DatasetVersion.project_id == project_id
        )
    )
    return (current or 0) + 1


def save_version(db: Session, project_id: int, note: str | None = None):
    """Capture the project's current dataset as a new version."""
    from app.models import DatasetVersion
    from app.models.image import Split

    snapshot = build_snapshot(db, project_id)
    if not snapshot.images:
        raise DatasetVersionError(
            "There are no images to save. Upload images before saving a dataset version."
        )

    version = next_version_number(db, project_id)
    path = snapshot_path_for(project_id, version)
    write_snapshot(snapshot, path)

    counts = snapshot.split_counts()
    row = DatasetVersion(
        project_id=project_id,
        version=version,
        note=(note or None),
        snapshot_path=str(path),
        total_images=len(snapshot.images),
        train_images=counts.get(Split.TRAIN, 0),
        val_images=counts.get(Split.VAL, 0),
        test_images=counts.get(Split.TEST, 0),
        total_boxes=snapshot.total_boxes,
        num_classes=len(snapshot.categories),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("Saved dataset v%s for project %s", version, project_id)
    return row


def load_snapshot(version) -> DatasetSnapshot:
    """Read a version's snapshot from disk."""
    path = Path(version.snapshot_path)
    if not path.exists():
        raise DatasetVersionError(
            f"The snapshot file for dataset v{version.version} is missing "
            f"({path}). It may have been deleted from storage."
        )
    return read_snapshot(path)


@dataclass
class RestoreResult:
    images_restored: int
    boxes_restored: int
    images_removed: int
    #: Images the version references whose FILE is gone from disk — reported so a
    #: partial restore is never silently presented as a complete one.
    missing_files: list[str]
    #: The safety version taken of the pre-restore state.
    backup_version: int


def restore_version(db: Session, project_id: int, version) -> RestoreResult:
    """Rewind the live dataset to `version`, after backing up the current state."""
    from app.models import Annotation, Category, Image
    from app.enums import CLASS_COLORS
    from app.services import storage

    snapshot = load_snapshot(version)

    # 1. Back up what's about to be replaced. Do this FIRST — if anything below
    #    fails, the user still has a save point for the state they were in.
    backup = save_version(
        db, project_id, note=f"Auto-saved before restoring v{version.version}"
    )

    # 2. Classes, matched BY NAME. Ids drift (a class deleted and re-added gets a
    #    new one), but the name is what the boxes mean. Missing classes are
    #    recreated so a restore can bring back a deleted one.
    existing = {
        c.name: c
        for c in db.scalars(
            select(Category).where(Category.project_id == project_id)
        ).all()
    }
    for sc in snapshot.categories:
        if sc.name not in existing:
            category = Category(
                project_id=project_id,
                name=sc.name,
                color=sc.color or CLASS_COLORS[len(existing) % len(CLASS_COLORS)],
            )
            db.add(category)
            db.flush()
            existing[sc.name] = category
    # snapshot category id -> live category id
    cat_by_snapshot_id = {
        sc.id: existing[sc.name].id for sc in snapshot.categories if sc.name in existing
    }

    # 3. Images. Keyed by the stored uuid filename, which is stable across
    #    delete/restore because the FILE is never removed.
    current = {
        img.filename: img
        for img in db.scalars(
            select(Image).where(Image.project_id == project_id)
        ).all()
    }
    wanted = {si.filename: si for si in snapshot.images}

    # Anything added since the version goes away. Only the ROW — its bytes stay
    # on disk, and the backup version above still references it, so this is
    # reversible.
    images_removed = 0
    for filename, img in current.items():
        if filename not in wanted:
            db.delete(img)
            images_removed += 1

    project_dir = storage.project_dir(project_id)
    missing_files: list[str] = []
    images_restored = 0
    boxes_restored = 0

    for filename, si in wanted.items():
        img = current.get(filename)
        if img is None:
            # Recreate a row for an image that was deleted after this version.
            if not (project_dir / filename).exists():
                missing_files.append(si.original_filename)
                continue
            img = Image(
                project_id=project_id,
                filename=si.filename,
                original_filename=si.original_filename,
                width=si.width,
                height=si.height,
                size_bytes=si.size_bytes,
                split=si.split,
            )
            db.add(img)
            db.flush()
        else:
            img.split = si.split

        # Replace this image's ACCEPTED boxes with the version's. Proposals are
        # left in place — see the module docstring.
        for stale in db.scalars(
            select(Annotation).where(
                Annotation.image_id == img.id, Annotation.proposed.is_(False)
            )
        ).all():
            db.delete(stale)

        for sa in si.annotations:
            category_id = cat_by_snapshot_id.get(sa.category_id)
            if category_id is None:
                continue  # class vanished from the snapshot's own list; drop
            db.add(
                Annotation(
                    image_id=img.id,
                    category_id=category_id,
                    x=sa.x,
                    y=sa.y,
                    width=sa.width,
                    height=sa.height,
                    confidence=sa.confidence,
                    source=sa.source,
                    reviewed=True,
                    proposed=False,
                )
            )
            boxes_restored += 1
        images_restored += 1

    db.commit()
    logger.info(
        "Restored project %s to dataset v%s (%d images, %d boxes)",
        project_id,
        version.version,
        images_restored,
        boxes_restored,
    )
    return RestoreResult(
        images_restored=images_restored,
        boxes_restored=boxes_restored,
        images_removed=images_removed,
        missing_files=missing_files,
        backup_version=backup.version,
    )


def has_any_version(db: Session, project_id: int) -> bool:
    """Whether the project has ever been saved — the gate on training."""
    from app.models import DatasetVersion

    return (
        db.scalar(
            select(func.count(DatasetVersion.id)).where(
                DatasetVersion.project_id == project_id
            )
        )
        or 0
    ) > 0
