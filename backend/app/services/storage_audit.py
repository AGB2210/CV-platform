"""
What a project is holding on disk, and what can safely be let go.

Three different "this isn't needed" questions get asked about image files, and
they are NOT the same question. Conflating them is how a cleanup feature
deletes something a restore was relying on.

    UNSAVED    a live image that appears in no saved version.
               Real, visible, annotatable dataset content — just not captured
               in a save point yet. Deleting it is a USER decision, never
               automatic: the upload -> annotate -> save workflow spends all its
               time in this state, so "unsaved" does not mean "unwanted".

    ORPHANED   a FILE on disk referenced by neither a live image row nor any
               version snapshot. Nothing in the app can reach it and no restore
               can ever need it. Pure waste, safe to reclaim.

    RETAINED   a file with no live row, but referenced by a version snapshot.
               Looks orphaned and absolutely is not — this is the bytes-stay-
               on-disk rule that makes restore possible (services/storage.py).
               Deleting these turns every version referencing them into a
               promise the app cannot keep.

The third category is the reason this module reads the version snapshots rather
than just diffing the directory against the images table. That diff is the
obvious implementation and it is wrong.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.services import storage
from app.services.dataset_snapshot import read_snapshot

logger = logging.getLogger(__name__)


@dataclass
class StorageReport:
    """What this project is holding, and what could be freed."""

    total_images: int = 0
    #: Live images in no saved version — the user's call, never ours.
    unsaved_images: int = 0
    #: Files reachable by nothing. Safe to delete.
    orphan_files: int = 0
    orphan_bytes: int = 0
    #: Files kept only because a version references them. NOT deletable.
    retained_files: int = 0
    retained_bytes: int = 0
    #: Version snapshots that could not be read, so their filenames are unknown.
    #: Non-empty means the orphan list is deliberately incomplete — see
    #: `_referenced_by_versions`.
    unreadable_versions: list[str] = field(default_factory=list)


def _referenced_by_versions(db: Session, project_id: int) -> tuple[set[str], list[str]]:
    """Every stored filename any saved version depends on.

    A snapshot that cannot be read is reported rather than treated as empty.
    Treating it as empty would mark its files orphaned and delete exactly the
    data the version exists to protect — the failure mode this whole module is
    arranged to avoid.
    """
    from app.models import DatasetVersion

    referenced: set[str] = set()
    unreadable: list[str] = []

    for version in db.scalars(
        select(DatasetVersion).where(DatasetVersion.project_id == project_id)
    ).all():
        try:
            snapshot = read_snapshot(Path(version.snapshot_path))
        except Exception as exc:  # noqa: BLE001 — a bad file must not delete data
            logger.warning("Cannot read snapshot for v%s: %s", version.version, exc)
            unreadable.append(f"v{version.version}")
            continue
        referenced.update(image.filename for image in snapshot.images)

    return referenced, unreadable


def unsaved_image_ids(db: Session, project_id: int) -> list[int]:
    """Live images that no saved version contains."""
    from app.models import Image

    referenced, _ = _referenced_by_versions(db, project_id)
    return [
        image.id
        for image in db.scalars(
            select(Image).where(Image.project_id == project_id)
        ).all()
        if image.filename not in referenced
    ]


def audit(db: Session, project_id: int) -> StorageReport:
    """Measure the project's disk usage without changing anything."""
    from app.models import Image

    live = {
        image.filename
        for image in db.scalars(
            select(Image).where(Image.project_id == project_id)
        ).all()
    }
    referenced, unreadable = _referenced_by_versions(db, project_id)

    report = StorageReport(
        total_images=len(live),
        unsaved_images=len(live - referenced),
        unreadable_versions=unreadable,
    )

    project_dir = storage.project_dir(project_id)
    for path in project_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in live:
            continue  # in use right now
        size = path.stat().st_size
        if path.name in referenced:
            # No live row, but a version needs it. This is the retention rule
            # working as designed, not waste.
            report.retained_files += 1
            report.retained_bytes += size
        else:
            report.orphan_files += 1
            report.orphan_bytes += size

    return report


def reclaim_orphans(db: Session, project_id: int) -> tuple[int, int]:
    """Delete files nothing can reach. Returns (files_removed, bytes_freed).

    Refuses to do anything if any version snapshot is unreadable: without the
    full picture of what versions depend on, "unreferenced" is a guess, and
    guessing here deletes irreplaceable data.
    """
    from app.models import Image

    live = {
        image.filename
        for image in db.scalars(
            select(Image).where(Image.project_id == project_id)
        ).all()
    }
    referenced, unreadable = _referenced_by_versions(db, project_id)
    if unreadable:
        raise ValueError(
            f"Cannot reclaim safely: {len(unreadable)} version snapshot(s) "
            f"({', '.join(unreadable)}) could not be read, so the files they "
            f"depend on are unknown. Fix or delete those versions first."
        )

    keep = live | referenced
    removed = 0
    freed = 0
    for path in storage.project_dir(project_id).iterdir():
        if not path.is_file() or path.name in keep:
            continue
        size = path.stat().st_size
        try:
            path.unlink()
        except OSError:  # locked, or vanished between listing and unlinking
            continue
        removed += 1
        freed += size

    logger.info(
        "Reclaimed %d orphaned file(s), %d bytes, from project %s",
        removed,
        freed,
        project_id,
    )
    return removed, freed
