"""
One-off backfill: rewrite absolute snapshot/checkpoint paths as relative ones.

WHY
---
`DatasetVersion.snapshot_path` and `TrainingJob.checkpoint_path` used to be
stored absolute:

    C:/Users/someone/Downloads/cv app/storage/versions/2/v1.json

which bakes one machine's directory layout into the data. Rename the folder,
move the project to another drive, or clone it elsewhere, and every saved
version and every trained model becomes unreachable — while the rows still look
perfectly healthy.

New rows are written relative (see app/config.py::to_storage_path). This
converts the ones written before that.

NOT URGENT. `from_storage_path` accepts both shapes, so the app works whether or
not this has run — an absolute path is used as-is. Running it is what makes the
project directory MOVABLE.

USAGE
-----
    ./backend/venv/Scripts/python.exe scripts/backfill_relative_paths.py
    ./backend/venv/Scripts/python.exe scripts/backfill_relative_paths.py --dry-run

Safe to re-run: a path already relative is left alone. A path pointing OUTSIDE
storage/ is also left alone and reported — that isn't something this app
creates, and rewriting it would lose information rather than fix anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings, to_storage_path  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import DatasetVersion, TrainingJob  # noqa: E402
from sqlalchemy import select  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, don't write")
    args = parser.parse_args()

    init_db()
    root = settings.STORAGE_DIR.resolve()
    print(f"storage root: {root}\n")

    db = SessionLocal()
    try:
        changed = 0
        outside: list[str] = []

        targets = (
            (DatasetVersion, "snapshot_path", "dataset version"),
            (TrainingJob, "checkpoint_path", "training run"),
        )

        for model, field, label in targets:
            for row in db.scalars(select(model)).all():
                current = getattr(row, field)
                if not current:
                    continue
                if not Path(current).is_absolute():
                    continue  # already relative

                relative = to_storage_path(current)
                if Path(relative).is_absolute():
                    # to_storage_path gives back an absolute path only when it
                    # sits outside storage/ — nothing this app writes does.
                    outside.append(f"{label} {row.id}: {current}")
                    continue

                print(f"  {label} {row.id}: {current}")
                print(f"    -> {relative}")
                if not args.dry_run:
                    setattr(row, field, relative)
                changed += 1

        if not args.dry_run:
            db.commit()

        verb = "would rewrite" if args.dry_run else "rewrote"
        print(f"\n{verb} {changed} path(s).")
        if outside:
            print(f"\n{len(outside)} path(s) point outside storage/ and were left as-is:")
            for item in outside:
                print(f"  {item}")
        if not changed and not outside:
            print("Nothing to do — every path is already relative.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
