"""
One-off backfill: compute Image.content_hash for rows that predate the column.

WHY THIS IS NEEDED
------------------
`_add_missing_columns()` in database.py ALTERs new nullable columns in at
startup, which is enough to make the app run — but it cannot fill them. Rows
uploaded before `content_hash` existed have NULL, and duplicate detection on
upload only compares against non-NULL hashes. So without this, re-uploading a
folder into an OLD project still creates duplicates: the new rows get hashes and
dedupe against each other, but nothing recognises them as copies of what was
already there.

One of several hand-written backfills in this project (see the others in this
directory). Each one is an argument for replacing `_add_missing_columns` with
Alembic, which can add a column but cannot fill it.

USAGE
-----
    ./backend/venv/Scripts/python.exe scripts/backfill_content_hash.py
    ./backend/venv/Scripts/python.exe scripts/backfill_content_hash.py --dry-run

Safe to re-run: it only touches rows where content_hash IS NULL. Stop it at any
point and run it again — each row is independent, and progress is committed in
batches rather than all at the end.

A row whose FILE is missing is left NULL and reported. That's a real condition
(an image deleted before versions existed, or storage moved) and inventing a
hash for bytes we don't have would be worse than leaving the gap visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Image  # noqa: E402
from app.services import storage  # noqa: E402
from sqlalchemy import select  # noqa: E402

#: Commit every N rows. Small enough that an interrupted run keeps most of its
#: work, large enough not to fsync per image.
BATCH = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing anything",
    )
    args = parser.parse_args()

    # `content_hash` is ALTERed in by _add_missing_columns(), which normally
    # only runs when the app starts. Calling init_db() here means this script
    # works on a database whose server hasn't been restarted since the column
    # was added — otherwise the first thing it does is fail with
    # "no such column: images.content_hash", which looks like a broken script
    # rather than a startup that hasn't happened yet.
    init_db()

    db = SessionLocal()
    try:
        rows = list(
            db.scalars(
                select(Image).where(Image.content_hash.is_(None)).order_by(Image.id)
            ).all()
        )
        if not rows:
            print("Nothing to do — every image already has a content hash.")
            return 0

        print(f"{len(rows)} image(s) without a content hash.")
        hashed = 0
        missing: list[str] = []

        for i, image in enumerate(rows, start=1):
            path = storage.project_dir(image.project_id) / image.filename
            try:
                digest = storage.content_digest(path.read_bytes())
            except OSError:
                missing.append(f"project {image.project_id}: {image.original_filename}")
                continue

            if not args.dry_run:
                image.content_hash = digest
            hashed += 1

            if not args.dry_run and i % BATCH == 0:
                db.commit()
                print(f"  ...{i}/{len(rows)}")

        if not args.dry_run:
            db.commit()

        verb = "would hash" if args.dry_run else "hashed"
        print(f"\n{verb} {hashed} image(s).")
        if missing:
            print(f"{len(missing)} row(s) left NULL — file not found on disk:")
            for m in missing[:20]:
                print(f"  {m}")
            if len(missing) > 20:
                print(f"  ...and {len(missing) - 20} more")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
