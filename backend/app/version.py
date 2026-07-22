"""
The version, read from one place.

WHY A FILE AND NOT A LITERAL
----------------------------
This number once lived in three files that each declared their own value and
read none of the others, so they drifted the moment the second one was written
— which is what happens to every number declared in more than one place.

So `VERSION` at the repo root is the single source of truth. Python reads it
here, the release workflow reads it with `cat`, and CI FAILS if
`package.json` or a release tag disagrees with it. That last part is the point:
a convention nobody checks is a convention that has already drifted.

The value is `0.0.0-dev` until the first real release. This project is
unreleased; `1.0.0` will be the first stable, full-featured build. See
scripts/version.py.

WHY NOT DERIVE IT FROM GIT
--------------------------
`git describe` is the usual answer and it's wrong for this project: a release
artifact is an unpacked zip with no `.git` directory, so the app would have no
way to report its own version exactly where knowing it matters most.
"""

from __future__ import annotations

from pathlib import Path

#: Repo root — this file is backend/app/version.py, so up three.
_VERSION_FILE = Path(__file__).resolve().parent.parent.parent / "VERSION"


def _read() -> str:
    try:
        value = _VERSION_FILE.read_text(encoding="utf-8").strip()
        return value or "0.0.0+unknown"
    except OSError:
        # Never fatal. A missing VERSION file should degrade to an obviously
        # wrong string, not stop the app from starting.
        return "0.0.0+unknown"


__version__ = _read()
