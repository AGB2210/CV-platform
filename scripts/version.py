"""
Read, check, or bump the project version — the only way it should change.

    python scripts/version.py                 # what is it?
    python scripts/version.py --check         # do all the places agree?
    python scripts/version.py --set 0.2.0     # set it everywhere
    python scripts/version.py --bump minor    # 0.1.0 -> 0.2.0
    python scripts/version.py --bump patch    # 0.1.0 -> 0.1.1

WHY THIS EXISTS
---------------
The version used to live in three files that didn't read each other, and they
disagreed: package.json said 0.0.0, main.py said 0.1.0, a README badge said
v0.1. That is what happens to any number declared in more than one place.

`VERSION` at the repo root is now the source of truth. Python reads it at
import (app/version.py); this script is what keeps the copies that CANNOT read
it — `package.json`, which npm requires to hold a literal — in step, and
`--check` is what CI runs so they can never silently drift again.

THE SCHEME
----------
Semantic versioning, and the numbers mean something:

    MAJOR   incompatible change to how the app is used or its data is stored.
            Stays 0 until the phase plan in README.md is complete.
    MINOR   a phase lands, or a feature arrives. 0.1.0 is phases 0-4;
            phase 5 (evaluation + inference) will be 0.2.0.
    PATCH   fixes and internal work, nothing new to use.

A release tag is this number with a `v`: VERSION 0.2.0 -> tag v0.2.0. The
release workflow refuses to publish if a tag and this file disagree, so the
number on a download always matches the code inside it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
PACKAGE_JSON = ROOT / "frontend" / "package.json"

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read() -> str:
    if not VERSION_FILE.exists():
        sys.exit(f"No VERSION file at {VERSION_FILE}")
    value = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not SEMVER.match(value):
        sys.exit(f"VERSION holds {value!r}, which is not MAJOR.MINOR.PATCH")
    return value


def write(version: str) -> None:
    VERSION_FILE.write_text(version + "\n", encoding="utf-8")

    # package.json must hold a literal — npm can't read our VERSION file — so
    # it's written here rather than left to drift.
    data = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))
    data["version"] = version
    PACKAGE_JSON.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def check() -> int:
    """Report every place that disagrees. Exit non-zero if any does."""
    version = read()
    problems: list[str] = []

    package_version = json.loads(PACKAGE_JSON.read_text(encoding="utf-8")).get("version")
    if package_version != version:
        problems.append(
            f"frontend/package.json says {package_version!r}, VERSION says {version!r}"
        )

    # The app must report the same thing at runtime — that's the copy users see.
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from app.version import __version__ as app_version

        if app_version != version:
            problems.append(f"app.version says {app_version!r}, VERSION says {version!r}")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"could not import app.version: {exc}")

    print(f"VERSION: {version}")
    if problems:
        print("\nOut of step:")
        for p in problems:
            print(f"  - {p}")
        print("\nRun: python scripts/version.py --set " + version)
        return 1
    print("  frontend/package.json  ok")
    print("  app.version            ok")
    return 0


def bump(current: str, part: str) -> str:
    major, minor, patch = (int(x) for x in SEMVER.match(current).groups())
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify every place agrees")
    group.add_argument("--set", dest="value", help="set an exact version")
    group.add_argument("--bump", choices=("major", "minor", "patch"))
    args = parser.parse_args()

    if args.check:
        return check()

    if args.value or args.bump:
        current = read()
        new = args.value if args.value else bump(current, args.bump)
        if not SEMVER.match(new):
            sys.exit(f"{new!r} is not MAJOR.MINOR.PATCH")
        write(new)
        print(f"{current} -> {new}")
        print("\nNext:")
        print("  git commit -am 'Release " + new + "'")
        print(f"  git tag -a v{new} -m 'Release {new}'")
        print(f"  git push origin main && git push origin v{new}")
        return 0

    print(read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
