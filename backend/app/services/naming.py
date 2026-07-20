"""
What counts as "the same name", everywhere in the app.

Three different things carry user-chosen names — projects, classes, and the two
kinds of version — and they had drifted into three different answers:

  - Projects accepted exact duplicates. Two "Street Scenes" in the list, nothing
    to tell them apart by.
  - Classes were guarded by a DB unique constraint, which in SQLite is
    CASE-SENSITIVE, so "car" and "Car" both existed. They then export as two
    separate classes and the model learns to split one concept in half.
  - Version labels already compared case-insensitively (version_naming.py).

The rule is now one thing: names are compared with surrounding whitespace
stripped and case folded. `casefold()` rather than `lower()` because it handles
the cases `lower()` misses (German "ß" vs "SS", Turkish dotless ı) — this costs
nothing and it's the function that actually means "compare these as text".

WHY THE CHECK IS IN THE APPLICATION, NOT THE DATABASE
-----------------------------------------------------
A case-insensitive DB guarantee needs a unique index with COLLATE NOCASE, and
the homegrown migration helper in database.py is add-COLUMN only — it cannot add
an index to an existing table. So the case-insensitive part is enforced here,
while the exact-duplicate part stays enforced by the existing unique constraint.

That means the case-variant check has a theoretical race: two simultaneous
requests could both see "no duplicate" before either commits. For a single-user
local tool that window isn't reachable, and the alternative is a real migration
tool — which is the same seam `_add_missing_columns` already documents itself as
waiting for. Worth knowing rather than worth pretending isn't there.
"""

from __future__ import annotations


def normalized(name: str | None) -> str:
    """The comparison key for a user-facing name."""
    return (name or "").strip().casefold()


def collides(name: str | None, existing: list[str | None]) -> bool:
    """Whether `name` would read as one of `existing`."""
    key = normalized(name)
    return bool(key) and key in {normalized(e) for e in existing}
