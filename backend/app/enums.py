"""
Shared enumerations.

Kept in one module so the API layer, the ORM, and (later) the ML code all agree
on the same vocabulary without importing each other.
"""

from enum import Enum


class TaskType(str, Enum):
    """What a project is trying to predict.

    Inheriting from `str` as well as `Enum` matters: it makes members behave as
    plain strings for JSON serialisation and SQL binding, so `TaskType.OBJECT_
    DETECTION` round-trips as "object_detection" with no manual conversion.

    IMPORTANT — why the DB column is a plain String, not sa.Enum:
    SQLite has no native ENUM type. SQLAlchemy emulates one with a VARCHAR plus
    a CHECK constraint, and SQLite cannot ALTER a CHECK constraint — changing it
    requires rebuilding the whole table and copying the data. So adding
    SEGMENTATION later would mean a genuine migration.

    Storing a plain String and validating at the Pydantic layer instead means
    adding a task type is a one-line code change with ZERO schema migration.
    The DB stays permissive; the API stays strict. That's the "extensible
    without a rewrite" requirement, discharged.
    """

    OBJECT_DETECTION = "object_detection"
    # SEGMENTATION = "segmentation"   # later phase — uncomment, no migration needed


# The palette assigned to classes as they're created.
#
# Chosen for distinguishability when drawn as overlapping bounding boxes on
# arbitrary photos: reasonably saturated (must stay visible over busy imagery),
# but spread around the hue wheel so adjacent classes never look alike. This is
# the one place in the app where saturated colour is correct — it's data
# encoding, not decoration, and it deliberately does not use the muted UI accent.
CLASS_COLORS: list[str] = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#ea580c",  # orange
    "#9333ea",  # purple
    "#0891b2",  # cyan
    "#ca8a04",  # yellow
    "#db2777",  # pink
    "#4d7c0f",  # olive
    "#4f46e5",  # indigo
    "#b91c1c",  # dark red
    "#0d9488",  # teal
]
