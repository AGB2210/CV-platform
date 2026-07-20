"""
Naming rules shared by dataset versions and model versions.

Both are lists the user picks from, so both need the same guarantee: no two
entries can read the same. The subtlety is that an unnamed version still HAS a
label — its number, "v3" — so uniqueness has to be checked against labels, not
just against the names people typed. Otherwise naming something "v3" while a
different version is already displaying as v3 produces two identical rows and a
genuinely ambiguous list.

Kept in one place because the dataset and training routes must not drift apart
on what counts as a duplicate.
"""

from __future__ import annotations


class DuplicateNameError(Exception):
    """The requested name collides with another version's label."""


def label_for(name: str | None, version: int) -> str:
    """How a version presents in a list: its name, else its number."""
    return name if name else f"v{version}"


def clean_name(raw: str | None) -> str | None:
    """Normalise user input. Blank (or whitespace) means "no name" — i.e. revert
    to the numeric label, which is how a rename is undone."""
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def ensure_unique(
    new_name: str | None,
    version: int,
    others: list[tuple[str | None, int]],
) -> None:
    """Raise if `new_name` would duplicate another entry's label.

    `others` is (name, version) for every OTHER version in the same scope — the
    project for dataset versions, the project AND trainer for model versions.

    Clearing the name is always allowed: the numeric label it falls back to is
    unique by construction.
    """
    if new_name is None:
        return

    # What another version currently displays as...
    taken = {label_for(n, v).casefold() for n, v in others}
    # ...plus every other version's NUMERIC label, even when it has a custom
    # name. "v2" always means version 2: without this, renaming v2 to "baseline"
    # would free up "v2" for version 5 to claim, and the number would stop
    # meaning anything.
    taken |= {f"v{v}".casefold() for _, v in others}

    if new_name.casefold() in taken:
        raise DuplicateNameError(
            f"“{new_name}” is already used by another version. Pick a different name."
        )
