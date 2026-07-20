"""
Dataset import — recognise what the user actually uploaded.

Someone dropping a zip could reasonably be handing us any of these:

  1. Loose images                     -> just images, no labels
  2. Images + one COCO json           -> flat annotated dataset
  3. train/valid/test subfolders,     -> Roboflow's COCO export: THREE separate
     each with its own COCO json         COCO files, each scoped to its folder
  4. Any of the above nested one      -> zips usually contain a top folder
     level down inside a wrapper dir

This module figures out which, without asking. The alternative — a dropdown
where you declare your format — is the thing that makes tools annoying.

THE RULE THAT MATTERS FOR ROBOFLOW EXPORTS
------------------------------------------
Each split's COCO file has its OWN image_id numbering, starting at 1. train's
image id 1 and valid's image id 1 are different pictures. So annotations MUST be
resolved against the COCO file from the same folder — never globally. Getting
this wrong doesn't error: it silently attaches train's boxes to valid's images,
and you find out when mAP is inexplicably garbage.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from app.services import storage

logger = logging.getLogger(__name__)

# Filenames that hold COCO annotations. Roboflow writes the first; the official
# COCO release uses instances_*.json; everyone else picks something reasonable.
COCO_FILENAMES = (
    "_annotations.coco.json",
    "annotations.coco.json",
    "annotations.json",
    "instances.json",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class CocoDoc:
    """A parsed COCO file, indexed for lookup."""

    #: COCO category_id -> class name
    categories: dict[int, str]
    #: lowercase file_name -> {"id", "width", "height"}
    images_by_name: dict[str, dict]
    #: COCO image_id -> list of raw annotation dicts
    anns_by_image: dict[int, list[dict]]
    #: How many boxes were computed from a segmentation polygon because the
    #: annotation carried no usable bbox. Surfaced in the import notes — the
    #: user should know their segmentation dataset was read as detection.
    derived_boxes: int = 0

    @property
    def class_names(self) -> list[str]:
        return list(self.categories.values())


@dataclass
class SplitGroup:
    """One split's worth of files, plus the COCO doc that describes them."""

    split: str
    #: relative path inside the archive -> extracted path on disk
    image_files: dict[str, Path] = field(default_factory=dict)
    coco: CocoDoc | None = None


@dataclass
class ImportPlan:
    """What we found, ready to execute (or to report if nothing useful)."""

    groups: list[SplitGroup] = field(default_factory=list)
    #: Every class name mentioned by any COCO file, in first-seen order.
    class_names: list[str] = field(default_factory=list)
    #: True when the archive used train/valid/test folders.
    has_split_folders: bool = False
    #: Human-readable notes surfaced in the upload result.
    notes: list[str] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(len(g.image_files) for g in self.groups)

    @property
    def has_annotations(self) -> bool:
        return any(g.coco is not None for g in self.groups)

    @property
    def splits_present(self) -> set[str]:
        return {g.split for g in self.groups if g.image_files}


def parse_coco(path: Path) -> CocoDoc | None:
    """Parse a COCO json into lookup tables. Returns None if it isn't COCO.

    Deliberately permissive about what counts as COCO: `images` and
    `annotations` are enough. Real files in the wild routinely omit `info`,
    `licenses`, and sometimes even `categories` (when every box shares a class).
    Rejecting those would mean rejecting datasets that every other tool accepts.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        logger.warning("Not valid JSON: %s (%s)", path, exc)
        return None

    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        return None

    categories = {
        int(c["id"]): str(c["name"])
        for c in data.get("categories", [])
        if "id" in c and "name" in c
    }

    images_by_name: dict[str, dict] = {}
    for img in data["images"]:
        name = img.get("file_name")
        if not name:
            continue
        # Some exporters write "images/foo.jpg" or even an absolute path in
        # file_name. We only ever match on the basename, lowercased — Windows
        # is case-insensitive and Linux isn't, and a dataset shouldn't stop
        # importing because of that.
        key = Path(str(name)).name.lower()
        images_by_name[key] = {
            "id": int(img["id"]),
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
        }

    anns_by_image: dict[int, list[dict]] = {}
    derived_boxes = 0
    for ann in data["annotations"]:
        if "image_id" not in ann:
            continue
        if not _valid_bbox(ann.get("bbox")):
            # No usable bbox. Before dropping the annotation, try to derive one
            # from its segmentation — see _bbox_from_segmentation.
            box = _bbox_from_segmentation(ann.get("segmentation"))
            if box is None:
                continue
            ann = {**ann, "bbox": box}
            derived_boxes += 1
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    return CocoDoc(
        categories=categories,
        images_by_name=images_by_name,
        anns_by_image=anns_by_image,
        derived_boxes=derived_boxes,
    )


def _valid_bbox(bbox) -> bool:
    """A COCO bbox we can actually use: four numbers with positive extent."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        _, _, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0


def _bbox_from_segmentation(segmentation) -> list[float] | None:
    """The tightest box around a segmentation polygon, or None.

    WHY THIS EXISTS: this app is object detection only, so it reads `bbox` and
    ignores `segmentation`. That's fine for a dataset carrying both — but
    instance-segmentation exports frequently omit `bbox` entirely, since the
    polygon is the label and the box is derivable. Those datasets used to import
    as images with ZERO boxes, silently: no error, no warning, just a dataset
    that looked like it had lost its annotations.

    A polygon's bounding box is exactly what a detector wants from it, and
    computing it is the same thing COCO tooling does. So a segmentation dataset
    becomes a usable detection dataset instead of an empty one.

    RLE masks (`{"counts": ..., "size": ...}`, used for crowd regions) are NOT
    handled: decoding them needs pycocotools, and they're a small minority. They
    return None and the annotation is skipped, as before.
    """
    # COCO polygons: [[x1, y1, x2, y2, ...], ...] — one flat list per part.
    if not isinstance(segmentation, list) or not segmentation:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for part in segmentation:
        if not isinstance(part, (list, tuple)) or len(part) < 6:
            # Fewer than 3 points isn't a polygon.
            continue
        try:
            coords = [float(v) for v in part]
        except (TypeError, ValueError):
            continue
        xs.extend(coords[0::2])
        ys.extend(coords[1::2])

    if not xs or not ys:
        return None

    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    if w <= 0 or h <= 0:
        return None
    return [x, y, w, h]


def _strip_wrapper(root: Path) -> Path:
    """Descend through single-directory wrappers.

    Zips almost always contain one top-level folder ("my-dataset-v3/"), and
    Roboflow's are no exception. Without this, every path check below would have
    to know about a directory name we can't predict. Loops in case of
    "export/dataset/train/...".
    """
    current = root
    for _ in range(4):  # bounded: a pathological zip shouldn't spin us
        entries = [p for p in current.iterdir() if not p.name.startswith("__MACOSX")]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
        else:
            break
    return current


def _is_junk(path: Path) -> bool:
    """Skip macOS metadata and hidden files.

    A zip made on a Mac contains __MACOSX/._foo.jpg for every real foo.jpg.
    Those are AppleDouble resource forks, not images — importing them produces a
    pile of corrupt-file errors that look like the user's fault.
    """
    parts = [p.lower() for p in path.parts]
    return (
        any(p == "__macosx" for p in parts)
        or path.name.startswith("._")
        or path.name.startswith(".")
        or path.name.lower() == "thumbs.db"
    )


def analyse(root: Path) -> ImportPlan:
    """Inspect an extracted archive and work out what it is."""
    from app.models.image import Split

    root = _strip_wrapper(root)
    plan = ImportPlan()

    # --- Case 3/4: split folders -------------------------------------------
    split_dirs: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or _is_junk(child):
            continue
        split = Split.from_folder(child.name)
        if split:
            split_dirs[split] = child

    if split_dirs:
        plan.has_split_folders = True
        for split, directory in split_dirs.items():
            group = SplitGroup(split=split)

            # Images may sit directly in the split folder (Roboflow COCO) or
            # under an images/ subdir (Roboflow YOLO). rglob covers both.
            for f in sorted(directory.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES and not _is_junk(f):
                    group.image_files[f.name] = f

            # Find THIS split's COCO file. Scoped to the folder — see the
            # module docstring for why that is the whole ballgame.
            for candidate in COCO_FILENAMES:
                found = next(directory.rglob(candidate), None)
                if found:
                    group.coco = parse_coco(found)
                    break
            if group.coco is None:
                # Fall back to any .json in the folder that parses as COCO.
                for f in sorted(directory.rglob("*.json")):
                    if _is_junk(f):
                        continue
                    doc = parse_coco(f)
                    if doc:
                        group.coco = doc
                        break

            if group.image_files:
                plan.groups.append(group)
                n = len(group.image_files)
                labelled = "with annotations" if group.coco else "no annotations found"
                plan.notes.append(f"{split}: {n} image(s), {labelled}")
    else:
        # --- Case 1/2: flat ------------------------------------------------
        group = SplitGroup(split=Split.TRAIN)
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES and not _is_junk(f):
                group.image_files[f.name] = f

        for candidate in COCO_FILENAMES:
            found = next(root.rglob(candidate), None)
            if found:
                group.coco = parse_coco(found)
                break
        if group.coco is None:
            for f in sorted(root.rglob("*.json")):
                if _is_junk(f):
                    continue
                doc = parse_coco(f)
                if doc:
                    group.coco = doc
                    break

        if group.image_files:
            plan.groups.append(group)
            if group.coco:
                plan.notes.append(
                    f"COCO annotations detected for {len(group.image_files)} image(s)"
                )

    # Union of class names across every split, order preserved. Roboflow writes
    # identical categories in all three files, but we can't rely on that — and
    # dict.fromkeys gives dedupe + stable order in one step.
    names: list[str] = []
    for group in plan.groups:
        if group.coco:
            names.extend(group.coco.class_names)
    plan.class_names = list(dict.fromkeys(names))

    # Say so when boxes came from polygons. The dataset imported fine, but the
    # user handed over a segmentation dataset and got a detection one — that's a
    # conversion, and conversions should be visible.
    derived = sum(g.coco.derived_boxes for g in plan.groups if g.coco)
    if derived:
        plan.notes.append(
            f"{derived} box(es) derived from segmentation outlines — this project "
            "is object detection, so each polygon became its bounding box."
        )

    return plan


@dataclass
class ImportResult:
    images_added: int = 0
    annotations_added: int = 0
    classes_created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)
    has_split_folders: bool = False
    notes: list[str] = field(default_factory=list)


def execute(db, project_id: int, plan: ImportPlan) -> ImportResult:
    """Turn an ImportPlan into images, classes and annotations.

    Takes `db` as a parameter rather than importing a session: this is a service,
    and it must be callable from a route, a background job, or a test script.
    """
    from app.models import Annotation, Category
    from app.models.image import Image
    from app.enums import CLASS_COLORS

    result = ImportResult(has_split_folders=plan.has_split_folders, notes=list(plan.notes))

    # --- classes ------------------------------------------------------------
    # Merge by NAME against what the project already has. Re-importing a dataset
    # must not create a second "car" — the unique constraint would reject it
    # anyway, but silently reusing the existing class is the useful behaviour.
    existing = {
        c.name: c
        for c in db.scalars(
            select(Category).where(Category.project_id == project_id)
        ).all()
    }
    for name in plan.class_names:
        if name in existing:
            continue
        category = Category(
            project_id=project_id,
            name=name,
            color=CLASS_COLORS[len(existing) % len(CLASS_COLORS)],
        )
        db.add(category)
        db.flush()  # need the id before annotations reference it
        existing[name] = category
        result.classes_created.append(name)

    # --- images + annotations ----------------------------------------------
    for group in plan.groups:
        for name, path in group.image_files.items():
            try:
                content = path.read_bytes()
                saved = storage.save_image(project_id, content, name)
            except (storage.ImageRejected, OSError) as exc:
                result.skipped.append(f"{name}: {exc}")
                continue

            image = Image(
                project_id=project_id,
                filename=saved.filename,
                original_filename=saved.original_filename,
                width=saved.width,
                height=saved.height,
                size_bytes=saved.size_bytes,
                split=group.split,
            )
            db.add(image)
            db.flush()
            result.images_added += 1
            result.splits[group.split] = result.splits.get(group.split, 0) + 1

            if group.coco is None:
                continue

            # Resolve against THIS group's COCO doc only. See module docstring:
            # each split numbers its image_ids from 1, so a global lookup would
            # cheerfully attach train's boxes to valid's images.
            meta = group.coco.images_by_name.get(name.lower())
            if meta is None:
                continue

            for raw in group.coco.anns_by_image.get(meta["id"], []):
                bbox = raw.get("bbox") or []
                if len(bbox) != 4:
                    continue
                x, y, w, h = (float(v) for v in bbox)
                if w <= 0 or h <= 0:
                    continue

                class_name = group.coco.categories.get(int(raw.get("category_id", -1)))
                category = existing.get(class_name) if class_name else None
                if category is None:
                    continue

                # Clamp against the REAL image dimensions read off disk, not the
                # width/height COCO claims. They disagree more often than you'd
                # expect — resized exports with stale metadata are common — and
                # the file is the authority.
                x = max(0.0, min(x, saved.width))
                y = max(0.0, min(y, saved.height))
                w = min(w, saved.width - x)
                h = min(h, saved.height - y)
                if w <= 0 or h <= 0:
                    continue

                db.add(
                    Annotation(
                        image_id=image.id,
                        category_id=category.id,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=None,  # human-authored ground truth
                        # A third provenance value alongside auto/manual. It is
                        # not "auto" — which matters, because the annotation job
                        # deletes source="auto" boxes on re-run and must never
                        # touch imported ground truth.
                        source="imported",
                        reviewed=True,
                    )
                )
                result.annotations_added += 1

    db.commit()
    return result


def extract_archive(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing path traversal and zip bombs.

    ZIP SLIP: a zip entry named "../../../etc/passwd" makes a naive
    extractall() write outside the destination. Python's extractall() has
    guarded against this since 3.6.2, but we check anyway — this is the one
    place untrusted input becomes filesystem paths, and the check is three
    lines.

    ZIP BOMB: unlike the in-memory path this replaced, extractall() writes to
    DISK, so a small archive declaring terabytes of uncompressed content fills
    the volume before anything downstream gets a chance to reject it. The
    header's declared sizes are checked first — they're attacker-controlled and
    can lie, so the running total is also checked as members are written, which
    is the figure that can't be faked.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) > storage.MAX_ZIP_MEMBERS:
            raise ValueError(
                f"Archive contains more than {storage.MAX_ZIP_MEMBERS} entries."
            )
        declared = sum(m.file_size for m in members)
        if declared > storage.MAX_ZIP_BYTES:
            raise ValueError(
                f"Archive expands to more than "
                f"{storage.MAX_ZIP_BYTES // (1024**3)} GB."
            )

        for member in members:
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe path in archive: {member.filename!r}")

        written = 0
        for member in members:
            zf.extract(member, dest)
            if member.is_dir():
                continue
            # The real size on disk, not the header's claim.
            written += (dest / member.filename).stat().st_size
            if written > storage.MAX_ZIP_BYTES:
                raise ValueError(
                    f"Archive expands to more than "
                    f"{storage.MAX_ZIP_BYTES // (1024**3)} GB."
                )
