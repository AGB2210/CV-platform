"""
Dataset import — recognise what the user actually uploaded.

Someone dropping a zip — or picking a folder — could reasonably be handing us
any of these:

  1. Loose images                     -> just images, no labels
  2. Images + one COCO json           -> flat annotated dataset
  3. train/valid/test subfolders,     -> Roboflow's COCO export: THREE separate
     each with its own COCO json         COCO files, each scoped to its folder
  4. A YOLO export                    -> data.yaml + per-image .txt labels,
                                         usually under images/ and labels/
  5. Any of the above nested one      -> zips usually contain a top folder
     level down inside a wrapper dir

This module figures out which, without asking. The alternative — a dropdown
where you declare your format — is the thing that makes tools annoying.

COCO AND YOLO ARE STRUCTURALLY DIFFERENT, NOT JUST DIFFERENT SYNTAX
-------------------------------------------------------------------
COCO is one document per dataset holding absolute corner-based pixel boxes and
carrying class names inline. YOLO is one FILE per image holding normalised
centre-based boxes that name classes only by index, with the actual names in a
separate data.yaml. So they need genuinely different readers, and the YOLO one
has to convert coordinates and resolve indices before anything downstream sees
them — everything past `execute` speaks absolute pixels and class names.

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
class YoloDoc:
    """YOLO labels for one split: a class list plus per-image .txt files.

    The counterpart to CocoDoc. YOLO stores one label FILE per image rather
    than one document per dataset, and the class list lives separately (in
    data.yaml or classes.txt) because the label files only ever say "class 3".
    """

    #: Ordered class names. The INDEX is the identity here — YOLO label files
    #: reference classes by position, so this order is load-bearing in a way
    #: COCO's category ids are not.
    class_names: list[str]
    #: lowercase image filename -> its .txt label file
    label_files: dict[str, Path] = field(default_factory=dict)
    #: Boxes computed from a segmentation polygon (YOLO-seg labels).
    derived_boxes: int = 0


@dataclass
class SplitGroup:
    """One split's worth of files, plus the labels that describe them."""

    split: str
    #: relative path inside the archive -> extracted path on disk
    image_files: dict[str, Path] = field(default_factory=dict)
    coco: CocoDoc | None = None
    yolo: YoloDoc | None = None

    @property
    def has_labels(self) -> bool:
        return self.coco is not None or self.yolo is not None


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
        return any(g.has_labels for g in self.groups)

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


# --- YOLO -------------------------------------------------------------------
# YOLO is the other format people actually have. It differs from COCO in every
# structural way: one label FILE per image instead of one document per dataset,
# coordinates NORMALISED to 0-1 instead of absolute pixels, centre-based
# cx/cy/w/h instead of corner-based x/y/w/h, and class names held separately
# because label files only ever say "class 3".

#: Where the class list lives, in the order we should look.
YOLO_CLASS_FILENAMES = ("data.yaml", "data.yml", "classes.txt", "obj.names")


def parse_yolo_classes(path: Path) -> list[str]:
    """Read the class list from data.yaml / classes.txt. [] if unreadable.

    data.yaml holds `names:` as either a list (["car", "person"]) or, in newer
    ultralytics exports, a dict keyed by index ({0: "car", 1: "person"}). The
    dict form must be sorted BY KEY — dict order in a YAML file is not
    guaranteed to be index order, and getting it wrong silently renames every
    class to a different one.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception:  # noqa: BLE001 — a malformed yaml is not fatal
            return []
        if not isinstance(data, dict):
            return []
        names = data.get("names")
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names, key=lambda k: int(k))]
        if isinstance(names, list):
            return [str(n) for n in names]
        return []

    # classes.txt / obj.names: one name per line, position IS the class index.
    return [line.strip() for line in text.splitlines() if line.strip()]


def _yolo_label_dir_for(image_dir: Path, root: Path) -> Path | None:
    """Find the labels/ directory matching a directory of images.

    YOLO exports put labels in a sibling of images/ ("train/images" ->
    "train/labels"), which is the layout ultralytics writes and expects. Some
    tools instead drop the .txt next to the .jpg. Both are handled — the caller
    falls back to the image's own directory.
    """
    if image_dir.name.lower() == "images":
        sibling = image_dir.parent / "labels"
        if sibling.is_dir():
            return sibling
    # "labels" alongside the split folder, e.g. train/ + labels/train/
    candidate = root / "labels" / image_dir.name
    return candidate if candidate.is_dir() else None


def parse_yolo(
    image_files: dict[str, Path], root: Path, class_names: list[str]
) -> YoloDoc | None:
    """Pair images with their .txt label files. None if there are none."""
    label_files: dict[str, Path] = {}
    for name, image_path in image_files.items():
        stem_txt = Path(name).with_suffix(".txt").name
        label_dir = _yolo_label_dir_for(image_path.parent, root)
        for candidate in (
            (label_dir / stem_txt) if label_dir else None,
            image_path.with_suffix(".txt"),
        ):
            if candidate and candidate.is_file():
                label_files[name.lower()] = candidate
                break

    if not label_files:
        return None
    return YoloDoc(class_names=class_names, label_files=label_files)


def parse_yolo_label_file(path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    """Read one .txt into [(class_index, [x, y, w, h])] in ABSOLUTE pixels.

    Two label shapes share this file extension:
      "3 0.5 0.5 0.2 0.1"                  detection — cx cy w h, normalised
      "3 0.1 0.1 0.4 0.1 0.4 0.5 0.1 0.5"  segmentation — a normalised polygon

    Both are read. The polygon becomes its bounding box, for the same reason
    the COCO path derives one: this app is object detection, and a polygon's
    box is exactly what a detector wants from it.

    Coordinates are converted to absolute pixels here so everything downstream
    speaks one unit. Doing it later means carrying a "is this normalised?" flag
    through the import, which is precisely the kind of thing that gets lost.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []

    out: list[tuple[int, list[float]]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_index = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue

        if len(values) == 4:
            cx, cy, w, h = values
            # Centre-based -> corner-based, and normalised -> absolute.
            x = (cx - w / 2) * width
            y = (cy - h / 2) * height
            bw, bh = w * width, h * height
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = [v * width for v in values[0::2]]
            ys = [v * height for v in values[1::2]]
            x, y = min(xs), min(ys)
            bw, bh = max(xs) - x, max(ys) - y
        else:
            continue

        if bw <= 0 or bh <= 0:
            continue
        out.append((class_index, [x, y, bw, bh]))
    return out


def _find_yolo_classes(root: Path) -> list[str]:
    """The dataset's class list, from wherever it keeps it."""
    for candidate in YOLO_CLASS_FILENAMES:
        found = next(root.rglob(candidate), None)
        if found:
            names = parse_yolo_classes(found)
            if names:
                return names
    return []


def _strip_wrapper(root: Path) -> Path:
    """Descend through single-directory wrappers.

    Zips almost always contain one top-level folder ("my-dataset-v3/"), and
    Roboflow's are no exception. Without this, every path check below would have
    to know about a directory name we can't predict. Loops in case of
    "export/dataset/train/...".

    IT MUST NOT DESCEND INTO A SPLIT FOLDER. A dataset containing only `valid/`
    is a single-child directory, so this used to step inside it — and from there
    the layout looks flat, so every image was imported as TRAIN. Uploading a
    validation set silently turned it into a training set, which is the one
    mistake that quietly invalidates every metric a model reports afterwards.
    A lone `train/` was harmless by luck; `valid/` and `test/` were not.
    """
    from app.models.image import Split

    current = root
    for _ in range(4):  # bounded: a pathological zip shouldn't spin us
        entries = [p for p in current.iterdir() if not p.name.startswith("__MACOSX")]
        if len(entries) != 1 or not entries[0].is_dir():
            break
        if Split.from_folder(entries[0].name):
            break  # that's the dataset, not a wrapper around it
        current = entries[0]
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

    # Resolved once for the whole dataset: YOLO keeps its class list in a single
    # data.yaml at the root, not per split, and the INDEX in that list is what
    # every label file refers to.
    yolo_classes = _find_yolo_classes(root)

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

            # No COCO? Try YOLO. Checked second because a dataset carrying both
            # is almost always a COCO export with stray .txt files, and COCO
            # carries class NAMES with the boxes while YOLO needs a separate
            # class list that may be missing.
            if group.coco is None:
                group.yolo = parse_yolo(group.image_files, root, yolo_classes)

            if group.image_files:
                plan.groups.append(group)
                n = len(group.image_files)
                labelled = (
                    "with annotations" if group.has_labels else "no annotations found"
                )
                plan.notes.append(f"{split}: {n} image(s), {labelled}")
            elif group.has_labels:
                # Labels for images already in the project — see the
                # annotation-only path in execute().
                plan.groups.append(group)
                plan.notes.append(f"{split}: annotations only, no image files")
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

        if group.coco is None:
            group.yolo = parse_yolo(group.image_files, root, yolo_classes)

        if not group.image_files and group.has_labels:
            plan.groups.append(group)
            plan.notes.append(
                "Annotations only — matching them to images already in this project"
            )
        elif group.image_files:
            plan.groups.append(group)
            if group.coco:
                plan.notes.append(
                    f"COCO annotations detected for {len(group.image_files)} image(s)"
                )
            elif group.yolo:
                plan.notes.append(
                    f"YOLO annotations detected for {len(group.yolo.label_files)} "
                    f"of {len(group.image_files)} image(s)"
                )

    # Union of class names across every split, order preserved. Roboflow writes
    # identical categories in all three files, but we can't rely on that — and
    # dict.fromkeys gives dedupe + stable order in one step.
    #
    # For YOLO the order is not cosmetic: label files reference classes by
    # INDEX, so the list must stay in data.yaml's order or every box is
    # relabelled to a different class.
    names: list[str] = []
    for group in plan.groups:
        if group.coco:
            names.extend(group.coco.class_names)
        elif group.yolo:
            names.extend(group.yolo.class_names)
    plan.class_names = list(dict.fromkeys(names))

    # YOLO labels with no class list anywhere: the boxes reference indices we
    # cannot name. Synthesise names from the indices actually used rather than
    # dropping the annotations — "class_0" is recoverable by renaming, a
    # discarded box is not.
    if not plan.class_names and any(g.yolo for g in plan.groups):
        used: set[int] = set()
        for group in plan.groups:
            if not group.yolo:
                continue
            for label_path in group.yolo.label_files.values():
                for line in parse_yolo_label_file(label_path, 1, 1):
                    used.add(line[0])
        if used:
            plan.class_names = [f"class_{i}" for i in range(max(used) + 1)]
            plan.notes.append(
                "No data.yaml or classes.txt found, so classes are named by their "
                "index. Rename them on the Dataset page."
            )

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
    #: Images already in this project, byte-for-byte, and therefore not added
    #: again. Reported rather than silent — re-importing a folder used to
    #: double the dataset with no indication anything had happened.
    duplicates_skipped: int = 0
    #: Boxes written as PROPOSALS because their image was already here. They
    #: wait for Accept/Reject rather than overwriting existing work.
    proposals_created: int = 0
    #: How many existing images received those proposals.
    reannotated_images: int = 0


def _labelled_names(group: SplitGroup) -> list[str]:
    """Original filenames this group has labels for.

    Needed by the annotation-only path, where there are no image files to walk
    — the file itself is the only list of what it describes.
    """
    if group.coco is not None:
        return list(group.coco.images_by_name)
    if group.yolo is not None:
        return list(group.yolo.label_files)
    return []


def _boxes_for(
    group: SplitGroup, name: str, width: int, height: int, plan: ImportPlan, categories: dict
) -> list[tuple[int, float, float, float, float]]:
    """One image's boxes as (category_id, x, y, w, h) in absolute pixels.

    The single place COCO and YOLO converge. Both readers produce corner-based
    absolute pixels here, so everything downstream — clamping, storing,
    proposing — is written once and cannot drift between the two formats.
    """
    out: list[tuple[int, float, float, float, float]] = []

    if group.yolo is not None:
        label_path = group.yolo.label_files.get(name.lower())
        if label_path is None:
            return out
        for class_index, (x, y, w, h) in parse_yolo_label_file(label_path, width, height):
            # The index IS the class identity in YOLO. Out of range means the
            # label file and the class list disagree — drop rather than guess.
            if not (0 <= class_index < len(plan.class_names)):
                continue
            category = categories.get(plan.class_names[class_index])
            if category is not None:
                out.append((category.id, x, y, w, h))
        return out

    if group.coco is None:
        return out

    # Resolve against THIS group's COCO doc only. See the module docstring:
    # each split numbers its image_ids from 1, so a global lookup would
    # cheerfully attach train's boxes to valid's images.
    meta = group.coco.images_by_name.get(name.lower())
    if meta is None:
        return out

    for raw in group.coco.anns_by_image.get(meta["id"], []):
        bbox = raw.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            continue
        class_name = group.coco.categories.get(int(raw.get("category_id", -1)))
        category = categories.get(class_name) if class_name else None
        if category is not None:
            out.append((category.id, x, y, w, h))
    return out


def execute(
    db, project_id: int, plan: ImportPlan, import_id: str | None = None
) -> ImportResult:
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

    # Every image already in this project, keyed by content. Loaded once: a
    # 5,000-file import would otherwise issue 5,000 SELECTs to ask the same
    # question. Keyed by hash AND by original filename, because re-annotation
    # arrives both ways — with the images again (matched by bytes) or as a bare
    # annotation file (matched by name, which is all it carries).
    existing_images = list(
        db.scalars(select(Image).where(Image.project_id == project_id)).all()
    )
    by_hash = {i.content_hash: i for i in existing_images if i.content_hash}
    by_name = {i.original_filename.lower(): i for i in existing_images}

    def attach(image, width: int, height: int, group, name: str, proposed: bool) -> int:
        """Write this file's boxes for one image. Returns how many landed.

        `proposed` is the whole re-annotation story. Boxes for a NEW image are
        accepted ground truth; boxes for an image already in the project arrive
        as PROPOSALS, because they conflict with whatever is already there and
        that conflict is the user's to resolve. It reuses the review workflow
        auto-annotation already has: dashed boxes on the canvas, Accept (which
        replaces your boxes on the images covered) or Reject.

        The alternative — overwrite on import — silently destroys manual
        corrections made since the first import, which is exactly the kind of
        loss nothing later can detect.
        """
        written = 0
        for category_id, x, y, w, h in _boxes_for(group, name, width, height, plan, existing):
            # Clamp against the REAL dimensions, not what the file claims. They
            # disagree more often than you'd expect — resized exports with stale
            # metadata are common — and the image is the authority.
            x = max(0.0, min(x, width))
            y = max(0.0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)
            if w <= 0 or h <= 0:
                continue
            db.add(
                Annotation(
                    image_id=image.id,
                    category_id=category_id,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=None,  # human-authored ground truth, not a model's
                    # A third provenance value alongside auto/manual. NOT "auto":
                    # an auto-annotate run clears its own leftover proposals, and
                    # must never discard imported ones awaiting review.
                    source="imported",
                    reviewed=not proposed,
                    proposed=proposed,
                )
            )
            written += 1
        return written

    # --- images + annotations ----------------------------------------------
    for group in plan.groups:
        # An annotation file uploaded on its own: no pixels to store, just
        # labels for images already here. Matched by original filename, which is
        # the only identifier such a file carries.
        if not group.image_files and group.has_labels:
            for name in _labelled_names(group):
                image = by_name.get(name.lower())
                if image is None:
                    result.skipped.append(f"{name}: no such image in this project")
                    continue
                n = attach(image, image.width, image.height, group, name, proposed=True)
                result.proposals_created += n
                if n:
                    result.reannotated_images += 1
            continue

        for name, path in group.image_files.items():
            try:
                content = path.read_bytes()
                # Checked before saving, so re-importing a folder doesn't strew
                # orphaned copies of every image across storage/.
                digest = storage.content_digest(content)
            except OSError as exc:
                result.skipped.append(f"{name}: {exc}")
                continue

            duplicate = by_hash.get(digest)
            if duplicate is not None:
                # The image is already here. Its bytes aren't stored again — but
                # the annotations that came with it are NOT discarded, which is
                # what used to happen: the file was silently ignored while its
                # classes were still created, so a corrected export produced a
                # phantom class, no new boxes, and a success response.
                result.duplicates_skipped += 1
                n = attach(
                    duplicate, duplicate.width, duplicate.height, group, name, proposed=True
                )
                result.proposals_created += n
                if n:
                    result.reannotated_images += 1
                continue

            try:
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
                content_hash=saved.content_hash,
                import_id=import_id,
                split=group.split,
            )
            db.add(image)
            db.flush()
            by_hash[saved.content_hash] = image
            by_name[saved.original_filename.lower()] = image
            result.images_added += 1
            result.splits[group.split] = result.splits.get(group.split, 0) + 1

            result.annotations_added += attach(
                image, saved.width, saved.height, group, name, proposed=False
            )

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
