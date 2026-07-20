"""Image upload / listing / deletion endpoints."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image
from app.schemas import ImageRead, UploadResult
from app.services import dataset_import, storage
from app.services.dataset_version import has_any_version

router = APIRouter(tags=["images"])


@router.get("/projects/{project_id}/images", response_model=list[ImageRead])
def list_images(
    project_id: int,
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[ImageRead]:
    """List a project's images, oldest first, with annotation counts.

    Paginated from the start. A CV dataset reaches thousands of images quickly,
    and an unbounded list endpoint is a latent way to hang the browser once the
    project gets real. limit is clamped server-side — a client asking for
    ?limit=999999 doesn't get to decide how much memory we allocate.

    Ordered by id ASC, not DESC: the review workflow walks the dataset in a
    stable order, and "next image" should mean the next one, not a list that
    reshuffles as you upload.

    Counts come from a GROUP BY subquery joined on, not a COUNT per image —
    the N+1 problem again. outerjoin so unannotated images still appear.
    """
    get_project_or_404(project_id, db)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    counts = (
        select(
            Annotation.image_id.label("image_id"),
            # Accepted boxes only — a proposal isn't an annotation. Counting
            # them would make an untouched image look annotated, and the grid's
            # whole job is telling you what still needs work.
            func.sum(case((Annotation.proposed.is_(False), 1), else_=0)).label("n"),
            # SUM over a boolean: SQLite stores it as 0/1, so summing gives the
            # reviewed count in the same pass rather than a second query.
            func.sum(
                case(
                    (
                        (Annotation.reviewed.is_(True))
                        & (Annotation.proposed.is_(False)),
                        1,
                    ),
                    else_=0,
                )
            ).label("reviewed"),
            func.sum(case((Annotation.proposed.is_(True), 1), else_=0)).label("proposed"),
        )
        .group_by(Annotation.image_id)
        .subquery()
    )

    rows = db.execute(
        select(
            Image,
            func.coalesce(counts.c.n, 0),
            func.coalesce(counts.c.reviewed, 0),
            func.coalesce(counts.c.proposed, 0),
        )
        .outerjoin(counts, counts.c.image_id == Image.id)
        .where(Image.project_id == project_id)
        .order_by(Image.id.asc())
        .limit(limit)
        .offset(offset)
    ).all()

    results: list[ImageRead] = []
    for image, n, reviewed, proposed in rows:
        data = ImageRead.model_validate(image)
        data.annotation_count = n
        data.reviewed_count = reviewed
        data.proposed_count = proposed
        results.append(data)
    return results


@router.post(
    "/projects/{project_id}/images",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
    paths: list[str] | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResult:
    """Upload images, a zip, or a whole folder — possibly an annotated dataset.

    A single endpoint handles all of it because the user's intent is identical
    ("add this to my project") and the distinction is detectable from the bytes.
    Making the UI ask "is this a COCO export?" would push our implementation
    detail into their workflow — they already know what they dragged in.

    THREE SHAPES, ONE ANALYSER
    --------------------------
    1. Loose image files          -> stored directly, no labels.
    2. A zip                      -> extracted, then analysed.
    3. A folder, or a selection    -> materialised into a temp directory at
       containing annotation files    those relative paths, then analysed.

    (2) and (3) both end up calling `dataset_import.analyse()` on a directory
    tree, which is the point: format detection, per-split scoping and the
    COCO/YOLO readers exist once. A folder picked in the browser and the same
    folder zipped must import identically, and the only way to guarantee that
    is for them to run the same code.

    `paths` carries each file's path relative to the chosen folder, because
    browsers put only the BASENAME in the multipart filename — the directory
    structure, which is exactly what tells train/ from val/, is otherwise lost.

    Partial success is the design: a batch of 50 with 2 corrupt files stores 48
    and reports the 2. Rejecting the whole batch would be hostile, and silently
    dropping them would be worse.
    """
    get_project_or_404(project_id, db)

    # A folder upload, or a selection carrying annotation files alongside the
    # images. Either way the structure matters, so it goes through the analyser
    # rather than being stored as loose pictures with the labels discarded.
    if paths or _has_annotation_files(files):
        return await _import_as_tree(db, project_id, files, paths)

    saved: list[storage.SavedImage] = []
    skipped: list[str] = []
    import_results: list[dataset_import.ImportResult] = []

    for upload in files:
        # `await upload.read()` — UploadFile is async because Starlette spools
        # large uploads to a temp file rather than holding them in RAM. This is
        # the one genuinely async part of the endpoint, which is why the
        # function is `async def`.
        content = await upload.read()
        name = upload.filename or "unnamed"

        if storage.is_zip(name, content):
            try:
                import_results.append(_import_zip(db, project_id, content))
            except (ValueError, zipfile.BadZipFile) as exc:
                skipped.append(f"{name}: {exc}")
        else:
            try:
                saved.append(storage.save_image(project_id, content, name))
            except storage.ImageRejected as exc:
                skipped.append(f"{name}: {exc}")

    # A zip took the import path and already committed its own rows, so the
    # plain-image path below has nothing to do for it.
    if import_results and not saved:
        merged = _merge_results(import_results, skipped)
        if merged.images_added == 0:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No valid images found in the archive. "
                + "; ".join(merged.skipped[:5]),
            )
        return _result_to_response(db, project_id, merged)

    if not saved and skipped:
        # Nothing at all was storable. A 201 here would be a lie, and the UI
        # would show a success toast for an upload that did nothing.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images found. " + "; ".join(skipped[:5]),
        )

    # One commit for the whole batch, not one per file. 500 individual commits
    # means 500 fsyncs, which is slow enough to feel broken; batching also means
    # the DB never reflects a half-finished upload.
    rows = [
        Image(
            project_id=project_id,
            filename=s.filename,
            original_filename=s.original_filename,
            width=s.width,
            height=s.height,
            size_bytes=s.size_bytes,
        )
        for s in saved
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)

    response = UploadResult(
        uploaded=[ImageRead.model_validate(r) for r in rows],
        skipped=skipped,
    )
    # Merge in anything a zip contributed alongside loose files in the same
    # request (rare, but the UI allows dropping both at once).
    if import_results:
        merged = _merge_results(import_results, [])
        response.annotations_imported = merged.annotations_added
        response.classes_created = merged.classes_created
        response.splits = merged.splits
        response.has_split_folders = merged.has_split_folders
        response.notes = merged.notes
    return response


# --- import helpers ---------------------------------------------------------

#: Files that carry labels rather than pixels. Their presence in a selection is
#: what turns "upload these pictures" into "import this dataset".
ANNOTATION_SUFFIXES = {".json", ".txt", ".yaml", ".yml"}


def _has_annotation_files(files: list[UploadFile]) -> bool:
    """Did the user include labels alongside the images?

    This is what makes "select images AND _annotations.coco.json" work in the
    file picker. Without it those files are silently not images, and the upload
    quietly drops every label in the selection.
    """
    return any(
        Path(f.filename or "").suffix.lower() in ANNOTATION_SUFFIXES for f in files
    )


def _safe_relative(raw: str, fallback: str) -> Path | None:
    """A user-supplied relative path, or None if it tries to escape.

    `paths` comes from the browser and is therefore untrusted. It's the only
    place in the upload flow where a client-supplied string becomes a
    filesystem path, so it gets the same treatment as a zip member: absolute
    paths, drive letters and any '..' component are refused outright rather
    than normalised, because a path that needed sanitising isn't one a real
    folder upload would have produced.
    """
    candidate = (raw or fallback).replace("\\", "/").strip("/")
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute() or path.drive or any(p == ".." for p in path.parts):
        return None
    return path


async def _import_as_tree(
    db: Session,
    project_id: int,
    files: list[UploadFile],
    paths: list[str] | None,
) -> UploadResult:
    """Rebuild the uploaded selection as a directory tree, then import it.

    The tree is scratch: images we keep are re-saved into storage/ under
    generated names by the importer, exactly as the zip path does.
    """
    tmp = Path(tempfile.mkdtemp(prefix="cvfolder_"))
    try:
        root = tmp / "tree"
        root.mkdir()
        rejected: list[str] = []
        written = 0

        for i, upload in enumerate(files):
            name = upload.filename or f"file{i}"
            # paths[i] pairs with files[i]: the frontend sends them in step.
            raw = paths[i] if paths and i < len(paths) else name
            relative = _safe_relative(raw, name)
            if relative is None:
                rejected.append(f"{name}: unsafe path {raw!r}")
                continue

            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(await upload.read())
            written += 1

        if not written:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "Nothing could be read from that selection. " + "; ".join(rejected[:5]),
            )

        plan = dataset_import.analyse(root)
        if not plan.groups:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No images found in that folder. A dataset needs image files, "
                "not only annotations.",
            )

        merged = dataset_import.execute(db, project_id, plan)
        merged.skipped.extend(rejected)
        return _result_to_response(db, project_id, merged)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _import_zip(db: Session, project_id: int, content: bytes) -> dataset_import.ImportResult:
    """Extract a zip to a temp dir, analyse it, and import what's there.

    The temp dir is deleted on every path — the extracted copy is scratch; the
    images we keep were re-saved into storage/ under generated names by
    `storage.save_image`.
    """
    tmp = Path(tempfile.mkdtemp(prefix="cvimport_"))
    try:
        archive = tmp / "upload.zip"
        archive.write_bytes(content)
        extract_dir = tmp / "x"
        dataset_import.extract_archive(archive, extract_dir)

        plan = dataset_import.analyse(extract_dir)
        if not plan.groups:
            raise ValueError("no images found in archive")
        return dataset_import.execute(db, project_id, plan)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _merge_results(
    results: list[dataset_import.ImportResult], extra_skipped: list[str]
) -> dataset_import.ImportResult:
    """Fold several zip imports (one per uploaded archive) into one summary."""
    merged = dataset_import.ImportResult()
    for r in results:
        merged.images_added += r.images_added
        merged.annotations_added += r.annotations_added
        merged.classes_created.extend(r.classes_created)
        merged.skipped.extend(r.skipped)
        merged.notes.extend(r.notes)
        merged.has_split_folders = merged.has_split_folders or r.has_split_folders
        for split, n in r.splits.items():
            merged.splits[split] = merged.splits.get(split, 0) + n
    merged.skipped.extend(extra_skipped)
    return merged


def _result_to_response(
    db: Session, project_id: int, merged: dataset_import.ImportResult
) -> UploadResult:
    """Shape an ImportResult into the upload response the frontend expects."""
    recent = list(
        db.scalars(
            select(Image)
            .where(Image.project_id == project_id)
            .order_by(Image.id.desc())
            .limit(merged.images_added)
        ).all()
    )
    return UploadResult(
        uploaded=[ImageRead.model_validate(r) for r in reversed(recent)],
        skipped=merged.skipped,
        annotations_imported=merged.annotations_added,
        classes_created=merged.classes_created,
        splits=merged.splits,
        has_split_folders=merged.has_split_folders,
        notes=merged.notes,
    )


@router.delete("/images/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: int, db: Session = Depends(get_db)) -> None:
    """Remove one image from the dataset.

    THE FILE IS KEPT ONCE THE PROJECT HAS SAVED VERSIONS.

    A dataset version stores metadata only — which images are in it, not their
    bytes — so restoring one can only work if the pictures are still on disk.
    Unlinking here would turn every version that references this image into a
    promise we can't keep, which is precisely the accident versions exist to
    prevent. The row goes (it leaves the live dataset immediately); the bytes
    stay, and a restore recreates the row pointing at them.

    Before the first save there is nothing that could restore it, so the file is
    removed as before rather than accumulating orphans for no benefit.
    """
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    project_id, filename = image.project_id, image.filename
    recoverable = has_any_version(db, project_id)
    db.delete(image)
    db.commit()

    if not recoverable:
        storage.delete_image_file(project_id, filename)
