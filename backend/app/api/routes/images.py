"""Image upload / listing / deletion endpoints."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

# Imported at module level ON PURPOSE. This used to be a lazy import inside
# get_thumbnail with a quoted `-> "FileResponse"` annotation — a forward
# reference Pydantic could never resolve, and ONE unresolvable annotation
# anywhere poisons the ENTIRE OpenAPI schema: /openapi.json 500'd, so /docs
# showed "Failed to load API definition" for three releases before anyone
# opened it. The import is starlette re-exported and costs nothing.
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image
from app.schemas import ImageRead, UploadResult
from app.services import dataset_import, storage, storage_audit
from app.services.dataset_version import has_any_version

router = APIRouter(tags=["images"])


#: Longest edge of a cached grid thumbnail. 256 is crisp in a ~150px grid cell
#: on a 2x display, and a 256px JPEG is ~10-30 KB against multi-MB originals —
#: which is the difference between a smooth 200-image scroll and a lagging one.
THUMB_SIZE = 256


@router.get("/thumbs/{project_id}/{filename}", response_class=FileResponse)
def get_thumbnail(project_id: int, filename: str) -> FileResponse:
    """A small cached JPEG of one stored image, for grids and filmstrips.

    The scroll-lag fix: grids used to render the ORIGINALS (multi-megabyte,
    thousands of pixels) into 150px cells, so a 200-image page decoded hundreds
    of megapixels while you scrolled. Cells now ask for this instead.

    Generated on first request and cached under storage/thumbs/ — a pure
    cache: deleting the directory costs regeneration, never data. Cacheable
    forever on the browser side too, because stored filenames are content-
    addressed uuids that never change their bytes.
    """
    # The filename is a path segment from the URL — refuse anything that could
    # walk out of the project directory.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such image")

    src = storage.project_dir(project_id) / filename
    if not src.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such image")

    from app.config import settings

    dest = settings.thumbs_dir / str(project_id) / f"{filename}.jpg"
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        from PIL import Image as PILImage

        dest.parent.mkdir(parents=True, exist_ok=True)
        with PILImage.open(src) as im:
            thumb = im.convert("RGB")  # JPEG has no alpha; RGBA sources would fail
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
            thumb.save(dest, "JPEG", quality=80)

    return FileResponse(
        dest,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/projects/{project_id}/images", response_model=list[ImageRead])
def list_images(
    project_id: int,
    response: Response,
    limit: int = 200,
    offset: int = 0,
    split: str | None = None,
    state: str | None = None,
    category_id: int | None = None,
    db: Session = Depends(get_db),
) -> list[ImageRead]:
    """List a project's images, oldest first, with annotation counts.

    Paginated from the start. A CV dataset reaches thousands of images quickly,
    and an unbounded list endpoint is a latent way to hang the browser once the
    project gets real. limit is clamped server-side — a client asking for
    ?limit=999999 doesn't get to decide how much memory we allocate.

    THE TOTAL COMES BACK IN THE `X-Total-Count` HEADER. Without it a client
    cannot tell "200 images" from "the first 200 of 5,000", and that is exactly
    what went wrong: the grid asked for the default page, got 200 rows, and
    rendered them as if they were the whole dataset. A header rather than an
    envelope keeps the response body a plain list, so the callers that legitimately
    want "some images" (Review, Visualize) need no changes.

    FILTERS RUN SERVER-SIDE (`split`, `state`, `category_id`) so they apply to
    the WHOLE dataset, not whichever page happens to be loaded. Filtering the
    loaded page client-side shipped a real contradiction: the stats banner said
    "1 image has no boxes" (a whole-dataset count) while the No-boxes filter
    found nothing, because that one image sat beyond page 1. `state` is one of
    `annotated` (has accepted boxes), `unannotated` (none), `pending` (has
    proposals). The total header reflects the filtered count.

    Ordered by id ASC, not DESC: the review workflow walks the dataset in a
    stable order, and "next image" should mean the next one, not a list that
    reshuffles as you upload.

    Counts come from a GROUP BY subquery joined on, not a COUNT per image —
    the N+1 problem again. outerjoin so unannotated images still appear.
    """
    get_project_or_404(project_id, db)
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    filters = [Image.project_id == project_id]
    if split is not None:
        filters.append(Image.split == split)
    if state == "annotated":
        filters.append(
            select(Annotation.id)
            .where(Annotation.image_id == Image.id, Annotation.proposed.is_(False))
            .exists()
        )
    elif state == "unannotated":
        filters.append(
            ~select(Annotation.id)
            .where(Annotation.image_id == Image.id, Annotation.proposed.is_(False))
            .exists()
        )
    elif state == "pending":
        filters.append(
            select(Annotation.id)
            .where(Annotation.image_id == Image.id, Annotation.proposed.is_(True))
            .exists()
        )
    if category_id is not None:
        filters.append(
            select(Annotation.id)
            .where(
                Annotation.image_id == Image.id,
                Annotation.category_id == category_id,
            )
            .exists()
        )

    total = db.scalar(select(func.count(Image.id)).where(*filters)) or 0
    response.headers["X-Total-Count"] = str(total)

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
        .where(*filters)
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


@router.get("/images/{image_id}", response_model=ImageRead)
def get_image(image_id: int, db: Session = Depends(get_db)) -> ImageRead:
    """One image by id, carrying the same counts the list rows do.

    The deep-link resolver. Review can be linked straight to an image — the
    worst-test-images grid on Evaluate does exactly that, and a shared URL
    can too — and that image is routinely OUTSIDE the first page the list
    endpoint returns. Without this, the editor had nothing to render and the
    page came up blank.
    """
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    n, reviewed, proposed = db.execute(
        select(
            func.sum(case((Annotation.proposed.is_(False), 1), else_=0)),
            func.sum(
                case(
                    (
                        (Annotation.reviewed.is_(True))
                        & (Annotation.proposed.is_(False)),
                        1,
                    ),
                    else_=0,
                )
            ),
            func.sum(case((Annotation.proposed.is_(True), 1), else_=0)),
        ).where(Annotation.image_id == image_id)
    ).one()
    data = ImageRead.model_validate(image)
    data.annotation_count = n or 0
    data.reviewed_count = reviewed or 0
    data.proposed_count = proposed or 0
    return data


@router.post(
    "/projects/{project_id}/images",
    response_model=UploadResult,
    status_code=status.HTTP_201_CREATED,
)
async def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
    paths: list[str] | None = Form(None),
    import_id: str | None = Form(None),
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
        return await _import_as_tree(db, project_id, files, paths, import_id)

    saved: list[storage.SavedImage] = []
    skipped: list[str] = []
    duplicates = 0
    seen = _existing_hashes(db, project_id)
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
            # Recognise a re-upload BEFORE writing the bytes, so a repeated
            # folder doesn't leave a pile of orphaned files on disk.
            digest = storage.content_digest(content)
            if digest in seen:
                duplicates += 1
                continue
            try:
                saved.append(storage.save_image(project_id, content, name))
                seen.add(digest)  # catches duplicates WITHIN one batch too
            except storage.ImageRejected as exc:
                skipped.append(f"{name}: {exc}")

    # A zip took the import path and already committed its own rows, so the
    # plain-image path below has nothing to do for it.
    if import_results and not saved:
        merged = _merge_results(import_results, skipped)
        # 400 only when NOTHING happened at all. A re-upload of an archive
        # whose images are all already here is a real outcome, not an error:
        # duplicates were recognised (and possibly its annotation file arrived
        # as fresh proposals) — reporting "no valid images" for that told the
        # user their zip was broken while quietly changing their project.
        if (
            merged.images_added == 0
            and merged.duplicates_skipped == 0
            and merged.proposals_created == 0
        ):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "No valid images found in the archive. "
                + "; ".join(merged.skipped[:5]),
            )
        return _result_to_response(db, project_id, merged)

    if not saved and skipped:
        # Nothing at all was storable. A 201 here would be a lie, and the UI
        # would show a success toast for an upload that did nothing.
        #
        # Every reason, not the first five: with a whole folder rejected, the
        # one line explaining why is often not among the first few.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images found.\n" + "\n".join(skipped),
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
            content_hash=s.content_hash,
            import_id=import_id,
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
        duplicates_skipped=duplicates,
        import_id=import_id,
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
        response.duplicates_skipped += merged.duplicates_skipped
    return response


# --- import helpers ---------------------------------------------------------

#: Files that carry labels rather than pixels. Their presence in a selection is
#: what turns "upload these pictures" into "import this dataset".
ANNOTATION_SUFFIXES = {".json", ".txt", ".yaml", ".yml"}


def _existing_hashes(db: Session, project_id: int) -> set[str]:
    """Content hashes already in this project.

    Fetched once per request rather than queried per file: a 400-image batch
    would otherwise issue 400 SELECTs to ask the same question. Rows predating
    the column have NULL and are excluded — they simply can't participate in
    duplicate detection until the backfill script has run over them.
    """
    return {
        h
        for h in db.scalars(
            select(Image.content_hash).where(
                Image.project_id == project_id, Image.content_hash.is_not(None)
            )
        ).all()
    }


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
    import_id: str | None = None,
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
                "Nothing importable in that selection — no image files, and no "
                "annotation file we could read as COCO or YOLO.",
            )

        merged = dataset_import.execute(db, project_id, plan, import_id)
        merged.skipped.extend(rejected)
        return _result_to_response(db, project_id, merged, import_id)
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
        merged.duplicates_skipped += r.duplicates_skipped
        merged.proposals_created += r.proposals_created
        merged.reannotated_images += r.reannotated_images
        for split, n in r.splits.items():
            merged.splits[split] = merged.splits.get(split, 0) + n
    merged.skipped.extend(extra_skipped)
    return merged


def _result_to_response(
    db: Session,
    project_id: int,
    merged: dataset_import.ImportResult,
    import_id: str | None = None,
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
        duplicates_skipped=merged.duplicates_skipped,
        import_id=import_id,
        proposals_created=merged.proposals_created,
        reannotated_images=merged.reannotated_images,
    )


class BulkDeleteImages(BaseModel):
    image_ids: list[int]


@router.post("/projects/{project_id}/images/bulk-delete")
def bulk_delete_images(
    project_id: int, payload: BulkDeleteImages, db: Session = Depends(get_db)
) -> dict:
    """Delete several images at once — also how "delete all" is sent.

    Selecting everything takes the same path as selecting one, so there is no
    separate "delete all" code path that could behave differently from the one
    that gets exercised daily.

    POST, not DELETE, for the same reason as the project bulk delete: a body on
    DELETE is legal but poorly supported, and a comma-joined query string breaks
    at a few hundred ids — which a dataset reaches immediately.

    Scoped to the project, so an id from another project can't be deleted just
    because it appeared in the request body. Ids that don't exist are reported,
    not fatal: deleting something already gone is not a failure.
    """
    get_project_or_404(project_id, db)

    images = list(
        db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.id.in_(payload.image_ids)
            )
        ).all()
    )
    # The file-retention rule is decided ONCE for the batch, not per image: it
    # depends on whether the project has any version, which cannot change
    # mid-loop. See delete_image for why the bytes are kept.
    recoverable = has_any_version(db, project_id)
    filenames = [img.filename for img in images]

    for img in images:
        db.delete(img)
    db.commit()

    if not recoverable:
        for filename in filenames:
            storage.delete_image_file(project_id, filename)

    found = {img.id for img in images}
    return {
        "deleted": len(images),
        "not_found": sorted(set(payload.image_ids) - found),
        #: True when the bytes were kept on disk so a restore can bring these
        #: back. The UI says so — "deleted 200 images" reads as permanent
        #: otherwise, and it isn't.
        "recoverable": recoverable,
    }


# --- storage housekeeping ---------------------------------------------------


@router.get("/projects/{project_id}/storage")
def storage_report(project_id: int, db: Session = Depends(get_db)) -> dict:
    """What this project is holding on disk, and what could be freed.

    Read-only. Three different "not needed" states get counted separately
    because they mean genuinely different things — see services/storage_audit.
    """
    get_project_or_404(project_id, db)
    report = storage_audit.audit(db, project_id)
    return {
        "total_images": report.total_images,
        "unsaved_images": report.unsaved_images,
        "orphan_files": report.orphan_files,
        "orphan_bytes": report.orphan_bytes,
        "retained_files": report.retained_files,
        "retained_bytes": report.retained_bytes,
        "unreadable_versions": report.unreadable_versions,
    }


@router.post("/projects/{project_id}/storage/reclaim")
def reclaim_storage(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Delete image files that nothing can reach.

    Only files referenced by neither a live row nor any version snapshot. Files
    kept solely because a version needs them are NOT touched — that retention is
    what makes restore work.
    """
    get_project_or_404(project_id, db)
    try:
        removed, freed = storage_audit.reclaim_orphans(db, project_id)
    except ValueError as exc:
        # An unreadable snapshot means we can't tell orphans from retained
        # files, and guessing deletes irreplaceable data.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    return {"files_removed": removed, "bytes_freed": freed}


@router.post("/projects/{project_id}/storage/discard-unsaved")
def discard_unsaved(project_id: int, db: Session = Depends(get_db)) -> dict:
    """Delete live images that no saved version contains.

    THE USER'S DECISION, NEVER AUTOMATIC. The whole upload -> annotate -> save
    workflow lives in the unsaved state, so "not in a version" does not mean
    "unwanted" — it usually means "still being worked on". This exists so an
    unwanted 700 MB upload can be undone in one action, not so the app can tidy
    up behind someone's back.

    Their annotations go too, by cascade. Nothing here is in a version, so
    nothing here is recoverable — which is why the UI states the count and
    colours the action destructive.
    """
    get_project_or_404(project_id, db)

    ids = storage_audit.unsaved_image_ids(db, project_id)
    if not ids:
        return {"deleted": 0, "bytes_freed": 0}

    images = list(db.scalars(select(Image).where(Image.id.in_(ids))).all())
    filenames = [img.filename for img in images]
    for img in images:
        db.delete(img)
    db.commit()

    # Files go too. These are in no version by definition, so unlike the normal
    # delete path there is nothing that could ever restore them and keeping the
    # bytes would just be litter.
    freed = 0
    project_dir = storage.project_dir(project_id)
    for filename in filenames:
        path = project_dir / filename
        try:
            freed += path.stat().st_size
            path.unlink()
        except OSError:
            continue

    return {"deleted": len(images), "bytes_freed": freed}


@router.post("/projects/{project_id}/imports/{import_id}/undo")
def undo_import(project_id: int, import_id: str, db: Session = Depends(get_db)) -> dict:
    """Remove every image added by one upload.

    A large folder is sent as many requests, and one failing at batch 12 of 27
    leaves eleven batches committed. The counts said so but nothing could act on
    it, so recovery meant picking those images out of the grid by hand.

    Every image from one upload carries the same `import_id`, so the whole thing
    comes out as a unit regardless of how many requests it took.

    Images from this import that a version has SINCE captured are kept and
    reported: once a save point depends on an image, silently removing it would
    break that version, and by saving the user has said they want it.
    """
    get_project_or_404(project_id, db)

    images = list(
        db.scalars(
            select(Image).where(
                Image.project_id == project_id, Image.import_id == import_id
            )
        ).all()
    )
    if not images:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "That import has nothing left to undo — it may already have been "
            "undone, or its images deleted.",
        )

    referenced, _ = storage_audit._referenced_by_versions(db, project_id)
    removable = [img for img in images if img.filename not in referenced]
    kept = len(images) - len(removable)

    filenames = [img.filename for img in removable]
    for img in removable:
        db.delete(img)
    db.commit()

    freed = 0
    project_dir = storage.project_dir(project_id)
    for filename in filenames:
        path = project_dir / filename
        try:
            freed += path.stat().st_size
            path.unlink()
        except OSError:
            continue

    return {"deleted": len(removable), "kept_in_versions": kept, "bytes_freed": freed}


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
