"""Image upload / listing / deletion endpoints."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Annotation, Image
from app.schemas import ImageRead, UploadResult
from app.services import dataset_import, storage

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
    db: Session = Depends(get_db),
) -> UploadResult:
    """Upload images, or a zip — which may be a whole annotated dataset.

    A single endpoint handles all of it because the user's intent is identical
    ("add this to my project") and the distinction is detectable from the bytes.
    Making the UI ask "is this a COCO export?" would push our implementation
    detail into their workflow — they already know what they dragged in.

    A zip is inspected rather than assumed (see services/dataset_import.py):
    loose images, a flat COCO dataset, or Roboflow's train/valid/test layout all
    import correctly with no format dropdown.

    Partial success is the design: a batch of 50 with 2 corrupt files stores 48
    and reports the 2. Rejecting the whole batch would be hostile, and silently
    dropping them would be worse.
    """
    get_project_or_404(project_id, db)

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
    """Delete one image, row then file — same ordering rationale as projects."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {image_id} not found")

    project_id, filename = image.project_id, image.filename
    db.delete(image)
    db.commit()

    storage.delete_image_file(project_id, filename)
