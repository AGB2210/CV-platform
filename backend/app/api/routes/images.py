"""Image upload / listing / deletion endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.routes.projects import get_project_or_404
from app.database import get_db
from app.models import Image
from app.schemas import ImageRead, UploadResult
from app.services import storage

router = APIRouter(tags=["images"])


@router.get("/projects/{project_id}/images", response_model=list[ImageRead])
def list_images(
    project_id: int,
    limit: int = 200,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[Image]:
    """List a project's images, newest first.

    Paginated from the start. A CV dataset reaches thousands of images quickly,
    and an unbounded list endpoint is a latent way to hang the browser once the
    project gets real. limit is clamped server-side — a client asking for
    ?limit=999999 doesn't get to decide how much memory we allocate.
    """
    get_project_or_404(project_id, db)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    return list(
        db.scalars(
            select(Image)
            .where(Image.project_id == project_id)
            .order_by(Image.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )


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
    """Upload one or more images, or a zip archive of images.

    A single endpoint handles both because the client's intent is identical
    ("add these images to my project") and the distinction is detectable from
    the bytes. Making the UI choose between two endpoints would push an
    implementation detail into the interface.

    Partial success is the design: a batch of 50 with 2 corrupt files stores 48
    and reports the 2. Rejecting the whole batch would be hostile, and silently
    dropping them would be worse.
    """
    get_project_or_404(project_id, db)

    saved: list[storage.SavedImage] = []
    skipped: list[str] = []

    for upload in files:
        # `await upload.read()` — UploadFile is async because Starlette spools
        # large uploads to a temp file rather than holding them in RAM. This is
        # the one genuinely async part of the endpoint, which is why the
        # function is `async def`.
        content = await upload.read()
        name = upload.filename or "unnamed"

        if storage.is_zip(name, content):
            try:
                zip_saved, zip_skipped = storage.save_zip(project_id, content)
                saved.extend(zip_saved)
                skipped.extend(zip_skipped)
            except storage.ImageRejected as exc:
                skipped.append(f"{name}: {exc}")
        else:
            try:
                saved.append(storage.save_image(project_id, content, name))
            except storage.ImageRejected as exc:
                skipped.append(f"{name}: {exc}")

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

    return UploadResult(
        uploaded=[ImageRead.model_validate(r) for r in rows],
        skipped=skipped,
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
