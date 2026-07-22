"""
Image storage: validating uploads, writing bytes to disk, reading them back.

All filesystem work for images funnels through here. Routes call these
functions; they never touch paths themselves. That keeps the security-sensitive
logic (what counts as an image, where a file is allowed to land) in one
auditable place instead of smeared across endpoints.

Layout on disk:

    storage/images/<project_id>/<uuid>.<ext>

Partitioning by project means deleting a project is one `rmtree`, and no single
directory accumulates every image ever uploaded — which matters because
filesystem listing degrades badly past a few tens of thousands of entries.
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageOps
from PIL import UnidentifiedImageError

from app.config import settings

logger = logging.getLogger(__name__)

# Formats we accept. Restricting this is a security control, not a convenience:
# it's the allowlist that decides what bytes we're willing to write to disk.
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Pillow's own format names, checked against the DECODED image rather than the
# filename. The extension is a claim; this is the verification.
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}

# Bounds the damage a single bad/hostile file can do.
MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MB per image
MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per archive
MAX_ZIP_MEMBERS = 20_000

# Pillow refuses images above ~89M pixels by default, guarding against
# "decompression bomb" files: a tiny PNG that expands to gigabytes of RAM when
# decoded. We keep that default and let the resulting exception mark the file
# as skipped rather than crash the request.


@dataclass
class SavedImage:
    """Metadata for one successfully stored image, ready for the DB."""

    filename: str
    original_filename: str
    width: int
    height: int
    size_bytes: int
    #: SHA-256 of the bytes — how a re-upload of the same picture is recognised.
    content_hash: str = ""


class ImageRejected(Exception):
    """A single file was not storable. Carries a user-facing reason.

    Deliberately not an HTTPException: this module knows nothing about HTTP.
    The route decides whether a rejection is a 400 (the whole request was one
    bad file) or just an entry in the `skipped` list (one bad file in a batch).
    """


def project_dir(project_id: int) -> Path:
    """Directory holding one project's images. Created on demand."""
    path = settings.images_dir / str(project_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_and_measure(content: bytes, original_name: str) -> tuple[int, int, str]:
    """Confirm bytes really are an allowed image. Return (width, height, ext).

    Note what this does NOT do: trust the filename or the client's Content-Type.
    Both are attacker-controlled. A file called `cat.jpg` sent as `image/jpeg`
    can contain anything at all. The only trustworthy check is handing the bytes
    to a decoder and seeing whether it produces an image.
    """
    if len(content) > MAX_IMAGE_BYTES:
        raise ImageRejected(
            f"larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        )
    if not content:
        raise ImageRejected("file is empty")

    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageRejected(f"unsupported extension '{ext or "(none)"}'")

    # Two passes are required by Pillow's API, not sloppiness: verify() checks
    # integrity but leaves the object unusable afterwards, so measuring needs a
    # fresh open. Both read from memory, so there's no second disk hit.
    try:
        with PILImage.open(io.BytesIO(content)) as img:
            img.verify()  # detects truncation / corruption
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected(f"not a readable image ({type(exc).__name__})") from exc

    try:
        with PILImage.open(io.BytesIO(content)) as img:
            fmt = img.format
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageRejected(f"could not read dimensions ({type(exc).__name__})") from exc

    if fmt not in ALLOWED_PIL_FORMATS:
        # Reached when the extension lies — e.g. a GIF renamed to .png.
        raise ImageRejected(f"decoded as {fmt}, which is not supported")

    if width <= 0 or height <= 0:
        raise ImageRejected("image has zero width or height")

    return width, height, ext


def normalise_orientation(content: bytes, ext: str) -> bytes:
    """Bake an EXIF rotation into the pixels, so W×H means one thing.

    THE BUG THIS PREVENTS. Cameras and phones almost never rotate the pixels
    when you turn the device — they store the image as the sensor read it and
    add an EXIF Orientation tag saying how to display it. So a portrait photo is
    commonly 4032×3024 on disk with "rotate 90" attached.

    Browsers honour that tag. PIL's `.size` does not. We stored the raw
    dimensions and the annotation canvas uses them as its SVG viewBox, so the
    coordinate space was TRANSPOSED relative to the picture the user was drawing
    on: every box on a rotated photo landed somewhere else entirely. Nothing
    errors, the boxes look fine while you draw them, and the dataset is wrong.

    Normalising here rather than teaching each consumer about EXIF means the
    bytes on disk are the truth. The canvas, both exporters, the trainer and any
    future model all see the same image, and none of them has to remember a rule.

    Cost: a rotated JPEG is re-encoded once, so it loses a little quality and
    the stored bytes differ from the file the user picked. That is the right
    trade against silently-misplaced ground truth. Untagged images — the large
    majority — are returned byte-for-byte untouched.
    """
    try:
        with PILImage.open(io.BytesIO(content)) as img:
            orientation = (img.getexif() or {}).get(0x0112)  # 0x0112 = Orientation
            if not orientation or orientation == 1:
                return content  # upright already; don't touch the bytes

            upright = ImageOps.exif_transpose(img)
            if upright is None:
                return content

            buf = io.BytesIO()
            fmt = "JPEG" if ext in (".jpg", ".jpeg") else (img.format or "PNG")
            # quality=95 because this is ground-truth imagery and the re-encode
            # is not something the user asked for.
            #
            # NOT subsampling="keep", which is only valid when saving an
            # unmodified JPEG. exif_transpose returns a new image with no
            # `format`, so it raises — and because that raise was swallowed
            # below, the whole function became a silent no-op that returned the
            # original bytes. The rotation "worked" and changed nothing.
            save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
            # exif_transpose already dropped the orientation tag, so the result
            # cannot be double-rotated by anything downstream.
            upright.save(buf, fmt, **save_kwargs)
            return buf.getvalue()
    except Exception:  # noqa: BLE001
        # A file we can't re-encode is not worth failing an upload over — the
        # validation pass above already proved it decodes. LOGGED, though:
        # silently returning the original is exactly how this shipped broken
        # once, and a rotated image stored un-rotated is a wrong dataset rather
        # than a missing feature.
        logger.warning(
            "Could not normalise EXIF orientation; storing as-is", exc_info=True
        )
        return content


def save_image(project_id: int, content: bytes, original_name: str) -> SavedImage:
    """Validate and store one image. Raises ImageRejected if unusable."""
    width, height, ext = _validate_and_measure(content, original_name)

    # Before anything measures or hashes it: rotate the pixels upright if EXIF
    # says the camera was sideways, then re-measure. Order matters — the hash
    # and the stored dimensions must describe the bytes we actually keep.
    upright = normalise_orientation(content, ext)
    if upright is not content:
        content = upright
        width, height, ext = _validate_and_measure(content, original_name)

    # A generated name, never the user's. This is what makes path traversal
    # structurally impossible rather than something we have to remember to
    # sanitise: an upload called "../../../etc/passwd" becomes "a1b2....jpg" and
    # lands in the project directory like everything else.
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = project_dir(project_id) / stored_name
    dest.write_bytes(content)

    return SavedImage(
        filename=stored_name,
        original_filename=Path(original_name).name[:255],  # strip any path parts
        width=width,
        height=height,
        size_bytes=len(content),
        content_hash=content_digest(content),
    )


def content_digest(content: bytes) -> str:
    """SHA-256 of an image's bytes.

    Identity for "is this the same picture?". Filenames can't answer that —
    stored names are generated UUIDs, and every dataset in the world calls its
    files img_0001.jpg. Bytes are the only thing that actually distinguishes.

    SHA-256 rather than a faster non-cryptographic hash because a collision here
    silently DISCARDS a real image as a duplicate, and at 256 bits that will not
    happen by accident. Hashing is not the bottleneck either: it's a few hundred
    MB/s against a decode-and-validate step that already read the same bytes.
    """
    return hashlib.sha256(content).hexdigest()


def is_zip(filename: str, content: bytes) -> bool:
    """True if this upload looks like a zip.

    Checks the magic number, not just the extension — the extension is a hint
    from the client, the first four bytes are evidence. ("PK\\x03\\x04" is the
    zip local file header signature.)
    """
    if content[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return True
    return filename.lower().endswith(".zip")


def delete_image_file(project_id: int, filename: str) -> None:
    """Remove one image's bytes. Missing file is not an error.

    Deleting an already-absent file should be a no-op: the caller's intent —
    "this should not exist" — is satisfied either way, and raising would make
    cleanup after a partial failure impossible.
    """
    path = project_dir(project_id) / filename
    path.unlink(missing_ok=True)
    # Its cached thumbnail goes with it — a thumb of a deleted image is dead
    # bytes that would sit in the cache forever.
    (settings.thumbs_dir / str(project_id) / f"{filename}.jpg").unlink(missing_ok=True)


def delete_project_dir(project_id: int) -> None:
    """Remove a project's entire image directory (and its thumbnail cache)."""
    shutil.rmtree(settings.images_dir / str(project_id), ignore_errors=True)
    shutil.rmtree(settings.thumbs_dir / str(project_id), ignore_errors=True)


def delete_project_files(project_id: int, job_ids: list[int]) -> None:
    """Remove EVERYTHING a project owns on disk.

    A project's bytes live in three places, and deleting one used to be the
    whole cleanup:

        storage/images/<project_id>/   uploads
        storage/versions/<project_id>/ dataset snapshots
        storage/runs/<job_id>/         checkpoints, one dir per training run

    The rows for all three go by FK cascade when the project row is deleted, so
    nothing dangles in the database and the leak is invisible from inside the
    app — but the files stayed forever. A project with ten training runs left
    ~50 MB of checkpoints behind, and the only way to notice was to look at the
    disk.

    `job_ids` has to be collected BEFORE the project row is deleted, because
    afterwards there is nothing left to say which run directories were its.

    ignore_errors throughout: this runs after the DB commit, so a file that
    can't be removed (locked on Windows, already gone) must not raise. The row
    is already gone and the caller's intent is satisfied; a leftover file is
    wasted disk, while an exception here would surface as a failed delete for
    an operation that actually succeeded.
    """
    delete_project_dir(project_id)
    shutil.rmtree(settings.versions_dir / str(project_id), ignore_errors=True)
    for job_id in job_ids:
        shutil.rmtree(settings.runs_dir / str(job_id), ignore_errors=True)
