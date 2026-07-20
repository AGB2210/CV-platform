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

import io
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from app.config import settings

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


def save_image(project_id: int, content: bytes, original_name: str) -> SavedImage:
    """Validate and store one image. Raises ImageRejected if unusable."""
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
    )


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


def delete_project_dir(project_id: int) -> None:
    """Remove a project's entire image directory."""
    shutil.rmtree(settings.images_dir / str(project_id), ignore_errors=True)
