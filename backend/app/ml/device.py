"""
Device selection and VRAM accounting.

Isolated in its own module because every model in the project needs the same
answers ("which device? how much memory is left?") and because it's the one
place that knows this machine has a small GPU.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return the best available torch device as a string.

    Imports torch lazily. This module is imported by the API layer at startup,
    and importing torch costs several seconds and hundreds of MB of RSS — we
    don't want to pay that just to serve /api/projects. It's only paid when a
    model is actually used.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_info() -> dict:
    """Human-readable device summary, surfaced in the UI so the user knows
    whether a job will take 20 seconds or 20 minutes."""
    try:
        import torch
    except ImportError:
        return {"available": False, "device": "cpu", "name": "PyTorch not installed"}

    if not torch.cuda.is_available():
        return {
            "available": True,
            "device": "cpu",
            "name": "CPU",
            "total_vram_gb": None,
            # A CPU fallback is real but slow — Grounding DINO on CPU is minutes
            # per image rather than seconds. Saying so up front is kinder than
            # letting someone queue 500 images and wonder why nothing happens.
            "note": "No CUDA GPU detected. Inference will be very slow.",
        }

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    return {
        "available": True,
        "device": "cuda",
        "name": props.name,
        "total_vram_gb": round(total_gb, 1),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def vram_snapshot() -> dict:
    """Current VRAM usage in GB. Returns zeros on CPU.

    `allocated` is what torch currently holds in live tensors. `reserved` is
    what torch has claimed from the driver — it caches freed blocks rather than
    returning them, so reserved is almost always higher than allocated and is
    the number that actually matters for "will the next model fit".
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}

        return {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            ),
        }
    except ImportError:
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "total_gb": 0.0}


def empty_cache() -> None:
    """Return torch's cached VRAM blocks to the driver.

    Normally an anti-pattern — torch's caching allocator exists precisely so it
    can reuse blocks without round-tripping the driver, and calling this in a
    hot loop makes things slower.

    Here it's necessary. With 4 GB total, unloading Grounding DINO must actually
    free the memory before SAM loads, or Grounded SAM OOMs. Deleting the Python
    object drops the reference but leaves the block in torch's cache, where the
    NEXT model's allocation cannot use it if it needs a differently-shaped
    block. This is called on unload, not per inference.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
