"""
Shared HuggingFace loading policy for annotators.

ONCE CACHED, A MODEL MUST NOT NEED THE NETWORK.

from_pretrained() phones home to check for updates even when every byte is
already on disk, so anything that upsets that call — offline, HF down, rate
limiting, or a stale auth token — fails a job whose weights are sitting right
there. That happened with Grounding DINO: an expired token in
~/.cache/huggingface/token got a 401, which transformers reports as the deeply
misleading "not a valid model identifier".

For a tool whose premise is local operation, depending on huggingface.co at
inference time is a bug. So: try online (needed for the genuine first
download), and on ANY failure fall back to the cache. Broad except on purpose
— the failure modes are network errors, HTTP errors, and OSError depending on
how deep it got, and the response is identical for all of them.

Every transformers-backed annotator loads through here so the policy exists
once, not re-derived per model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def from_pretrained_with_fallback(
    loader: Callable[..., Any], model_id: str, approx_size: str, **kwargs: Any
) -> Any:
    """`loader.from_pretrained(model_id)` with the offline-cache fallback.

    `loader` is the class or factory (AutoProcessor, a *ForObjectDetection
    class…); `approx_size` is human text for the first-download error
    ("~700 MB").
    """
    try:
        return loader.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — see module docstring
        logger.warning(
            "Could not reach HuggingFace for %s (%s). Falling back to the local cache.",
            model_id,
            exc,
        )
        try:
            return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
        except Exception:
            # Genuinely not cached: the first run does need a download, and
            # saying so beats re-raising a 401 that blames the model id.
            raise RuntimeError(
                f"{model_id} is not in the local cache and HuggingFace could "
                f"not be reached. The first run needs to download {approx_size}. "
                f"If you have an expired HuggingFace token, remove "
                f"~/.cache/huggingface/token — a stale token is rejected with "
                f"401 even for public models, while anonymous access works."
            ) from exc
