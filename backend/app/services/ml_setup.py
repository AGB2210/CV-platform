"""
Install the heavy ML stack on demand, from inside the running app.

WHY THIS EXISTS
---------------
torch + transformers + ultralytics are several gigabytes and only two features
need them (Auto-annotate, Train). Installing them at first launch would make
every user — including someone who just wants to look at the UI — wait for a
multi-gigabyte download they may never use. So the core app installs in about a
minute, and this installs the rest the first time a feature actually needs it,
with progress the frontend can show. Nothing about it touches a command line.

THE HARD PART IS torch, AND IT IS HANDLED HERE
----------------------------------------------
The correct torch build depends on the machine: a CUDA build for an NVIDIA GPU,
a CPU build otherwise, and the CUDA build must not be newer than the installed
driver supports. Picking wrong is either a multi-gigabyte CPU download onto a
GPU box (slow training, silently) or a driver-mismatch import error. So this
detects the GPU via `nvidia-smi`, reads the CUDA version the driver reports, and
selects the matching PyTorch wheel index — the judgement a human would otherwise
have to make by reading pytorch.org.

STATE
-----
The install runs once, process-wide, in a background thread. Its progress lives
in a module-level object rather than the database: it is not per-project, it
outlives no restart worth persisting across, and `is_installed()` re-checks the
real environment every call, so the truth is never the cached flag. If the
server is killed mid-install, the next status check simply reports "not
installed" and the user clicks again.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: The packages that must import for the ML features to work. torch is the big
#: one; the other two are what Auto-annotate (transformers) and Train
#: (ultralytics) each additionally require. Checked by find_spec, which does not
#: pay torch's multi-hundred-MB import cost just to answer "is it there?".
_REQUIRED = ("torch", "transformers", "ultralytics")

_BACKEND = Path(__file__).resolve().parent.parent.parent
_REQUIREMENTS_ML = _BACKEND / "requirements-ml.txt"

# PyTorch publishes a wheel index per CUDA version. Highest first: we pick the
# newest index the driver can support. Kept short on purpose — these are the
# builds torch actually ships; an unknown driver falls back to the lowest.
_TORCH_CUDA_INDEXES: tuple[tuple[tuple[int, int], str], ...] = (
    ((12, 6), "https://download.pytorch.org/whl/cu126"),
    ((12, 4), "https://download.pytorch.org/whl/cu124"),
    ((12, 1), "https://download.pytorch.org/whl/cu121"),
    ((11, 8), "https://download.pytorch.org/whl/cu118"),
)
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass
class _InstallState:
    """Progress of the one install, shared between the worker and the pollers."""

    status: str = "idle"  # idle | running | done | failed
    phase: str = ""  # human-readable current step
    error: str | None = None
    #: The tail of pip's output, so a stuck install is diagnosable from the UI.
    log_tail: list[str] = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None


_state = _InstallState()
_lock = threading.Lock()
_MAX_LOG_LINES = 40


def is_installed() -> bool:
    """Whether every required ML package is importable right now.

    The single source of truth for "are the ML features available". Re-derived
    from the environment on every call rather than trusting a flag, so it is
    correct after an install completes, after a manual `pip install`, or after a
    half-finished install left some packages present and others not.
    """
    return all(importlib.util.find_spec(name) is not None for name in _REQUIRED)


def status() -> dict:
    """Everything the frontend needs to decide what to show."""
    with _lock:
        install = {
            "status": _state.status,
            "phase": _state.phase,
            "error": _state.error,
            "log_tail": list(_state.log_tail),
        }
    return {
        "installed": is_installed(),
        "install": install,
    }


def _driver_cuda_version() -> tuple[int, int] | None:
    """The CUDA version the NVIDIA driver supports, from `nvidia-smi`, or None.

    `nvidia-smi` prints the driver's max CUDA in its header, and the label has
    DRIFTED between driver generations: older builds say "CUDA Version: 12.6",
    newer ones "CUDA UMD Version: 13.3". Both are matched. That number is the
    highest CUDA a torch build may use on this driver — not any installed toolkit
    — which is exactly what wheel selection needs. No `nvidia-smi` (or a
    non-NVIDIA machine) returns None and we install the CPU build.

    Measured on a real machine: matching only "CUDA Version:" quietly missed a
    driver reporting "CUDA UMD Version:", so a CUDA box was about to be handed
    the CPU build — the precise silent failure this function exists to prevent.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:  # noqa: BLE001 — any failure means "treat as no GPU"
        return None
    # "CUDA Version: X.Y" or "CUDA UMD Version: X.Y" — an optional word between.
    match = re.search(r"CUDA\s+(?:[A-Za-z]+\s+)?Version:\s*(\d+)\.(\d+)", out)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _torch_index_url() -> str:
    """The PyTorch wheel index to install from, chosen for this machine."""
    driver = _driver_cuda_version()
    if driver is None:
        return _TORCH_CPU_INDEX
    for version, url in _TORCH_CUDA_INDEXES:
        if version <= driver:
            return url
    # A GPU whose driver is older than any index we know: the CPU build still
    # runs (just without acceleration), which beats an install that won't import.
    return _TORCH_CPU_INDEX


def install_plan() -> dict:
    """What an install WOULD do, without doing it. Drives the confirm UI."""
    driver = _driver_cuda_version()
    index = _torch_index_url()
    return {
        "gpu_detected": driver is not None,
        "driver_cuda": f"{driver[0]}.{driver[1]}" if driver else None,
        "torch_build": "cpu" if index == _TORCH_CPU_INDEX else index.rsplit("/", 1)[-1],
        "torch_index_url": index,
    }


def _append_log(line: str) -> None:
    with _lock:
        _state.log_tail.append(line)
        if len(_state.log_tail) > _MAX_LOG_LINES:
            del _state.log_tail[: len(_state.log_tail) - _MAX_LOG_LINES]


def _set(status: str | None = None, phase: str | None = None, error: str | None = None) -> None:
    with _lock:
        if status is not None:
            _state.status = status
        if phase is not None:
            _state.phase = phase
        if error is not None:
            _state.error = error


def _pip(args: list[str], phase: str) -> None:
    """Run one pip command, streaming its output into the log. Raises on failure."""
    _set(phase=phase)
    logger.info("ml_setup: %s", phase)
    proc = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", *args],
        cwd=str(_BACKEND),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            _append_log(line)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"pip exited {code} during: {phase}")


def _verify_import() -> None:
    """Confirm torch imports in a FRESH process, retrying a transient block.

    A fresh subprocess rather than importing here: importing torch into the web
    process would pin hundreds of MB for a check, and the real features import it
    lazily when actually used. The retry is for Windows Smart App Control, which
    blocks a freshly written unsigned DLL on first load while it queries a
    reputation service, then allows it seconds later — a one-time hiccup, not a
    real failure, so retrying beats surfacing it as one.
    """
    last = ""
    for attempt in range(3):
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            cwd=str(_BACKEND),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            _append_log(f"torch import OK (cuda available: {result.stdout.strip()})")
            return
        last = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        last = last[0]
        _append_log(f"verify attempt {attempt + 1} failed: {last}")
        time.sleep(3)
    raise RuntimeError(f"torch installed but would not import: {last}")


def _run_install() -> None:
    try:
        index = _torch_index_url()
        # torch FIRST, from the machine-specific index. Installing it before the
        # rest means the correct build is resident before anything can pull in a
        # generic one as a dependency.
        _pip(
            ["torch", "torchvision", "--index-url", index],
            phase=f"Installing PyTorch ({'CPU' if index == _TORCH_CPU_INDEX else index.rsplit('/', 1)[-1]})",
        )
        # Then everything else — transformers, ultralytics and friends. torch is
        # already satisfied, so this leaves the machine-specific build alone.
        _pip(
            ["-r", str(_REQUIREMENTS_ML)],
            phase="Installing transformers, ultralytics and dependencies",
        )
        _set(phase="Verifying")
        _verify_import()
        _set(status="done", phase="Ready")
        with _lock:
            _state.finished_at = time.time()
        logger.info("ml_setup: install complete")
    except Exception as exc:  # noqa: BLE001 — background thread, nothing to bubble to
        logger.exception("ml_setup: install failed")
        _set(status="failed", error=str(exc))
        with _lock:
            _state.finished_at = time.time()


class MlSetupError(Exception):
    """Domain error — the route maps it to an HTTP status."""


def start_install() -> None:
    """Begin the install in the background. Idempotent-ish and guarded.

    Refuses if the stack is already present (nothing to do) or an install is
    already running (so two clicks don't launch two pip processes fighting over
    the same site-packages).
    """
    if is_installed():
        raise MlSetupError("The ML dependencies are already installed.")
    with _lock:
        if _state.status == "running":
            raise MlSetupError("An install is already in progress.")
        _state.status = "running"
        _state.phase = "Starting"
        _state.error = None
        _state.log_tail = []
        _state.started_at = time.time()
        _state.finished_at = None
    threading.Thread(target=_run_install, name="ml-install", daemon=True).start()
