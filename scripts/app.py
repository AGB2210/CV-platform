#!/usr/bin/env python3
"""
The launcher. One click, one process, the whole app.

Double-click `start.bat`, or run this directly. It does everything needed to go
from a bare checkout (or an unpacked release) to a working website, and then
serves it:

    python scripts/app.py
    python scripts/app.py --no-browser
    python scripts/app.py --setup-only     # install/build, then exit
    python scripts/app.py --port 9000

WHAT "EVERYTHING" MEANS
-----------------------
  1. Create backend/venv and install the CORE dependencies (fast — seconds).
  2. Build the frontend to frontend/dist IF this is a source checkout and the
     build is missing or stale. A release ships dist already built and has no
     frontend source, so this step is skipped there and Node is never needed.
  3. Serve the built frontend AND the API from ONE uvicorn process on ONE port
     (FastAPI serves dist at / — see app/main.py). No second dev server, no
     hot-reload: this is deliberately the same thing a user runs, so working on
     the app means experiencing exactly what they experience.

The heavy ML stack (torch, transformers, ultralytics — several GB) is NOT
installed here. It installs itself the first time someone opens a feature that
needs it (Auto-annotate or Train), from inside the running app, with progress
shown on the page. That keeps first launch to about a minute instead of a long
multi-gigabyte download nobody asked for yet — and it still never touches a
command line. See app/services/ml_setup.py.

STANDARD LIBRARY ONLY. This runs before the venv exists, so it cannot import
anything that isn't shipped with Python itself.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# Resolved from this file, never the working directory, so double-clicking the
# .bat (which can start anywhere) still finds everything.
#   app.py -> scripts/ -> <repo root>
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
FRONTEND_SRC = FRONTEND / "src"
DIST = FRONTEND / "dist"
VENV = BACKEND / "venv"

IS_WINDOWS = os.name == "nt"
DEFAULT_PORT = 8000


# --- Console ----------------------------------------------------------------


def setup_console() -> None:
    """Make the console safe for our output on Windows.

    Two separate problems, both Windows-only:
      - The default console codepage is cp1252, which cannot encode the ✓ / →
        glyphs below — and when output is piped (a captured run, a CI log) Python
        picks cp1252 for the pipe too, so a status line crashes the launcher with
        UnicodeEncodeError. Reconfiguring to UTF-8 with errors="replace" makes
        the worst case a wrong glyph, never a crash.
      - ANSI colour codes are inert until virtual-terminal processing is enabled
        on the console handle.
    """
    if not IS_WINDOWS:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING on the console's stdout handle.
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001 — colour is a nicety, never fatal
        pass


_C = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "dim": "\033[2m",
    "reset": "\033[0m",
}


def step(msg: str) -> None:
    print(f"{_C['cyan']}→{_C['reset']} {msg}")


def ok(msg: str) -> None:
    print(f"{_C['green']}✓{_C['reset']} {msg}")


def warn(msg: str) -> None:
    print(f"{_C['yellow']}!{_C['reset']} {msg}")


def die(msg: str, hint: str | None = None) -> None:
    print(f"\n{_C['red']}[X]{_C['reset']} {msg}")
    if hint:
        print(f"    {hint}")
    sys.exit(1)


# --- Small helpers ----------------------------------------------------------


def venv_python() -> Path:
    return VENV / ("Scripts" if IS_WINDOWS else "bin") / ("python.exe" if IS_WINDOWS else "python")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _needs_install(source: Path, stamp: Path) -> bool:
    """True unless `stamp` records the exact current hash of `source`.

    Hash, not mtime: `git checkout` and a fresh clone rewrite mtimes without
    changing content, and would otherwise force a needless reinstall every time.
    """
    if not stamp.exists():
        return True
    return stamp.read_text(encoding="utf-8").strip() != _hash_file(source)


def _write_stamp(source: Path, stamp: Path) -> None:
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(_hash_file(source), encoding="utf-8")


def _run(cmd: list[str], cwd: Path, desc: str) -> None:
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        die(f"{desc} failed (exit {result.returncode}).")


def _find_npm() -> str | None:
    """Locate npm, or None. Only needed when the frontend must be BUILT."""
    for name in ("npm.cmd", "npm") if IS_WINDOWS else ("npm",):
        found = shutil.which(name)
        if found:
            return found
    return None


# --- Reap a stale server ----------------------------------------------------


def reap_stale_server(port: int) -> None:
    """Kill a uvicorn of OURS left holding the port by a previous run.

    Much simpler than the old dev launcher's reaper: there is one process and no
    `--reload`, so there is no reloader parent and no orphaned multiprocessing
    worker to chase — just, at most, one stale uvicorn. Scoped to our own repo
    path so an unrelated server on the same port is never touched.
    """
    if not IS_WINDOWS or not port_in_use(port):
        return
    root_low = str(ROOT).lower()
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' } | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Csv -NoTypeInformation",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except Exception:  # noqa: BLE001
        return
    for line in out.splitlines():
        low = line.lower()
        if "uvicorn" in low and root_low in low.replace("/", "\\"):
            pid = line.split(",", 1)[0].strip('"')
            if pid.isdigit() and int(pid) != os.getpid():
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
    time.sleep(1.5)


# --- Setup steps ------------------------------------------------------------


def ensure_python_ok() -> None:
    if sys.version_info < (3, 10):
        die(
            f"Python 3.10+ is required; this is {sys.version.split()[0]}.",
            "Install a newer Python from https://www.python.org/downloads/",
        )
    if not BACKEND.is_dir():
        die("backend/ not found.", f"Run this from the app folder. Looked in {ROOT}")


def ensure_backend() -> Path:
    """Create the venv and install the CORE dependencies if needed."""
    python = venv_python()
    if not python.exists():
        step("Creating the Python environment (first run only)…")
        _run([sys.executable, "-m", "venv", str(VENV)], BACKEND, "venv creation")
        python = venv_python()
        ok("Environment created")

    req = BACKEND / "requirements.txt"
    stamp = VENV / ".requirements.stamp"
    if _needs_install(req, stamp):
        step("Installing core dependencies (first run only, ~1 minute)…")
        _run([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"], BACKEND, "pip upgrade")
        _run([str(python), "-m", "pip", "install", "-q", "-r", str(req)], BACKEND, "pip install")
        _write_stamp(req, stamp)
        ok("Core dependencies installed")
    else:
        ok("Core dependencies up to date")
    return python


def _dist_is_stale() -> bool:
    """Does frontend/dist need rebuilding from source?

    Only meaningful in a SOURCE checkout. If there is no frontend/src at all this
    is a release build: dist ships prebuilt and there is nothing to compare.
    """
    if not FRONTEND_SRC.is_dir():
        return False  # release: never build
    index = DIST / "index.html"
    if not index.exists():
        return True  # nothing built yet
    built_at = index.stat().st_mtime
    # Rebuild if any input is newer than the built output. Covers the source
    # tree and the files that change how it is built.
    inputs = list(FRONTEND_SRC.rglob("*"))
    for extra in ("index.html", "vite.config.ts", "package.json", "index.css"):
        inputs.append(FRONTEND / extra)
    return any(p.is_file() and p.stat().st_mtime > built_at for p in inputs)


def ensure_frontend_built() -> None:
    """Build frontend/dist when working from source and it is stale.

    A release has no source and ships dist prebuilt, so this is a no-op there and
    Node is never required. Only a source checkout that has changed since its
    last build pays the (sub-second) build cost.
    """
    if not FRONTEND_SRC.is_dir():
        if not (DIST / "index.html").exists():
            die(
                "No built frontend, and no frontend source to build it from.",
                "This looks like a broken release. Re-download it.",
            )
        ok("Frontend ready (prebuilt)")
        return

    if not _dist_is_stale():
        ok("Frontend up to date")
        return

    npm = _find_npm()
    if npm is None:
        die(
            "The frontend needs building, but Node/npm was not found on PATH.",
            "Install Node 18+ from https://nodejs.org/, open a NEW terminal, and "
            "run this again. (A downloaded release needs no Node — this only "
            "applies when working from source.)",
        )

    lock = FRONTEND / "package-lock.json"
    node_stamp = FRONTEND / "node_modules" / ".lock.stamp"
    if not (FRONTEND / "node_modules").is_dir() or _needs_install(lock, node_stamp):
        step("Installing frontend build tools (first run only)…")
        result = subprocess.run([npm, "ci"], cwd=str(FRONTEND))
        if result.returncode != 0:
            warn("npm ci failed; falling back to npm install")
            _run([npm, "install"], FRONTEND, "npm install")
        _write_stamp(lock, node_stamp)

    step("Building the interface…")
    _run([npm, "run", "build"], FRONTEND, "frontend build")
    ok("Interface built")


# --- Serve ------------------------------------------------------------------


def _open_when_ready(url: str, health_url: str, proc: subprocess.Popen) -> None:
    """Poll the health endpoint in the background, open the browser when it answers."""
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            return  # server died during startup; main() will report it
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    ok(f"Ready  {url}")
                    try:
                        webbrowser.open(url)
                    except Exception:  # noqa: BLE001
                        pass
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.4)


def serve(python: Path, host: str, port: int, open_browser: bool) -> int:
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{port}"

    # No authentication anywhere in the app. On loopback that is fine — only this
    # machine can reach it. Binding to 0.0.0.0 exposes an unauthenticated
    # upload-and-delete API to the whole network, so warn rather than oblige
    # quietly.
    if host not in ("127.0.0.1", "localhost", "::1"):
        warn(f"Binding to {host}, not just this machine.")
        print("    The app has NO login. Anyone who can reach this port can read,")
        print("    upload and delete every project. Only do this on a trusted network.\n")

    proc = subprocess.Popen(
        [
            str(python),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(BACKEND),
    )

    if open_browser:
        threading.Thread(
            target=_open_when_ready, args=(url, f"{url}/api/health", proc), daemon=True
        ).start()
    else:
        step(f"Serving at {url}")

    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n  Stopping…")
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CV Platform app.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Install dependencies and build the frontend, then exit.",
    )
    args = parser.parse_args()

    setup_console()
    print(f"\n  {_C['dim']}CV Platform{_C['reset']}\n")

    ensure_python_ok()
    reap_stale_server(args.port)
    python = ensure_backend()
    ensure_frontend_built()

    if args.setup_only:
        ok("Setup complete.")
        return 0

    if port_in_use(args.port):
        die(
            f"Port {args.port} is already in use.",
            "Another copy may be running. Close it, or pass --port to use another.",
        )

    print()
    return serve(python, args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
