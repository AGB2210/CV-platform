"""
Production launcher — run a RELEASE build of the app.

WHY THIS EXISTS SEPARATELY FROM dev.py
--------------------------------------
`scripts/dev.py` runs two processes: uvicorn for the API and Vite for the UI.
That needs Node, `frontend/node_modules`, and the frontend SOURCE — none of
which exist in a release artifact. A release ships the frontend already
compiled to `frontend/dist`, and FastAPI serves it from the same origin (see
`app/main.py`), so there is exactly one process and no Node on the machine at
all.

Trying to make one launcher cover both would mean a script that behaves
differently depending on files it happens to find, which is harder to reason
about than two short scripts with one job each.

WHAT IT STILL DOES
------------------
The same first-run setup dev.py does, minus everything frontend: find Python,
create `backend/venv`, install `requirements.txt`, then start uvicorn WITHOUT
`--reload` (reload watches the filesystem and restarts on write — useful while
editing, pure overhead for a release).

Stdlib only, because it has to run before the venv it creates exists.

    python scripts/serve.py                # http://127.0.0.1:8000
    python scripts/serve.py --port 9000
    python scripts/serve.py --host 0.0.0.0 # see the warning below
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
VENV = BACKEND / "venv"
DIST = ROOT / "frontend" / "dist"


def venv_python() -> Path:
    """Interpreter inside the venv, per-platform."""
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def die(msg: str, hint: str | None = None) -> None:
    print(f"\n  [X] {msg}")
    if hint:
        print(f"      {hint}")
    sys.exit(1)


def ensure_backend() -> Path:
    """Create the venv and install dependencies if they're missing."""
    python = venv_python()
    if not python.exists():
        print("  Creating the Python environment (first run only)...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        python = venv_python()

    # Cheap check: can the app be imported? If yes, dependencies are in place.
    probe = subprocess.run(
        [str(python), "-c", "import fastapi, uvicorn"],
        cwd=BACKEND,
        capture_output=True,
    )
    if probe.returncode != 0:
        print("  Installing dependencies (first run only, a minute or two)...")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--quiet", "-r", "requirements.txt"],
            cwd=BACKEND,
            check=True,
        )
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind. Defaults to loopback — see the warning below.",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("\n  CV Platform\n")

    if not BACKEND.is_dir():
        die("backend/ not found.", f"Run this from the app folder. Looked in {ROOT}")

    if not DIST.is_dir():
        die(
            "No built frontend found at frontend/dist.",
            "This script runs a RELEASE build. For development use start.bat, "
            "which runs the Vite dev server instead.",
        )

    # THE APP HAS NO AUTHENTICATION OF ANY KIND. On loopback that's fine — only
    # this machine can reach it. Binding to 0.0.0.0 publishes an unauthenticated
    # upload-and-delete API to the whole network, so it warns rather than
    # quietly obliging.
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  [!] Binding to {args.host}, not just this machine.")
        print("      The app has NO login. Anyone who can reach this port can")
        print("      read, upload and delete every project. Only do this on a")
        print("      network you trust.\n")

    python = ensure_backend()

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"  Starting...  {url}\n")
    if not args.no_browser:
        # Fire-and-forget: if no browser is configured this must not be fatal.
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    # No --reload: it watches the filesystem and restarts on write, which is
    # useful while editing and pure overhead for a release.
    try:
        return subprocess.run(
            [
                str(python),
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            cwd=BACKEND,
        ).returncode
    except KeyboardInterrupt:
        print("\n  Stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
