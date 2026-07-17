#!/usr/bin/env python3
"""
One-click development launcher.

Starts the FastAPI backend and the Vite frontend together, waits until both are
actually serving, then opens the browser. Ctrl+C stops both cleanly.

Run it directly, or double-click `start.bat` in the repo root.

    python scripts/dev.py
    python scripts/dev.py --no-browser
    python scripts/dev.py --setup-only

DESIGN NOTES
------------
Why Python instead of doing this all in the .bat file? Batch is genuinely bad at
the four things this script needs:
  1. Killing a *process tree* (uvicorn --reload and vite both spawn children —
     killing the parent alone orphans them and leaves the ports occupied).
  2. Interleaving two log streams with readable prefixes.
  3. Polling an HTTP endpoint to know when a server is *ready* rather than
     merely *launched*.
  4. Reporting a useful error when something is missing.
Python is already a hard prerequisite for the backend, so leaning on it costs
nothing. The .bat is a 3-line shim that exists only so this is double-clickable.

This script uses the STANDARD LIBRARY ONLY. It has to run before the venv exists
and before anything is installed, so it cannot import requests, rich, or even
our own app code.
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
import urllib.request
import webbrowser
from pathlib import Path

# --- Paths ------------------------------------------------------------------
# Resolved from this file's location, never from the current working directory,
# so double-clicking the .bat (which may start anywhere) still works.
#   dev.py -> scripts/ -> <repo root>
ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
VENV = BACKEND / "venv"

IS_WINDOWS = os.name == "nt"

API_PORT = 8000
WEB_PORT = 5173
API_HEALTH_URL = f"http://127.0.0.1:{API_PORT}/api/health"
WEB_URL = f"http://localhost:{WEB_PORT}"


# --- Terminal output --------------------------------------------------------
class C:
    """ANSI codes. Muted palette, matching the app's own restraint."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


class G:
    """Glyphs. Populated by setup_console() with a plain-ASCII fallback."""

    OK = "OK"
    FAIL = "X"
    STEP = "->"
    WARN = "!"
    PIPE = "|"
    DASH = "-"


def setup_console() -> None:
    """Make the console safe for colour and non-ASCII output.

    Two distinct Windows problems, both of which bite a double-clicked .bat:

    1. ENCODING. Python defaults stdout to the legacy ANSI codepage (cp1252
       here), which cannot encode characters like ✓ or →. Printing one raises
       UnicodeEncodeError — and since our error path prints ✗, the launcher
       would crash while reporting a crash. We force UTF-8 on the stream and
       switch the console's output codepage to match, then verify it actually
       took and fall back to ASCII glyphs if not. Never assume; check.

    2. ANSI. Windows Terminal handles escape codes natively, but legacy
       conhost.exe needs the virtual-terminal flag set explicitly, or colour
       codes print as literal garbage.
    """
    # 1a. Force the Python-side encoding.
    #
    # line_buffering=True matters as much as the encoding. Python block-buffers
    # stdout when it isn't a TTY (piping to a log file, running under a CI or
    # task runner). The child-log pump flushes explicitly, but this script's own
    # status lines would sit in the buffer — so "Frontend ready" appears only
    # once some later output happens to fill it, making a working launch look
    # like a hang.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

    if IS_WINDOWS:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # 1b. Tell the console itself to interpret bytes as UTF-8 (chcp
            #     65001). Without this, UTF-8 bytes render as mojibake.
            kernel32.SetConsoleOutputCP(65001)

            # 2. Enable ANSI escape processing.
            STD_OUTPUT_HANDLE = -11
            ENABLE_PROCESSED_OUTPUT = 0x0001
            ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(
                kernel32.GetStdHandle(STD_OUTPUT_HANDLE),
                ENABLE_PROCESSED_OUTPUT
                | ENABLE_WRAP_AT_EOL_OUTPUT
                | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            )
        except Exception:
            # Not fatal — worst case is monochrome ASCII, which still works.
            pass

    # 1c. Verify rather than assume. If the stream still can't carry these
    #     characters, keep the ASCII defaults.
    try:
        probe = "✓✗→│—"
        probe.encode(sys.stdout.encoding or "ascii")
        G.OK, G.FAIL, G.STEP, G.WARN, G.PIPE, G.DASH = "✓", "✗", "→", "!", "│", "—"
    except Exception:
        pass


def step(msg: str) -> None:
    print(f"{C.CYAN}{G.STEP}{C.RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{C.GREEN}{G.OK}{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}{G.WARN}{C.RESET} {msg}")


def die(msg: str, hint: str | None = None) -> None:
    """Print an actionable error and exit.

    Every failure path in this script routes through here, because the whole
    point of a launcher is that a newcomer gets a fix, not a stack trace.
    """
    print(f"\n{C.RED}{C.BOLD}{G.FAIL} {msg}{C.RESET}")
    if hint:
        print(f"  {C.DIM}{hint}{C.RESET}")
    sys.exit(1)


# --- Environment helpers ----------------------------------------------------
def venv_python() -> Path:
    """Path to the interpreter inside the backend venv."""
    return VENV / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def port_in_use(port: int) -> bool:
    """True if anything is listening on that port, over IPv4 *or* IPv6.

    Checking both families is not pedantry. On Windows, Vite binds to [::1]
    (IPv6 loopback) only — an IPv4-only probe of 127.0.0.1 reports the port as
    free while the dev server is very much running. The launcher would then
    happily start a second Vite, which dies with an opaque EADDRINUSE from deep
    inside node. getaddrinfo() enumerates every address 'localhost' resolves to,
    so we test all of them.
    """
    try:
        candidates = socket.getaddrinfo(
            "localhost", port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror:
        return False

    for family, socktype, proto, _canonname, sockaddr in candidates:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.settimeout(0.4)
                if s.connect_ex(sockaddr) == 0:
                    return True
        except OSError:
            continue
    return False


def reap_stale_servers() -> int:
    """Kill dev servers left over from a previous launcher run. Returns the count.

    WHY THIS IS NECESSARY (learned the hard way, twice)
    --------------------------------------------------
    A previous session's uvicorn can survive — the launcher gets killed without
    its `finally` running, or a task is torn down without a signal. The port
    check alone does NOT catch it, because a leftover *reloader* isn't holding a
    port; it's just sitting there watching files.

    Then you edit a file and both reloaders wake up. Each kills and respawns its
    own worker, and the two workers race for port 8000. Sometimes the ORPHAN
    wins — so the server answering your requests is running last session's code.
    WatchFiles cheerfully prints "Reloading..." the whole time.

    The symptom is maddening and specific: you fix a bug, the reload message
    appears, and the API keeps returning the old behaviour. You then debug
    perfectly correct code.

    Scoped to THIS repo's processes by matching the command line against our own
    paths, so it can't touch an unrelated Python server you have running.
    """
    if not IS_WINDOWS:
        return 0

    try:
        # Win32_Process gives us the full command line, which is the only
        # reliable way to tell OUR uvicorn from anyone else's.
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -like '*uvicorn*' -or $_.CommandLine -like '*vite*' } | "
                "ForEach-Object { \"$($_.ProcessId)|$($_.CommandLine)\" }",
            ],
            capture_output=True,
            text=True,
            timeout=25,
        ).stdout
    except Exception:
        return 0

    # Match on the repo root, normalised — the command line may use either slash.
    root_variants = {str(ROOT).lower(), str(ROOT).lower().replace("\\", "/")}

    killed = 0
    for line in out.splitlines():
        if "|" not in line:
            continue
        pid_str, _, cmdline = line.partition("|")
        pid_str = pid_str.strip()
        if not pid_str.isdigit():
            continue
        low = cmdline.lower()
        if not any(root in low for root in root_variants):
            continue  # someone else's server — leave it alone
        if int(pid_str) == os.getpid():
            continue
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", pid_str], capture_output=True
        )
        killed += 1

    if killed:
        # Give the OS a moment to actually release the ports before preflight
        # checks them, or we'd report a conflict against a process we just
        # killed.
        time.sleep(1.5)
    return killed


def pid_on_port(port: int) -> str | None:
    """Best-effort lookup of which PID holds a port, so the error can name it."""
    if not IS_WINDOWS:
        return None
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "TCP" and "LISTENING" in parts:
                if parts[1].endswith(f":{port}"):
                    return parts[-1]
    except Exception:
        pass
    return None


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def needs_install(source: Path, stamp: Path) -> bool:
    """True if `source` changed since the last successful install.

    The stamp records the hash of the dependency manifest. Storing it INSIDE the
    install target (venv/, node_modules/) means it's already gitignored, and
    deleting that directory correctly invalidates the stamp — no stale marker
    claiming an install exists when it doesn't.
    """
    if not stamp.exists():
        return True
    try:
        return stamp.read_text(encoding="utf-8").strip() != file_hash(source)
    except Exception:
        return True


def write_stamp(source: Path, stamp: Path) -> None:
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(file_hash(source), encoding="utf-8")


def run_checked(cmd: list[str], cwd: Path, desc: str) -> None:
    """Run a setup command, streaming its output. Exit with context on failure."""
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        die(f"{desc} failed (exit {result.returncode})", "See the output above.")


# --- Preflight --------------------------------------------------------------
def preflight() -> str:
    """Verify the toolchain exists before touching anything. Returns npm path."""
    if sys.version_info < (3, 10):
        die(
            f"Python 3.10+ required, found {sys.version.split()[0]}",
            "Install a newer Python from https://python.org and re-run.",
        )

    npm = shutil.which("npm")
    if not npm:
        die(
            "npm not found on PATH",
            "Install Node.js 18+ from https://nodejs.org, then open a NEW terminal.",
        )

    return npm


def check_ports() -> None:
    """Fail fast if either port is occupied.

    Kept separate from preflight() because it's only relevant when we're about
    to bind: `--setup-only` installs dependencies and has no business demanding
    free ports.

    Starting anyway would either crash with an opaque bind error or — worse —
    leave the browser pointed at a stale server running different code, which is
    a genuinely maddening thing to debug.
    """
    for port, name in ((API_PORT, "backend"), (WEB_PORT, "frontend")):
        if port_in_use(port):
            pid = pid_on_port(port)
            hint = (
                f"Another {name} is already running on port {port}."
                if pid is None
                else f"PID {pid} is holding port {port}. Stop it with:  taskkill /F /PID {pid}"
            )
            die(f"Port {port} is already in use", hint)


# --- Setup ------------------------------------------------------------------
def ensure_backend() -> None:
    """Create the venv and install core deps if missing or out of date."""
    if not venv_python().exists():
        step("Creating Python virtual environment (first run only)…")
        run_checked([sys.executable, "-m", "venv", str(VENV)], BACKEND, "venv creation")
        ok("Virtual environment created")

    req = BACKEND / "requirements.txt"
    stamp = VENV / ".requirements.stamp"
    if needs_install(req, stamp):
        step("Installing backend dependencies…")
        run_checked(
            [str(venv_python()), "-m", "pip", "install", "-q", "--upgrade", "pip"],
            BACKEND,
            "pip upgrade",
        )
        run_checked(
            [str(venv_python()), "-m", "pip", "install", "-r", str(req)],
            BACKEND,
            "pip install",
        )
        write_stamp(req, stamp)
        ok("Backend dependencies installed")
    else:
        ok("Backend dependencies up to date")


def ensure_frontend(npm: str) -> None:
    """Install node_modules if missing or if the lockfile changed."""
    lock = FRONTEND / "package-lock.json"
    stamp = FRONTEND / "node_modules" / ".lock.stamp"

    if not (FRONTEND / "node_modules").exists() or needs_install(lock, stamp):
        step("Installing frontend dependencies (this can take a minute)…")
        # `npm ci` is the right call when a lockfile exists: it installs exactly
        # what's pinned and is reproducible, where `npm install` may quietly
        # update the lockfile. Fall back to `install` if ci can't run.
        result = subprocess.run([npm, "ci"], cwd=str(FRONTEND))
        if result.returncode != 0:
            warn("npm ci failed, falling back to npm install")
            run_checked([npm, "install"], FRONTEND, "npm install")
        write_stamp(lock, stamp)
        ok("Frontend dependencies installed")
    else:
        ok("Frontend dependencies up to date")


# --- Process management -----------------------------------------------------
class Service:
    """A managed child process with prefixed log streaming."""

    def __init__(self, name: str, colour: str, cmd: list[str], cwd: Path, env: dict):
        self.name = name
        self.colour = colour
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge streams — ordering stays readable
            text=True,
            bufsize=1,  # line buffered, so logs appear as they happen
            encoding="utf-8",
            errors="replace",  # never crash the launcher on odd bytes
            # A new process group lets us signal this tree without killing
            # ourselves, and stops the console's Ctrl+C from racing our own
            # orderly shutdown.
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
        )
        # Daemon thread: pumping output must not keep the interpreter alive at
        # exit if the child is wedged.
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Forward child output, tagged so two streams stay tellable apart."""
        assert self.proc and self.proc.stdout
        tag = f"{self.colour}{self.name:>4}{C.RESET} {C.GRAY}{G.PIPE}{C.RESET} "
        for line in self.proc.stdout:
            print(tag + line.rstrip(), flush=True)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Kill the whole process tree.

        `taskkill /T` is essential here, not paranoia: `uvicorn --reload` runs a
        supervisor that forks the real server, and `vite` spawns esbuild. Calling
        proc.terminate() kills only the parent and leaves the child holding the
        port — so the next launch fails the preflight port check with no obvious
        culprit.
        """
        if not self.alive:
            return
        assert self.proc
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True,
            )
        else:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def wait_until_ready(url: str, service: Service, timeout: float = 90) -> bool:
    """Poll `url` until it answers, or the service dies, or we time out.

    Waiting for a real HTTP response — rather than just sleeping a few seconds —
    is what makes the browser open on a working page instead of a connection
    error. The `service.alive` check means a crashed server reports immediately
    rather than after the full timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not service.alive:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


# --- Main -------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start the CV Platform backend and frontend together."
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open the browser on start"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="install dependencies and exit without starting servers",
    )
    args = parser.parse_args()

    # Must run before ANY output — it's what makes printing safe at all.
    setup_console()
    print(f"\n{C.BOLD}CV Platform{C.RESET} {C.DIM}{G.DASH} local development{C.RESET}\n")

    npm = preflight()
    ensure_backend()
    ensure_frontend(npm)

    if args.setup_only:
        ok("Setup complete")
        return 0

    # Everything below here is about STARTING servers, so nothing above may
    # touch running ones. `--setup-only` installs dependencies and exits — it
    # has no business killing an app you have running in another terminal, which
    # is exactly what it did while the reap sat above this guard.
    #
    # Clear out any servers a previous run left behind BEFORE checking ports:
    # an orphaned reloader silently serves stale code, and the port check can't
    # see it because an idle reloader holds no port. See reap_stale_servers().
    stale = reap_stale_servers()
    if stale:
        warn(f"Stopped {stale} leftover dev server process(es) from a previous run")

    check_ports()

    # PYTHONUNBUFFERED forces uvicorn's logs through immediately. Without it,
    # Python block-buffers when stdout is a pipe (as it is here, not a TTY), and
    # log lines arrive in delayed clumps — which looks exactly like a hang.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "FORCE_COLOR": "1"}

    api = Service(
        "api",
        C.BLUE,
        [
            str(venv_python()),
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--port",
            str(API_PORT),
        ],
        BACKEND,
        env,
    )
    web = Service("web", C.CYAN, [npm, "run", "dev"], FRONTEND, env)

    services = [api, web]

    try:
        print()
        step("Starting backend…")
        api.start()
        if not wait_until_ready(API_HEALTH_URL, api, timeout=60):
            die(
                "Backend failed to start",
                "Check the [api] output above for the traceback.",
            )
        ok(f"Backend ready  {C.DIM}{API_HEALTH_URL}{C.RESET}")

        step("Starting frontend…")
        web.start()
        if not wait_until_ready(WEB_URL, web, timeout=90):
            die("Frontend failed to start", "Check the [web] output above.")
        ok(f"Frontend ready {C.DIM}{WEB_URL}{C.RESET}")

        print(
            f"\n  {C.BOLD}App{C.RESET}       {WEB_URL}"
            f"\n  {C.DIM}API docs  http://localhost:{API_PORT}/docs{C.RESET}"
            f"\n\n  {C.DIM}Both servers hot-reload on save."
            f" Press Ctrl+C to stop.{C.RESET}\n"
        )

        if not args.no_browser:
            webbrowser.open(WEB_URL)

        # Supervise. If either process exits on its own, something broke — tear
        # the other one down rather than leaving a half-running app.
        while True:
            for svc in services:
                if not svc.alive:
                    code = svc.proc.returncode if svc.proc else "?"
                    warn(f"{svc.name} exited unexpectedly (code {code}) — shutting down")
                    return 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\n{C.DIM}Stopping…{C.RESET}")
        return 0
    finally:
        # Runs on every exit path — Ctrl+C, crash, or die(). Without this,
        # orphaned servers keep the ports bound and the next launch fails.
        for svc in services:
            svc.stop()
        ok("Stopped")


if __name__ == "__main__":
    sys.exit(main())
