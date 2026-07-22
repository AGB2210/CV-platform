"""
ML setup: GPU detection, torch-wheel selection, and the install endpoints.

The install itself downloads gigabytes and is not exercised here — these pin the
LOGIC around it, which is where the machine-specific judgement lives and where a
wrong answer is silent (a CPU build on a GPU box trains slowly and says nothing).
"""

from __future__ import annotations

import pytest

from app.services import ml_setup

# nvidia-smi header text, as different driver generations actually print it.
SMI_CLASSIC = "| NVIDIA-SMI 550.00   Driver Version: 550.00   CUDA Version: 12.4 |"
SMI_UMD = "| NVIDIA-SMI 610.74   KMD Version: 610.74   CUDA UMD Version: 13.3 |"
SMI_OLD = "| NVIDIA-SMI 470.00   Driver Version: 470.00   CUDA Version: 11.4 |"


def _fake_smi(monkeypatch, header: str | None) -> None:
    """Make _driver_cuda_version see a given nvidia-smi, or none at all."""
    if header is None:
        monkeypatch.setattr(ml_setup.shutil, "which", lambda _name: None)
        return
    monkeypatch.setattr(ml_setup.shutil, "which", lambda _name: "nvidia-smi")

    class _Result:
        stdout = header

    monkeypatch.setattr(ml_setup.subprocess, "run", lambda *a, **k: _Result())


def test_reads_classic_cuda_version(monkeypatch):
    _fake_smi(monkeypatch, SMI_CLASSIC)
    assert ml_setup._driver_cuda_version() == (12, 4)


def test_reads_umd_cuda_version(monkeypatch):
    """The format that shipped broken: 'CUDA UMD Version:' must parse too.

    A real driver (610.74) reported this, and matching only 'CUDA Version:'
    returned None, so a CUDA machine was about to be handed the CPU torch build.
    """
    _fake_smi(monkeypatch, SMI_UMD)
    assert ml_setup._driver_cuda_version() == (13, 3)


def test_no_gpu_returns_none(monkeypatch):
    _fake_smi(monkeypatch, None)
    assert ml_setup._driver_cuda_version() is None


def test_index_picks_highest_wheel_not_above_driver(monkeypatch):
    # Driver newer than any known wheel → the newest wheel we have.
    _fake_smi(monkeypatch, SMI_UMD)  # 13.3
    assert ml_setup._torch_index_url().endswith("cu126")

    # Driver between wheels → the highest that is not above it.
    _fake_smi(monkeypatch, SMI_CLASSIC)  # 12.4
    assert ml_setup._torch_index_url().endswith("cu124")


def test_old_driver_below_all_wheels_falls_back_to_cpu(monkeypatch):
    """A GPU too old for any wheel we ship still gets a build that imports."""
    _fake_smi(monkeypatch, SMI_OLD)  # 11.4, below the lowest wheel (11.8)
    assert ml_setup._torch_index_url() == ml_setup._TORCH_CPU_INDEX


def test_no_gpu_selects_cpu_build(monkeypatch):
    _fake_smi(monkeypatch, None)
    assert ml_setup._torch_index_url() == ml_setup._TORCH_CPU_INDEX


def test_plan_reports_what_it_would_install(monkeypatch):
    _fake_smi(monkeypatch, SMI_UMD)
    plan = ml_setup.install_plan()
    assert plan["gpu_detected"] is True
    assert plan["driver_cuda"] == "13.3"
    assert plan["torch_build"] == "cu126"


def test_status_endpoint_shape(client):
    body = client.get("/api/ml/status").json()
    assert "installed" in body
    assert "install" in body and "status" in body["install"]
    assert "plan" in body


def test_install_refused_when_already_present(client, monkeypatch):
    """With the stack present, install is a 409, not a wasted pip run."""
    monkeypatch.setattr(ml_setup, "is_installed", lambda: True)
    r = client.post("/api/ml/install")
    assert r.status_code == 409


def test_install_starts_when_absent(client, monkeypatch):
    """When absent, install is accepted (202) and the worker is dispatched.

    The actual install work is replaced with a no-op — NOT by stubbing
    threading (that would also break the TestClient's own threads and hang the
    request), but by replacing `_run_install`, so a real, harmless thread is
    spawned and the endpoint's contract is what's tested, not pip.
    """
    import threading as _t

    monkeypatch.setattr(ml_setup, "is_installed", lambda: False)
    ran = _t.Event()
    monkeypatch.setattr(ml_setup, "_run_install", lambda: ran.set())
    ml_setup._state.status = "idle"  # clear any prior test's state

    try:
        r = client.post("/api/ml/install")
        assert r.status_code == 202
        assert r.json()["install"]["status"] == "running"
        assert ran.wait(timeout=5), "the install worker was not dispatched"

        # A second start while running is refused, not a second pip process.
        ml_setup._state.status = "running"
        r2 = client.post("/api/ml/install")
        assert r2.status_code == 409
    finally:
        ml_setup._state.status = "idle"  # leave state clean for other tests


@pytest.mark.parametrize("header", [SMI_CLASSIC, SMI_UMD, SMI_OLD])
def test_detection_never_raises(monkeypatch, header):
    """Whatever nvidia-smi prints, selection returns a usable index, never an error."""
    _fake_smi(monkeypatch, header)
    url = ml_setup._torch_index_url()
    assert url.startswith("https://download.pytorch.org/whl/")
