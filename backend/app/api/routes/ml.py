"""
ML setup: is the heavy ML stack installed, and install it if not.

The two features that need torch (Auto-annotate, Train) ask `GET /ml/status`
first. If it reports `installed: false`, the page offers to install — `POST
/ml/install` starts it — and then polls `/ml/status` for progress, showing it
inline. The whole flow keeps the user out of a terminal.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.services import ml_setup

router = APIRouter(tags=["ml"])


@router.get("/ml/status")
def ml_status() -> dict:
    """Whether the ML dependencies are installed, plus any install in progress.

    Also returns the plan — what an install would download for THIS machine
    (GPU-detected torch build) — so the confirm step can say what it will do
    before the user commits to a multi-gigabyte download.
    """
    return {**ml_setup.status(), "plan": ml_setup.install_plan()}


@router.post("/ml/install", status_code=status.HTTP_202_ACCEPTED)
def ml_install() -> dict:
    """Begin installing the ML dependencies in the background.

    202, not 200: the work is accepted, not done. The client polls
    `/ml/status` and shows progress. Guarded against a double-start and against
    running when already installed — both map to 409 rather than launching a
    second pip process.
    """
    try:
        ml_setup.start_install()
    except ml_setup.MlSetupError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    return ml_setup.status()
