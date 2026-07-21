"""
Serving the built frontend from the API process.

This is what makes a release artifact a single runnable thing: one uvicorn, no
Node, the SPA on the same origin as /api so relative fetches work with no
configuration. Two bugs hid in it, both of which LOOKED fine:

  1. The fallback never fired, because StaticFiles RAISES HTTPException(404)
     rather than returning a 404 response — so checking `response.status_code`
     was dead code and deep links 404'd.
  2. Fixing that made the mount at "/" swallow unmatched /api paths, answering
     a mistyped endpoint with "200 OK" and an HTML page. A client asking for
     JSON got a page and a success code.

Both only showed up by asking the running server, which is why these assert on
status AND content type rather than status alone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """An app with a stand-in built frontend mounted, as a release has."""
    import app.main as main_module

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>SPA</title>", encoding="utf-8")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log(1)", encoding="utf-8")

    from fastapi import FastAPI

    from app.api import api_router

    fresh = FastAPI()
    fresh.include_router(api_router, prefix="/api")
    fresh.mount(
        "/", main_module.SpaStaticFiles(directory=dist, html=True), name="frontend"
    )
    monkeypatch.setattr("app.database.SessionLocal", __import__("app.database", fromlist=["SessionLocal"]).SessionLocal)
    return TestClient(fresh)


def test_root_serves_the_spa(served):
    r = served.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_real_assets_are_served(served):
    r = served.get("/assets/app.js")
    assert r.status_code == 200
    assert "console.log" in r.text


def test_a_client_side_route_falls_back_to_index(served):
    """react-router owns /projects/3/train; that URL only exists once the JS has
    booted. Refreshing the page or pasting a link asks the SERVER for it, and a
    plain static handler 404s a route the app genuinely has."""
    r = served.get("/projects/3/train")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SPA" in r.text


def test_unmatched_api_paths_still_404_as_json(served):
    """THE regression this guards: the SPA mount sits at "/", so it also catches
    /api paths the router didn't match. Answering a mistyped endpoint with 200
    and a page means a client asking for JSON gets HTML and a success code, and
    the mistake surfaces far away as a parse error."""
    r = served.get("/api/definitely-not-an-endpoint")
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]


def test_missing_static_image_still_404s(served):
    """Same rule for uploaded files — a missing image is a missing image, not
    a reason to hand back the application shell."""
    r = served.get("/static/images/99/nope.png")
    assert r.status_code == 404


def test_api_still_works_when_the_spa_is_mounted(served):
    r = served.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
