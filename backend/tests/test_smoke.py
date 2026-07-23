"""Fixture sanity: the app runs against the throwaway DB."""

from tests.conftest import make_project, upload_images


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


def test_isolated_db_starts_empty(client):
    assert client.get("/api/projects").json() == []


def test_create_project_and_upload(client):
    pid = make_project(client, "Smoke")
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    assert len(imgs) == 2
    assert {i["split"] for i in imgs} == {"train"}


def test_openapi_schema_generates(client):
    """/docs depends on /openapi.json, and ONE unresolvable annotation anywhere
    breaks the WHOLE schema. This 500'd for three releases (a quoted
    `-> "FileResponse"` forward reference on the thumbnail route) because
    nothing — tests, smoke, QA — ever asked for it. The README advertises
    /docs; this keeps that advertisement honest."""
    r = client.get("/openapi.json")
    assert r.status_code == 200, r.text[:300]
    doc = r.json()
    assert doc["info"]["title"]
    assert "/api/projects" in doc["paths"], "the schema actually covers the API"
