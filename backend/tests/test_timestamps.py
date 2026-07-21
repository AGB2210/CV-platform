"""
Timestamps: one clock (UTC), honestly labelled on the wire, and a
"last activity" that reflects the work people actually do.

These pin two bugs that shipped together and reinforced each other, so a green
run here means neither has come back:

  1. TWO CLOCKS. `func.now()` (SQLite CURRENT_TIMESTAMP) is UTC; `datetime.now()`
     is local. Both land in naive columns, so a single training_jobs row could
     hold created_at in UTC beside started_at in local — the same instant,
     recorded hours apart.
  2. AN UNLABELLED WIRE FORMAT. A naive datetime serialises with no zone, and
     JavaScript reads that as LOCAL time. A correct UTC value was therefore
     re-interpreted as local by the browser and every relative time was wrong by
     the machine's offset — silently, and only by an amount that looks plausible.

And one behavioural bug: "last activity" ignored annotations, which is the
activity this application is mostly for.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from tests.conftest import make_project, upload_images

# An ISO-8601 instant that explicitly says UTC. The trailing Z is the whole
# point — without it the browser guesses, and guesses local.
ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def test_utcnow_is_utc_not_local():
    """The helper must return UTC, whatever the machine's timezone.

    Compared against an independently-derived UTC rather than against
    `datetime.now()`, so the test is meaningful on a CI box already running UTC
    (where a local-time bug would be invisible).
    """
    from app.timestamps import utcnow

    now = utcnow()
    reference = datetime.now(timezone.utc).replace(tzinfo=None)

    assert now.tzinfo is None, "stored timestamps are naive by convention"
    assert abs((now - reference).total_seconds()) < 5


def test_as_utc_iso_marks_the_zone():
    from app.timestamps import as_utc_iso

    assert as_utc_iso(None) is None

    naive = datetime(2026, 7, 20, 7, 13, 20)
    assert as_utc_iso(naive) == "2026-07-20T07:13:20Z"

    # An aware value is CONVERTED, not relabelled: 12:43+05:30 is 07:13 UTC.
    aware = datetime(2026, 7, 20, 12, 43, 20, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert as_utc_iso(aware) == "2026-07-20T07:13:20Z"


def test_project_timestamps_are_sent_as_utc(client):
    """The API must label its instants, or the browser reads them as local."""
    pid = make_project(client)

    body = client.get(f"/api/projects/{pid}").json()
    for field in ("created_at", "updated_at", "last_activity_at"):
        assert ISO_UTC.match(body[field]), f"{field} not UTC-marked: {body[field]!r}"

    # And the value has to be RIGHT, not merely well-formed — a local time with
    # a Z stuck on it would pass a format check and be wrong by the offset.
    created = datetime.strptime(body["created_at"][:19], "%Y-%m-%dT%H:%M:%S")
    assert abs((datetime.now(timezone.utc).replace(tzinfo=None) - created).total_seconds()) < 60


def test_image_and_version_timestamps_are_sent_as_utc(client):
    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    assert ISO_UTC.match(images[0]["created_at"])

    version = client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})
    assert version.status_code in (200, 201), version.text
    assert ISO_UTC.match(version.json()["created_at"])


def test_last_activity_counts_annotations(client):
    """Labelling is activity. It used to be invisible.

    Only Image/DatasetVersion/TrainingJob `created_at` were considered, so an
    afternoon of annotating left "last activity" reading whenever the images
    happened to be uploaded — and a project you had just worked on sorted to the
    bottom of a list ordered by activity.
    """
    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    annotation_id = _annotate(client, pid, images[0]["id"])
    expected = _backdate_everything_except(client, annotation_id)

    after = client.get(f"/api/projects/{pid}").json()["last_activity_at"]
    assert after.startswith(expected.strftime("%Y-%m-%dT%H:%M:%S"))
    assert ISO_UTC.match(after)


def test_last_activity_counts_an_edit_not_just_a_creation(client):
    """Correcting existing boxes is work, and it only moves `updated_at`.

    A review pass that fixes labels creates nothing, so an implementation
    reading only `created_at` reports no activity for it. Forced apart in time
    for the same second-granularity reason as above.
    """
    from app.models import Annotation

    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    annotation_id = _annotate(client, pid, images[0]["id"])
    _backdate_everything_except(client, annotation_id)

    # Now age the annotation's CREATION too, leaving only its edit recent.
    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        row = db.get(Annotation, annotation_id)
        row.created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
        edited = datetime.now(timezone.utc).replace(tzinfo=None)
        row.updated_at = edited
        db.commit()
    finally:
        db.close()

    after = client.get(f"/api/projects/{pid}").json()["last_activity_at"]
    assert after.startswith(edited.strftime("%Y-%m-%dT%H:%M:%S")), (
        f"{after!r} is not the edit time — `updated_at` is not being counted"
    )


def test_last_activity_counts_adding_a_class(client):
    """A class added after everything else is the most recent thing that happened."""
    from app.models import Category, Image

    pid = make_project(client, classes=("car",))
    upload_images(client, pid, ["a.png"])

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=2)
        for row in db.query(Image).all():
            row.created_at = past
        for row in db.query(Category).all():
            row.created_at = past
        db.commit()
    finally:
        db.close()

    r = client.post(f"/api/projects/{pid}/classes", json={"name": "bicycle"})
    assert r.status_code in (200, 201), r.text

    after = client.get(f"/api/projects/{pid}").json()["last_activity_at"]
    # The new class is "now"; everything else is two days old.
    assert (
        datetime.now(timezone.utc).replace(tzinfo=None)
        - datetime.strptime(after[:19], "%Y-%m-%dT%H:%M:%S")
    ) < timedelta(minutes=5)


def test_last_activity_is_never_before_creation(client):
    """The floor. A project with nothing in it still has to answer honestly."""
    pid = make_project(client, classes=())
    body = client.get(f"/api/projects/{pid}").json()
    assert body["last_activity_at"] >= body["created_at"]


def _in_list(client, pid: int) -> dict:
    return next(p for p in client.get("/api/projects").json() if p["id"] == pid)


def _annotate(client, pid: int, image_id: int) -> int:
    category_id = client.get(f"/api/projects/{pid}/classes").json()[0]["id"]
    r = client.post(
        f"/api/images/{image_id}/annotations",
        json={"category_id": category_id, "x": 1, "y": 2, "width": 10, "height": 10},
    )
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _backdate_everything_except(client, annotation_id: int, days: int = 2) -> datetime:
    """Push every other timestamp into the past, and return the annotation's.

    WHY THIS IS NECESSARY. SQLite's CURRENT_TIMESTAMP has SECOND granularity, so
    in a test that runs in milliseconds an image upload and the box drawn on it
    share a timestamp exactly. An assertion like "activity moved after
    annotating" then passes whether or not annotations are counted at all —
    which is precisely how the original bug survived: the evidence that would
    distinguish the two behaviours was rounded away.

    Separating them in time is what gives the assertion teeth: only an
    implementation that actually reads annotation timestamps can return this
    value.
    """
    from app.models import Annotation, Category, Image, Project

    db = client.SessionLocal()  # type: ignore[attr-defined]
    try:
        past = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        for model in (Image, Category):
            for row in db.query(model).all():
                row.created_at = past
        # The project's OWN stamps are a floor on the answer, so they have to
        # move too — otherwise "now" wins and the test measures nothing.
        for row in db.query(Project).all():
            row.created_at = past
            row.updated_at = past
        annotation = db.get(Annotation, annotation_id)
        # Comfortably after `past`, and not rounded to the same second as it.
        recent = past + timedelta(days=1)
        annotation.created_at = recent
        annotation.updated_at = recent
        db.commit()
        return recent
    finally:
        db.close()


def test_list_last_activity_counts_annotations(client):
    """The Projects page is the screen with the complaint, so test that screen.

    THE BUG THIS PINS: `list_projects` built its own activity subqueries rather
    than sharing the helper the detail view used, and its set was narrower —
    images, versions and training runs, but not annotations. The list is what
    the Projects page renders, so labelling work was invisible exactly where it
    was being looked for.
    """
    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    annotation_id = _annotate(client, pid, images[0]["id"])

    # Everything else is two days old; the box is one day old. An implementation
    # that ignores annotations can only report the two-day-old stamp.
    expected = _backdate_everything_except(client, annotation_id)

    listed = _in_list(client, pid)["last_activity_at"]
    assert listed.startswith(expected.strftime("%Y-%m-%dT%H:%M:%S")), (
        f"list reported {listed!r}, which is not the annotation's timestamp "
        f"{expected!r} — annotations are not being counted"
    )


def test_list_and_detail_report_the_same_last_activity(client):
    """The two endpoints must not answer this question differently.

    Two implementations of one question is the `default_training_version`
    failure again: they drift, and the wrong one is the one on screen.
    """
    pid = make_project(client)
    images = upload_images(client, pid, ["a.png"])
    annotation_id = _annotate(client, pid, images[0]["id"])
    _backdate_everything_except(client, annotation_id)

    detail = client.get(f"/api/projects/{pid}").json()["last_activity_at"]
    listed = _in_list(client, pid)["last_activity_at"]
    assert listed == detail


def test_list_last_activity_is_utc_marked(client):
    pid = make_project(client)
    upload_images(client, pid, ["a.png"])
    listed = _in_list(client, pid)
    for field in ("created_at", "updated_at", "last_activity_at"):
        assert ISO_UTC.match(listed[field]), f"{field}: {listed[field]!r}"
