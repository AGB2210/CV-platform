"""
Proposal accept/reject — the heart of the annotation model.

The one rule that everything depends on: a PROPOSAL is not an annotation. It's
excluded from counts and exports until accepted; accepting replaces your boxes
on the covered images; rejecting leaves your boxes untouched.
"""

from tests.conftest import add_proposals, make_project, upload_images


def _cls(client, pid, name):
    return next(c for c in client.get(f"/api/projects/{pid}/classes").json() if c["name"] == name)


def test_proposals_excluded_from_counts(client):
    pid = make_project(client, "Counts")
    imgs = upload_images(client, pid, ["a.png"])
    car = _cls(client, pid, "car")
    add_proposals(client, imgs[0]["id"], car["id"], n=3)

    stats = client.get(f"/api/projects/{pid}/dataset/stats").json()
    assert stats["total_boxes"] == 0, "proposals must not count as annotations"
    assert stats["proposed_boxes"] == 3
    assert stats["annotated_images"] == 0

    # And the per-image list agrees.
    row = client.get(f"/api/projects/{pid}/images").json()[0]
    assert row["annotation_count"] == 0
    assert row["proposed_count"] == 3


def test_accept_all_replaces_existing_on_covered_images(client):
    pid = make_project(client, "Accept")
    imgs = upload_images(client, pid, ["a.png"])
    car = _cls(client, pid, "car")
    img_id = imgs[0]["id"]

    # A hand-drawn box on the image the model will also propose on.
    client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 200, "y": 5, "width": 20, "height": 20},
    )
    add_proposals(client, img_id, car["id"], n=2)

    r = client.post(f"/api/projects/{pid}/proposals/accept")
    assert r.status_code == 200
    assert r.json()["deleted_existing"] == 1  # the hand-drawn box replaced

    boxes = client.get(f"/api/images/{img_id}/annotations").json()
    assert all(not b["proposed"] for b in boxes)
    assert all(b["source"] == "auto" for b in boxes), "manual box was replaced"
    assert len(boxes) == 2


def test_reject_all_keeps_your_boxes(client):
    pid = make_project(client, "Reject")
    imgs = upload_images(client, pid, ["a.png"])
    car = _cls(client, pid, "car")
    img_id = imgs[0]["id"]

    made = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    ).json()
    add_proposals(client, img_id, car["id"], n=2)

    assert client.delete(f"/api/projects/{pid}/proposals").status_code == 204

    boxes = client.get(f"/api/images/{img_id}/annotations").json()
    assert len(boxes) == 1
    assert boxes[0]["id"] == made["id"], "your box survived untouched"


def test_accept_only_touches_covered_images(client):
    pid = make_project(client, "Scoped")
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    car = _cls(client, pid, "car")
    a, b = imgs[0]["id"], imgs[1]["id"]

    # A box on b, proposals only on a.
    client.post(
        f"/api/images/{b}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )
    add_proposals(client, a, car["id"], n=1)

    client.post(f"/api/projects/{pid}/proposals/accept")

    # b was never in the batch — its box is untouched.
    b_boxes = client.get(f"/api/images/{b}/annotations").json()
    assert len(b_boxes) == 1 and b_boxes[0]["source"] == "manual"


def test_per_image_accept_and_reject(client):
    pid = make_project(client, "PerImage")
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    car = _cls(client, pid, "car")
    a, b = imgs[0]["id"], imgs[1]["id"]
    add_proposals(client, a, car["id"], n=1)
    add_proposals(client, b, car["id"], n=1)

    # Accept only a.
    assert client.post(f"/api/images/{a}/proposals/accept").status_code == 200
    assert client.get(f"/api/projects/{pid}/proposals/count").json()["proposed_boxes"] == 1

    # Reject only b.
    assert client.delete(f"/api/images/{b}/proposals").status_code == 204
    assert client.get(f"/api/projects/{pid}/proposals/count").json()["proposed_boxes"] == 0

    assert len(client.get(f"/api/images/{a}/annotations").json()) == 1
    assert client.get(f"/api/images/{b}/annotations").json() == []


def test_accept_empty_batch_is_400(client):
    pid = make_project(client, "Empty")
    upload_images(client, pid, ["a.png"])
    assert client.post(f"/api/projects/{pid}/proposals/accept").status_code == 400


def test_preview_reports_what_accept_deletes(client):
    pid = make_project(client, "Preview")
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    car = _cls(client, pid, "car")
    a, b = imgs[0]["id"], imgs[1]["id"]

    client.post(
        f"/api/images/{a}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )
    client.post(
        f"/api/images/{b}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )
    add_proposals(client, a, car["id"], n=1)  # proposals only on a

    pv = client.get(f"/api/projects/{pid}/proposals/preview").json()
    assert pv["proposed_images"] == 1
    assert pv["existing_on_proposed_images"] == 1  # a's box, deleted on accept
    assert pv["existing_elsewhere"] == 1  # b's box, untouched


def test_proposals_never_export(client):
    """A proposal must not appear in a COCO export."""
    import io
    import json
    import zipfile

    pid = make_project(client, "Export", classes=("car",))
    imgs = upload_images(client, pid, ["a.png"])
    car = _cls(client, pid, "car")
    img_id = imgs[0]["id"]

    client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )
    add_proposals(client, img_id, car["id"], n=3)  # these must NOT export

    r = client.get(f"/api/projects/{pid}/export?format=coco")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        doc = json.loads(zf.read(next(n for n in zf.namelist() if n.endswith(".json"))))
    assert len(doc["annotations"]) == 1, "only the accepted box exports, not the 3 proposals"
