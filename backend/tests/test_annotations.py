"""Manual annotation CRUD and provenance rules."""

from tests.conftest import make_project, upload_images


def _cls(client, pid, name):
    return next(c for c in client.get(f"/api/projects/{pid}/classes").json() if c["name"] == name)


def _one_image(client):
    pid = make_project(client)
    imgs = upload_images(client, pid, ["a.png"])
    return pid, imgs[0]["id"], _cls(client, pid, "car")


def test_manual_box_is_reviewed_and_not_proposed(client):
    """A hand-drawn box needs no review — a human made it. It carries no
    fabricated confidence."""
    pid, img_id, car = _one_image(client)
    r = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )
    assert r.status_code == 201
    a = r.json()
    assert a["source"] == "manual"
    assert a["reviewed"] is True
    assert a["proposed"] is False
    assert a["confidence"] is None


def test_degenerate_and_negative_boxes_rejected(client):
    pid, img_id, car = _one_image(client)
    for bad in ({"width": 0}, {"height": 0}, {"x": -5}):
        body = {"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20, **bad}
        assert client.post(f"/api/images/{img_id}/annotations", json=body).status_code == 422


def test_box_clamped_to_image_bounds(client):
    pid, img_id, car = _one_image(client)  # image is 64x48
    a = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 50, "y": 40, "width": 500, "height": 500},
    ).json()
    assert a["x"] + a["width"] <= 64.01
    assert a["y"] + a["height"] <= 48.01


def test_cross_project_class_rejected(client):
    """A class from another project must not attach — it would export dangling."""
    pid, img_id, _ = _one_image(client)
    other = make_project(client, "Other")
    other_cls = client.post(f"/api/projects/{other}/classes", json={"name": "boat"}).json()
    r = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": other_cls["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    )
    assert r.status_code == 400


def test_relabel_does_not_reset_geometry(client):
    """exclude_unset: a category-only PATCH must not zero the coordinates."""
    pid, img_id, car = _one_image(client)
    person = _cls(client, pid, "person")
    made = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 7, "y": 8, "width": 20, "height": 21},
    ).json()

    r = client.patch(f"/api/annotations/{made['id']}", json={"category_id": person["id"]})
    a = r.json()
    assert a["category_id"] == person["id"]
    assert (a["x"], a["y"], a["width"], a["height"]) == (7, 8, 20, 21)


def test_delete_and_404s(client):
    pid, img_id, car = _one_image(client)
    made = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    ).json()
    assert client.delete(f"/api/annotations/{made['id']}").status_code == 204
    assert client.delete("/api/annotations/999999").status_code == 404
    assert client.patch("/api/annotations/999999", json={"x": 1}).status_code == 404
