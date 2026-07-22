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


# --- bulk image delete ------------------------------------------------------


def test_bulk_delete_images(client):
    from tests.conftest import make_project, upload_images

    pid = make_project(client, "BulkDel", classes=("car",))
    imgs = upload_images(client, pid, [f"i{i}.png" for i in range(5)])
    ids = [i["id"] for i in imgs]

    r = client.post(
        f"/api/projects/{pid}/images/bulk-delete", json={"image_ids": ids[:3]}
    )
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 3
    assert len(client.get(f"/api/projects/{pid}/images").json()) == 2


def test_bulk_delete_is_scoped_to_the_project(client):
    """An id from another project must not be deleted just because it was in
    the request body — the same rule the split endpoints follow."""
    from tests.conftest import make_project, upload_images

    a = make_project(client, "Mine", classes=("car",))
    b = make_project(client, "Theirs", classes=("car",))
    mine = upload_images(client, a, ["a.png"])[0]["id"]
    theirs = upload_images(client, b, ["b.png"])[0]["id"]

    r = client.post(
        f"/api/projects/{a}/images/bulk-delete", json={"image_ids": [mine, theirs]}
    ).json()
    assert r["deleted"] == 1
    assert r["not_found"] == [theirs]
    assert len(client.get(f"/api/projects/{b}/images").json()) == 1, "untouched"


def test_bulk_delete_keeps_files_once_a_version_exists(client):
    """The same retention rule as the single delete: bytes stay so a restore can
    bring the rows back. Decided once for the batch, not per image."""
    from tests.conftest import make_project, upload_images
    from app.services import storage

    pid = make_project(client, "Recover", classes=("car",))
    imgs = upload_images(client, pid, ["a.png", "b.png"])
    client.post(f"/api/projects/{pid}/dataset/versions", json={"note": None})

    r = client.post(
        f"/api/projects/{pid}/images/bulk-delete",
        json={"image_ids": [i["id"] for i in imgs]},
    ).json()
    assert r["deleted"] == 2
    assert r["recoverable"] is True
    for i in imgs:
        assert (storage.project_dir(pid) / i["filename"]).exists(), "bytes kept"


def test_bulk_delete_removes_files_before_any_version_exists(client):
    """Nothing could restore them, so keeping the bytes would just be litter."""
    from tests.conftest import make_project, upload_images
    from app.services import storage

    pid = make_project(client, "NoVersions", classes=("car",))
    imgs = upload_images(client, pid, ["a.png"])
    r = client.post(
        f"/api/projects/{pid}/images/bulk-delete",
        json={"image_ids": [imgs[0]["id"]]},
    ).json()
    assert r["recoverable"] is False
    assert not (storage.project_dir(pid) / imgs[0]["filename"]).exists()


# --- pagination -------------------------------------------------------------


def test_list_images_reports_the_total_beyond_the_page(client):
    """THE bug this guards: the grid asked for the default page, got 200 rows,
    and rendered them as the whole dataset. A 638-image project showed 200 with
    nothing indicating the rest existed.

    The body stays a plain list — callers that just want "some images" are
    unaffected — and the total rides in a header.
    """
    from tests.conftest import make_project, upload_images

    pid = make_project(client, "Paged", classes=("car",))
    upload_images(client, pid, [f"i{i}.png" for i in range(25)])

    r = client.get(f"/api/projects/{pid}/images?limit=10&offset=0")
    assert r.status_code == 200
    assert len(r.json()) == 10, "one page"
    assert r.headers["X-Total-Count"] == "25", "but the total is knowable"

    last = client.get(f"/api/projects/{pid}/images?limit=10&offset=20")
    assert len(last.json()) == 5
    assert last.headers["X-Total-Count"] == "25"


def test_pages_do_not_overlap_or_skip(client):
    """Stable id ordering means paging covers the set exactly once."""
    from tests.conftest import make_project, upload_images

    pid = make_project(client, "PageWalk", classes=("car",))
    upload_images(client, pid, [f"i{i}.png" for i in range(23)])

    seen: list[int] = []
    for offset in (0, 10, 20):
        page = client.get(f"/api/projects/{pid}/images?limit=10&offset={offset}").json()
        seen.extend(i["id"] for i in page)

    assert len(seen) == 23
    assert len(set(seen)) == 23, "no image appeared on two pages"


# --- server-side list filters (split / state / category) --------------------
#
# These exist because client-side filtering of a loaded PAGE shipped a real
# contradiction: the stats banner counted the whole dataset while the No-boxes
# filter searched only the 200 images on screen, so "1 image has no boxes"
# coexisted with an empty filter result. Filters must see everything.


def test_list_images_filters_by_state(client):
    from tests.conftest import make_project, upload_images

    pid = make_project(client)
    images = upload_images(client, pid, ["a.png", "b.png", "c.png"])
    car = client.get(f"/api/projects/{pid}/classes").json()[0]

    # a: accepted box. b: proposal only. c: nothing.
    client.post(
        f"/api/images/{images[0]['id']}/annotations",
        json={"category_id": car["id"], "x": 1, "y": 1, "width": 5, "height": 5},
    )
    from tests.conftest import add_proposals

    add_proposals(client, images[1]["id"], car["id"], n=1)

    def ids(params: str) -> set[int]:
        r = client.get(f"/api/projects/{pid}/images?{params}")
        return {i["id"] for i in r.json()}, r.headers.get("X-Total-Count")

    got, total = ids("state=annotated")
    assert got == {images[0]["id"]} and total == "1"
    got, total = ids("state=unannotated")
    # b has only a PROPOSAL, which is not an annotation — so b and c are both
    # unannotated. The proposals mental model, enforced at the filter.
    assert got == {images[1]["id"], images[2]["id"]} and total == "2"
    got, total = ids("state=pending")
    assert got == {images[1]["id"]} and total == "1"


def test_list_images_filters_by_split_and_reports_filtered_total(client):
    from tests.conftest import make_project, upload_images

    pid = make_project(client)
    images = upload_images(client, pid, ["a.png", "b.png", "c.png"])
    client.patch(f"/api/images/{images[0]['id']}/split", params={"split": "test"})

    r = client.get(f"/api/projects/{pid}/images?split=test")
    assert {i["id"] for i in r.json()} == {images[0]["id"]}
    assert r.headers["X-Total-Count"] == "1"  # the FILTERED total, not 3


# --- bulk replace (the review page's Save) ----------------------------------


def test_bulk_replace_updates_creates_and_deletes_atomically(client):
    """One PUT = the desired final state: kept ids update, missing ids delete,
    id-less items create."""
    pid, img_id, car = _one_image(client)
    keep = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    ).json()
    gone = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 30, "y": 5, "width": 10, "height": 10},
    ).json()

    r = client.put(
        f"/api/images/{img_id}/annotations",
        json={
            "annotations": [
                # kept, moved
                {"id": keep["id"], "category_id": car["id"], "x": 7, "y": 8, "width": 10, "height": 10},
                # drawn since last save
                {"category_id": car["id"], "x": 40, "y": 20, "width": 12, "height": 12},
            ]
        },
    )
    assert r.status_code == 200
    boxes = {b["id"]: b for b in r.json()}

    assert gone["id"] not in boxes, "box absent from the payload must be deleted"
    assert boxes[keep["id"]]["x"] == 7 and boxes[keep["id"]]["y"] == 8
    new = next(b for b in boxes.values() if b["id"] not in (keep["id"], gone["id"]))
    assert new["source"] == "manual" and new["reviewed"] is True
    assert new["confidence"] is None
    assert len(boxes) == 2


def test_bulk_replace_promotes_only_boxes_a_human_changed(client):
    """Re-sending an auto box unchanged must NOT launder it into 'manual' —
    only an actual edit is a human override."""
    from tests.conftest import add_proposals

    pid, img_id, car = _one_image(client)
    add_proposals(client, img_id, car["id"], n=2)
    client.post(f"/api/images/{img_id}/proposals/accept")
    a1, a2 = client.get(f"/api/images/{img_id}/annotations").json()
    assert a1["source"] == "auto" and a2["source"] == "auto"

    client.put(
        f"/api/images/{img_id}/annotations",
        json={
            "annotations": [
                # untouched: same geometry, same class
                {"id": a1["id"], "category_id": a1["category_id"], "x": a1["x"],
                 "y": a1["y"], "width": a1["width"], "height": a1["height"]},
                # nudged 2px
                {"id": a2["id"], "category_id": a2["category_id"], "x": a2["x"] + 2,
                 "y": a2["y"], "width": a2["width"], "height": a2["height"]},
            ]
        },
    )
    after = {b["id"]: b for b in client.get(f"/api/images/{img_id}/annotations").json()}
    assert after[a1["id"]]["source"] == "auto", "unchanged box must keep its provenance"
    assert after[a1["id"]]["confidence"] == 0.5
    assert after[a2["id"]]["source"] == "manual", "edited box is the human's now"


def test_bulk_replace_never_touches_proposals(client):
    """Save with an EMPTY list deletes your boxes — but a pending batch
    survives untouched. Accept/reject is its only exit."""
    from tests.conftest import add_proposals

    pid, img_id, car = _one_image(client)
    client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    )
    add_proposals(client, img_id, car["id"], n=2)

    r = client.put(f"/api/images/{img_id}/annotations", json={"annotations": []})
    assert r.status_code == 200
    boxes = client.get(f"/api/images/{img_id}/annotations").json()
    assert [b for b in boxes if not b["proposed"]] == []
    assert len([b for b in boxes if b["proposed"]]) == 2


def test_bulk_replace_stale_draft_conflicts_and_applies_nothing(client):
    """An id the server no longer has = the draft is stale. 409, and the save
    must not half-apply — the box that WAS deletable is still there."""
    pid, img_id, car = _one_image(client)
    survivor = client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 10, "height": 10},
    ).json()

    r = client.put(
        f"/api/images/{img_id}/annotations",
        json={
            "annotations": [
                {"id": 999999, "category_id": car["id"], "x": 1, "y": 1, "width": 5, "height": 5},
            ]
        },
    )
    assert r.status_code == 409
    boxes = client.get(f"/api/images/{img_id}/annotations").json()
    assert [b["id"] for b in boxes] == [survivor["id"]], "409 must change nothing"


def test_bulk_replace_rejects_cross_project_class(client):
    pid, img_id, _ = _one_image(client)
    other = make_project(client, "OtherBulk")
    other_cls = client.post(f"/api/projects/{other}/classes", json={"name": "boat"}).json()
    r = client.put(
        f"/api/images/{img_id}/annotations",
        json={
            "annotations": [
                {"category_id": other_cls["id"], "x": 1, "y": 1, "width": 5, "height": 5},
            ]
        },
    )
    assert r.status_code == 400


# --- export content options ---------------------------------------------------


def _export_names(client, pid, query):
    import io
    import zipfile

    r = client.get(f"/api/projects/{pid}/export?{query}")
    assert r.status_code == 200, r.text
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        return set(zf.namelist())


def test_export_content_options(client):
    """full = labels + images; annotations = labels only; images = files only."""
    pid, img_id, car = _one_image(client)
    client.post(
        f"/api/images/{img_id}/annotations",
        json={"category_id": car["id"], "x": 5, "y": 5, "width": 20, "height": 20},
    )

    full = _export_names(client, pid, "format=coco&content=full")
    assert any(n.endswith(".json") for n in full)
    assert any("/images/" in n for n in full)

    labels_only = _export_names(client, pid, "format=coco&content=annotations")
    assert any(n.endswith(".json") for n in labels_only)
    assert not any("/images/" in n for n in labels_only), "annotations-only must carry no image files"

    images_only = _export_names(client, pid, "content=images")
    assert not any(n.endswith(".json") for n in images_only), "images-only must carry no labels"
    assert any(n.endswith("a.png") for n in images_only)

    assert client.get(f"/api/projects/{pid}/export?content=nonsense").status_code == 400
