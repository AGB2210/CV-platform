"""
Dataset import: COCO recognition and — the one that really matters —
per-folder image_id resolution on a Roboflow train/valid/test export.

Each split's COCO file numbers its image_ids from 1, so train's id 1 and
valid's id 1 are DIFFERENT pictures. Resolving globally silently attaches
train's boxes to valid's images and only shows up later as garbage mAP. The
fixture below encodes each split's boxes in a distinct y-band so a cross-wire
is detectable.
"""

import io
import json
import zipfile

from tests.conftest import png_bytes


def _roboflow_zip() -> bytes:
    """A Roboflow-style export: a wrapper dir with train/valid/test, each a COCO
    file whose image_ids restart at 1, boxes in a split-specific y-band."""
    cats = [
        {"id": 1, "name": "widget", "supercategory": ""},
        {"id": 2, "name": "gadget", "supercategory": ""},
    ]
    # (folder, filenames, y-band identifying the split)
    splits = [
        ("train", ["t1.png", "t2.png", "t3.png"], 10),
        ("valid", ["v1.png", "v2.png"], 200),
        ("test", ["s1.png"], 400),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for folder, names, band in splits:
            images, anns = [], []
            for i, name in enumerate(names, start=1):  # image_id RESTARTS at 1
                zf.writestr(f"My-Dataset/{folder}/{name}", png_bytes(640, 480))
                images.append({"id": i, "file_name": name, "width": 640, "height": 480})
                anns.append(
                    {
                        "id": i,
                        "image_id": i,
                        "category_id": 1 if i % 2 else 2,
                        "bbox": [20, band, 100, 60],
                        "area": 6000,
                        "iscrowd": 0,
                    }
                )
            doc = {"images": images, "annotations": anns, "categories": cats}
            zf.writestr(f"My-Dataset/{folder}/_annotations.coco.json", json.dumps(doc))
        zf.writestr("My-Dataset/README.dataset.txt", "exported from roboflow")
    return buf.getvalue()


def test_roboflow_import_maps_folders_to_splits(client):
    pid = client.post("/api/projects", json={"name": "RF"}).json()["id"]
    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("rf.zip", _roboflow_zip(), "application/zip"))],
    )
    assert r.status_code == 201, r.text
    body = r.json()

    assert body["uploaded_count"] == 6
    assert body["annotations_imported"] == 6
    assert sorted(body["classes_created"]) == ["gadget", "widget"]
    assert body["has_split_folders"] is True
    assert body["splits"] == {"train": 3, "val": 2, "test": 1}  # 'valid' -> val
    assert body["needs_val_split"] is False


def test_per_folder_image_id_scoping(client):
    """THE test: every box must land on an image from its OWN split folder."""
    pid = client.post("/api/projects", json={"name": "RFScope"}).json()["id"]
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("rf.zip", _roboflow_zip(), "application/zip"))],
    )
    band = {"train": 10, "val": 200, "test": 400}
    for img in client.get(f"/api/projects/{pid}/images").json():
        for a in client.get(f"/api/images/{img['id']}/annotations").json():
            assert abs(a["y"] - band[img["split"]]) < 1, (
                f"{img['original_filename']} ({img['split']}) has a box at "
                f"y={a['y']}, expected {band[img['split']]} — folders cross-wired"
            )


def test_imported_boxes_are_ground_truth_not_proposals(client):
    pid = client.post("/api/projects", json={"name": "RFGT"}).json()["id"]
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("rf.zip", _roboflow_zip(), "application/zip"))],
    )
    all_boxes = [
        a
        for i in client.get(f"/api/projects/{pid}/images").json()
        for a in client.get(f"/api/images/{i['id']}/annotations").json()
    ]
    assert all_boxes
    assert all(not a["proposed"] for a in all_boxes), "an import is ground truth, not a proposal"
    assert all(a["source"] == "imported" for a in all_boxes)
    assert all(a["reviewed"] for a in all_boxes)


def test_flat_coco_import(client):
    """A flat folder (no train/valid/test) with one COCO file still imports."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ds/a.png", png_bytes(320, 240))
        doc = {
            "images": [{"id": 1, "file_name": "a.png", "width": 320, "height": 240}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "iscrowd": 0}
            ],
            "categories": [{"id": 1, "name": "thing"}],
        }
        zf.writestr("ds/_annotations.coco.json", json.dumps(doc))

    pid = client.post("/api/projects", json={"name": "Flat"}).json()["id"]
    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("flat.zip", buf.getvalue(), "application/zip"))],
    )
    body = r.json()
    assert body["uploaded_count"] == 1
    assert body["annotations_imported"] == 1
    assert body["has_split_folders"] is False


# --- segmentation datasets --------------------------------------------------
# This project is object detection only. A dataset carrying polygons is still a
# perfectly good detection dataset — every polygon has a bounding box.


def _seg_zip(with_bbox: bool) -> bytes:
    """An instance-segmentation COCO export, with or without explicit bboxes.

    The polygon is a 100x60 rectangle at (20, 30), so the derived box is
    knowable: exporters that omit `bbox` expect exactly this to be recomputed.
    """
    buf = io.BytesIO()
    images, anns = [], []
    with zipfile.ZipFile(buf, "w") as zf:
        for i, name in enumerate(["a.png", "b.png"], start=1):
            zf.writestr(f"Seg/{name}", png_bytes(640, 480))
            images.append({"id": i, "file_name": name, "width": 640, "height": 480})
            ann = {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "segmentation": [[20, 30, 120, 30, 120, 90, 20, 90]],
                "area": 6000,
                "iscrowd": 0,
            }
            if with_bbox:
                ann["bbox"] = [20, 30, 100, 60]
            anns.append(ann)
        doc = {
            "images": images,
            "annotations": anns,
            "categories": [{"id": 1, "name": "widget"}],
        }
        zf.writestr("Seg/_annotations.coco.json", json.dumps(doc))
    return buf.getvalue()


def _import(client, name: str, payload: bytes) -> dict:
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("seg.zip", payload, "application/zip"))],
    )
    assert r.status_code == 201, r.text
    return {"pid": pid, **r.json()}


def test_segmentation_fields_are_ignored_when_bboxes_exist(client):
    """Polygons present alongside bboxes: read the bbox, ignore the polygon."""
    body = _import(client, "SegBox", _seg_zip(with_bbox=True))
    assert body["annotations_imported"] == 2
    assert not any("derived" in n for n in body["notes"]), "nothing needed deriving"

    boxes = client.get(f"/api/images/{body['uploaded'][0]['id']}/annotations").json()
    assert (boxes[0]["x"], boxes[0]["y"]) == (20, 30)
    assert (boxes[0]["width"], boxes[0]["height"]) == (100, 60)


def test_segmentation_only_dataset_derives_boxes(client):
    """THE bug this guards: an export with polygons but no `bbox` imported as
    images with ZERO annotations, silently — it looked like the labels were lost.

    The polygon's bounding box is exactly what a detector wants from it, so it
    is computed rather than dropped, and the conversion is reported."""
    body = _import(client, "SegOnly", _seg_zip(with_bbox=False))
    assert body["annotations_imported"] == 2, "polygons became boxes"
    assert any("derived from segmentation" in n for n in body["notes"])

    boxes = client.get(f"/api/images/{body['uploaded'][0]['id']}/annotations").json()
    assert (boxes[0]["x"], boxes[0]["y"]) == (20, 30)
    assert (boxes[0]["width"], boxes[0]["height"]) == (100, 60), "tight around the polygon"


def test_rle_segmentation_is_skipped_not_crashed(client):
    """RLE masks need pycocotools to decode. Skipping is fine; crashing is not."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Rle/a.png", png_bytes(320, 240))
        doc = {
            "images": [{"id": 1, "file_name": "a.png", "width": 320, "height": 240}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": {"counts": "abc123", "size": [240, 320]},
                    "iscrowd": 1,
                }
            ],
            "categories": [{"id": 1, "name": "thing"}],
        }
        zf.writestr("Rle/_annotations.coco.json", json.dumps(doc))

    body = _import(client, "Rle", buf.getvalue())
    assert body["uploaded_count"] == 1, "the image still imports"
    assert body["annotations_imported"] == 0
