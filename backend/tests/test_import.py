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
from pathlib import Path
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


# --- YOLO -------------------------------------------------------------------
# The other format people actually have. Structurally unlike COCO: one .txt per
# image, normalised centre-based coordinates, classes named only by index.


def _yolo_zip(*, with_yaml: bool = True, seg: bool = False, splits=("train", "valid")) -> bytes:
    """An ultralytics-style export: data.yaml + <split>/images + <split>/labels.

    The box is the same rectangle in every image — 100x60 at (20, 30) on a
    640x480 canvas — expressed the YOLO way, so the absolute box the importer
    must reconstruct is knowable exactly.
    """
    cx, cy = (20 + 100 / 2) / 640, (30 + 60 / 2) / 480
    nw, nh = 100 / 640, 60 / 480
    if seg:
        # Same rectangle as a polygon: (20,30) (120,30) (120,90) (20,90).
        label = "0 {} {} {} {} {} {} {} {}".format(
            20 / 640, 30 / 480, 120 / 640, 30 / 480,
            120 / 640, 90 / 480, 20 / 640, 90 / 480,
        )
    else:
        label = f"0 {cx} {cy} {nw} {nh}"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if with_yaml:
            zf.writestr("ds/data.yaml", "names:\n  - widget\n  - gadget\nnc: 2\n")
        for split in splits:
            for i in (1, 2):
                zf.writestr(f"ds/{split}/images/{split}{i}.png", png_bytes(640, 480))
                zf.writestr(f"ds/{split}/labels/{split}{i}.txt", label)
    return buf.getvalue()


def test_yolo_export_imports_with_splits_and_boxes(client):
    pid = client.post("/api/projects", json={"name": "Yolo"}).json()["id"]
    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("y.zip", _yolo_zip(), "application/zip"))],
    )
    assert r.status_code == 201, r.text
    body = r.json()

    assert body["uploaded_count"] == 4
    assert body["annotations_imported"] == 4
    assert body["splits"] == {"train": 2, "val": 2}, "'valid' maps to val"
    assert body["classes_created"] == ["widget", "gadget"], "data.yaml order preserved"


def test_yolo_coordinates_convert_to_absolute_pixels(client):
    """YOLO is normalised and centre-based; we store absolute and corner-based.

    Getting this wrong doesn't error — it stores a plausible box in the wrong
    place, which survives review and trains a model on bad geometry.
    """
    pid = client.post("/api/projects", json={"name": "YoloCoords"}).json()["id"]
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("y.zip", _yolo_zip(splits=("train",)), "application/zip"))],
    )
    img = client.get(f"/api/projects/{pid}/images").json()[0]
    box = client.get(f"/api/images/{img['id']}/annotations").json()[0]
    assert round(box["x"]) == 20 and round(box["y"]) == 30
    assert round(box["width"]) == 100 and round(box["height"]) == 60


def test_yolo_segmentation_labels_become_boxes(client):
    """A YOLO-seg polygon is read as its bounding box, like the COCO path."""
    pid = client.post("/api/projects", json={"name": "YoloSeg"}).json()["id"]
    client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("y.zip", _yolo_zip(seg=True, splits=("train",)), "application/zip"))],
    )
    img = client.get(f"/api/projects/{pid}/images").json()[0]
    box = client.get(f"/api/images/{img['id']}/annotations").json()[0]
    assert round(box["x"]) == 20 and round(box["y"]) == 30
    assert round(box["width"]) == 100 and round(box["height"]) == 60


def test_yolo_without_class_list_names_classes_by_index(client):
    """No data.yaml means the indices can't be named — but the boxes are still
    real, so they import under a placeholder name that can be renamed."""
    pid = client.post("/api/projects", json={"name": "YoloNoYaml"}).json()["id"]
    r = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("y.zip", _yolo_zip(with_yaml=False, splits=("train",)), "application/zip"))],
    )
    body = r.json()
    assert body["annotations_imported"] == 2, "boxes kept, not dropped"
    assert body["classes_created"] == ["class_0"]
    assert any("index" in n for n in body["notes"])


# --- folder upload ----------------------------------------------------------
# The browser sends only BASENAMES in the multipart filename, so the directory
# structure — which is exactly what distinguishes train/ from val/ — arrives in
# a parallel `paths` field. A folder and the same folder zipped must import
# identically; both run through dataset_import.analyse().


def _folder_files(tree: dict[str, bytes]):
    """Build the (files, paths) multipart a folder upload produces."""
    files = [
        ("files", (Path(rel).name, data, "application/octet-stream"))
        for rel, data in tree.items()
    ]
    files += [("paths", (None, rel)) for rel in tree]
    return files


def test_folder_with_split_subfolders_maps_to_splits(client):
    pid = client.post("/api/projects", json={"name": "Folder"}).json()["id"]
    tree = {
        "ds/train/a.png": png_bytes(64, 48),
        "ds/train/b.png": png_bytes(64, 48),
        "ds/valid/c.png": png_bytes(64, 48),
        "ds/test/d.png": png_bytes(64, 48),
    }
    r = client.post(f"/api/projects/{pid}/images", files=_folder_files(tree))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["uploaded_count"] == 4
    assert body["splits"] == {"train": 2, "val": 1, "test": 1}
    assert body["has_split_folders"] is True


def test_folder_without_split_names_goes_entirely_to_train(client):
    """"It is fine if no naming notation is followed. In that case, all go to
    train" — the split tool on the Dataset page carves out val afterwards."""
    pid = client.post("/api/projects", json={"name": "Flatish"}).json()["id"]
    tree = {
        "photos/one.png": png_bytes(64, 48),
        "photos/nested/two.png": png_bytes(64, 48),
    }
    body = client.post(f"/api/projects/{pid}/images", files=_folder_files(tree)).json()
    assert body["uploaded_count"] == 2
    assert body["splits"] == {"train": 2}
    assert body["has_split_folders"] is False
    assert body["needs_val_split"] is True, "the UI must prompt for a val split"


def test_folder_upload_rejects_path_traversal(client):
    """`paths` is client-supplied, so it's the one place an upload could try to
    write outside the scratch tree."""
    pid = client.post("/api/projects", json={"name": "Evil"}).json()["id"]
    tree_files = [
        ("files", ("ok.png", png_bytes(64, 48), "image/png")),
        ("files", ("bad.png", png_bytes(64, 48), "image/png")),
        ("paths", (None, "ds/ok.png")),
        ("paths", (None, "../../escaped.png")),
    ]
    body = client.post(f"/api/projects/{pid}/images", files=tree_files).json()
    assert body["uploaded_count"] == 1, "only the safe one landed"
    assert any("unsafe path" in s for s in body["skipped"])


def test_selecting_images_and_a_coco_json_together_imports_labels(client):
    """The file picker case: images and _annotations.coco.json chosen together.

    They used to upload as plain images with every label silently discarded,
    because a .json simply isn't an image.
    """
    pid = client.post("/api/projects", json={"name": "PickBoth"}).json()["id"]
    doc = {
        "images": [{"id": 1, "file_name": "a.png", "width": 640, "height": 480}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}
        ],
        "categories": [{"id": 1, "name": "widget"}],
    }
    files = [
        ("files", ("a.png", png_bytes(640, 480), "image/png")),
        ("files", ("_annotations.coco.json", json.dumps(doc).encode(), "application/json")),
    ]
    body = client.post(f"/api/projects/{pid}/images", files=files).json()
    assert body["uploaded_count"] == 1
    assert body["annotations_imported"] == 1, "the json was read, not dropped"
    assert body["classes_created"] == ["widget"]
