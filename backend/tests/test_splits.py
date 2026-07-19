"""Train/val/test split assignment — percentage and manual."""

import pytest

from tests.conftest import make_project, upload_images


def _split(client, pid, train, val, test, only_train=False):
    return client.post(
        f"/api/projects/{pid}/dataset/split",
        json={"train_pct": train, "val_pct": val, "test_pct": test, "only_train": only_train},
    )


@pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 33, 100])
def test_split_sums_to_n_and_val_never_empty(client, n):
    """round(), not int() — 80/20 on 3 images must not give an empty val set,
    the exact bug this guards against."""
    pid = make_project(client, f"Split{n}")
    upload_images(client, pid, [f"i{i}.png" for i in range(n)])
    counts = _split(client, pid, 0.8, 0.2, 0.0).json()
    assert counts["train"] + counts["val"] + counts["test"] == n
    if n >= 2:
        assert counts["val"] >= 1


def test_three_way_split_fills_all(client):
    pid = make_project(client, "ThreeWay")
    upload_images(client, pid, [f"i{i}.png" for i in range(10)])
    c = _split(client, pid, 0.7, 0.2, 0.1).json()
    assert c["train"] >= 1 and c["val"] >= 1 and c["test"] >= 1
    assert sum(c.values()) == 10


def test_percentages_must_sum_to_one(client):
    pid = make_project(client, "BadPct")
    upload_images(client, pid, ["a.png", "b.png"])
    assert _split(client, pid, 0.8, 0.8, 0.0).status_code == 400


def test_split_is_reproducible(client):
    """Fixed seed: the same dataset splits the same way every call."""
    pid = make_project(client, "Repro")
    upload_images(client, pid, [f"i{i}.png" for i in range(20)])
    _split(client, pid, 0.8, 0.2, 0.0)
    first = {i["id"]: i["split"] for i in client.get(f"/api/projects/{pid}/images").json()}
    _split(client, pid, 0.8, 0.2, 0.0)
    second = {i["id"]: i["split"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert first == second


def test_only_train_leaves_test_untouched(client):
    """Carve val out of train (imported train+test, no valid) without
    disturbing the existing test set."""
    pid = make_project(client, "OnlyTrain")
    upload_images(client, pid, [f"i{i}.png" for i in range(10)])
    # Put 4 into test manually first.
    ids = [i["id"] for i in client.get(f"/api/projects/{pid}/images").json()][:4]
    client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": ids, "split": "test"},
    )
    c = _split(client, pid, 0.75, 0.25, 0.0, only_train=True).json()
    assert c["test"] == 4, "only_train must not touch the test set"
    assert c["val"] >= 1


def test_split_selected_moves_exactly_those(client):
    pid = make_project(client, "SplitSel")
    upload_images(client, pid, [f"i{i}.png" for i in range(5)])
    ids = [i["id"] for i in client.get(f"/api/projects/{pid}/images").json()][:2]
    r = client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": ids, "split": "test"},
    )
    assert r.status_code == 200
    after = {i["id"]: i["split"] for i in client.get(f"/api/projects/{pid}/images").json()}
    assert all(after[i] == "test" for i in ids)
    assert sum(1 for s in after.values() if s == "test") == 2


def test_split_selected_rejects_bad_split(client):
    pid = make_project(client, "BadSplit")
    upload_images(client, pid, ["a.png"])
    ids = [i["id"] for i in client.get(f"/api/projects/{pid}/images").json()]
    r = client.post(
        f"/api/projects/{pid}/dataset/split-selected",
        json={"image_ids": ids, "split": "nonsense"},
    )
    assert r.status_code == 400
