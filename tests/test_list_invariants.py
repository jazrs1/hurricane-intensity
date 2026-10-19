"""Storm-list invariants I3, I4 and I5 (FEATURES.md), through hurricane_intensity.splits."""
from pathlib import Path

import pytest

from hurricane_intensity.splits import load_splits, load_storm_list

TEST_2025 = {"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"}

# I5 (approved by John 2026-10-18): the training set load_splits(train, val)[0] of every storms_train*.txt
# never includes a val_expanded, H9 or 2025 storm. The only exception is below.
FORBIDDEN_LISTS = ("storms_holdout_h9.txt", "storms_val_expanded.txt", "storms_test_2025.txt")

# Validation list each training list is used with. A new storms_train*.txt fails until it is added here,
# so a v2 list is never checked against a guessed validation set.
VAL_LIST_FOR = {
    "storms_train.txt": "storms_val.txt",
    "storms_train_baseline.txt": "storms_val.txt",
    "storms_train_control2.txt": "storms_val_expanded.txt",
    "storms_train_expanded.txt": "storms_val_expanded.txt",
}

# logs/decisions.md, 2026-10-13 (class-imbalance reassignment): Beryl AL022024 and Lee AL132023, both H9
# storms, were moved into the legacy storms_train.txt on purpose. H9 is valid only for checkpoints trained
# on storms_train_control2.txt / storms_train_expanded.txt. Exactly these two, in exactly that list.
OVERLAP_ALLOWLIST = {"storms_train.txt": {"AL022024", "AL132023"}}


def ids(root, name):
    return set(load_storm_list(root / name))


def train_lists():
    return sorted(p.name for p in Path(__file__).resolve().parents[1].glob("storms_train*.txt"))


def test_every_train_list_has_a_val_list(checkout_root):
    unmapped = sorted(set(train_lists()) - set(VAL_LIST_FOR))
    assert not unmapped, f"add these to VAL_LIST_FOR in {__file__}: {unmapped}"


@pytest.mark.parametrize("train_list", train_lists())
def test_training_set_is_disjoint(checkout_root, tmp_path, train_list):
    if train_list not in VAL_LIST_FOR:
        pytest.fail(f"{train_list} has no entry in VAL_LIST_FOR")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    train = set(load_splits(
        train_list=checkout_root / train_list,
        val_list=checkout_root / VAL_LIST_FOR[train_list],
        holdout_list=empty,
    )[0])
    assert train, f"{train_list} has an empty training set"
    allowed = OVERLAP_ALLOWLIST.get(train_list, set())
    for other in FORBIDDEN_LISTS:
        overlap = sorted((train & ids(checkout_root, other)) - allowed)
        assert not overlap, f"training set of {train_list} overlaps {other}: {overlap}"


def val_lists():
    return sorted(p.name for p in Path(__file__).resolve().parents[1].glob("storms_val*.txt"))


# I5 extension (approved by John 2026-10-18): a validation list, i.e. anything train.py's --val-list takes for
# epoch/checkpoint selection, never includes an H9 or 2025 storm. load_splits takes the validation storms out
# of the training set first, so the training-set test above can't see this: a held-out storm added to a val
# list and to the train lists (Fiona AL092022 in the adversary's case) would pass it.
@pytest.mark.parametrize("val_list", val_lists())
def test_val_list_never_includes_h9_or_2025(checkout_root, val_list):
    for other in ("storms_holdout_h9.txt", "storms_test_2025.txt"):
        overlap = sorted(ids(checkout_root, val_list) & ids(checkout_root, other))
        assert not overlap, f"{val_list} (checkpoint selection) includes {other} storms: {overlap}"


def test_allowlist_is_still_accurate(checkout_root):
    """Each allowlisted storm is still in its list and still forbidden, so the allowlist can't go stale."""
    forbidden = set().union(*(ids(checkout_root, f) for f in FORBIDDEN_LISTS))
    for train_list, allowed in OVERLAP_ALLOWLIST.items():
        assert allowed <= ids(checkout_root, train_list), f"{train_list} no longer lists {sorted(allowed)}"
        assert allowed <= forbidden


def test_2025_test_set_is_the_six_storms(checkout_root):
    assert ids(checkout_root, "storms_test_2025.txt") == TEST_2025


def test_no_2025_test_storm_in_any_other_list(checkout_root):
    for path in sorted(checkout_root.glob("storms_*.txt")):
        if path.name == "storms_test_2025.txt":
            continue
        overlap = sorted(set(load_storm_list(path)) & TEST_2025)
        assert not overlap, f"{path.name} names 2025 test storms: {overlap}"


def test_no_2026_id_in_any_list(checkout_root):
    lists = sorted(checkout_root.glob("storms_*.txt"))
    assert lists
    for path in lists:
        bad = [s for s in load_storm_list(path) if s.endswith("2026")]
        assert not bad, f"{path.name} names 2026 storms: {bad}"


def test_no_2026_processed_dir(main_root):
    bad = sorted(p.name for p in (main_root / "data" / "processed").glob("*2026*"))
    assert not bad, f"data/processed has 2026 entries: {bad}"
