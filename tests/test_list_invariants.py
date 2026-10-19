"""Storm-list invariants I3, I4 and I5 (FEATURES.md).

Every storms_*.txt at the repo root must have a role in LIST_ROLES; an unclassified list fails. Each list is
read twice: independently here (utf-8-sig, '#' comments, strict ATCF format) and through
hurricane_intensity.splits.load_storm_list, and the two must agree, so an encoding problem can't hide an ID.
"""
import re
from pathlib import Path

import pytest

from hurricane_intensity.splits import load_splits, load_storm_list

ROOT = Path(__file__).resolve().parents[1]
ATCF_ID = re.compile(r"^(AL|EP|CP)\d{6}$")
TEST_2025 = {"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"}

# Role of every storm list. "historical" lists are kept for reproducing old runs (H6, the baseline held-out set,
# the Grad-CAM selection); they get the format and 2026 checks only.
LIST_ROLES = {
    "storms_train.txt": "train",
    "storms_train_baseline.txt": "train",
    "storms_train_control2.txt": "train",
    "storms_train_expanded.txt": "train",
    "storms_val.txt": "val",
    "storms_val_expanded.txt": "val",
    "storms_holdout_h9.txt": "holdout",
    "storms_test_2025.txt": "test",
    "storms_holdout.txt": "historical",
    "storms_holdout_baseline.txt": "historical",
    "storms_holdout_h6.txt": "historical",
    "storms_gradcam.txt": "historical",
}
ROLES = {"train", "val", "holdout", "test", "historical"}

# I5 (approved by John 2026-10-18): the training set load_splits(train, val)[0] of every training list never
# includes a val_expanded, H9 or 2025 storm, nor any storm of a holdout- or test-role list.
FORBIDDEN_IN_TRAINING = ("storms_holdout_h9.txt", "storms_val_expanded.txt", "storms_test_2025.txt")

# Validation list each training list is used with. A new training list fails until it is added here, so it is
# never checked against a guessed validation set.
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


def all_lists():
    return sorted(p.name for p in ROOT.glob("storms_*.txt"))


def read_ids(name):
    """Independent reader: utf-8-sig, strip '#' comments and whitespace, keep file order."""
    out = []
    for line in (ROOT / name).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def ids(name):
    return set(read_ids(name))


def with_role(role):
    return [n for n in all_lists() if LIST_ROLES.get(n) == role]


def test_every_list_has_a_role():
    unclassified = [n for n in all_lists() if n not in LIST_ROLES]
    assert not unclassified, f"give these a role in LIST_ROLES in {__file__}: {unclassified}"
    assert set(LIST_ROLES.values()) <= ROLES
    missing = sorted(set(LIST_ROLES) - set(all_lists()))
    assert not missing, f"LIST_ROLES names lists that no longer exist: {missing}"


@pytest.mark.parametrize("name", all_lists())
def test_ids_are_well_formed_and_read_the_same(name):
    raw = read_ids(name)
    bad = [s for s in raw if not ATCF_ID.match(s)]
    assert not bad, f"{name}: not ATCF ids: {bad!r}"
    assert load_storm_list(ROOT / name) == raw, f"{name}: load_storm_list disagrees with a utf-8-sig read"


@pytest.mark.parametrize("name", all_lists())
def test_no_2026_id(name):
    bad = [s for s in read_ids(name) if s.endswith("2026")]
    assert not bad, f"{name} names 2026 storms: {bad}"


def test_every_train_list_has_a_val_list():
    unmapped = [n for n in with_role("train") if n not in VAL_LIST_FOR]
    assert not unmapped, f"add these to VAL_LIST_FOR in {__file__}: {unmapped}"


@pytest.mark.parametrize("train_list", with_role("train"))
def test_training_set_is_disjoint(tmp_path, train_list):
    if train_list not in VAL_LIST_FOR:
        pytest.fail(f"{train_list} has no entry in VAL_LIST_FOR")
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    train = set(load_splits(
        train_list=ROOT / train_list,
        val_list=ROOT / VAL_LIST_FOR[train_list],
        holdout_list=empty,
    )[0])
    assert train, f"{train_list} has an empty training set"
    allowed = OVERLAP_ALLOWLIST.get(train_list, set())
    others = set(FORBIDDEN_IN_TRAINING) | set(with_role("holdout")) | set(with_role("test"))
    for other in sorted(others):
        overlap = sorted((train & ids(other)) - allowed)
        assert not overlap, f"training set of {train_list} overlaps {other}: {overlap}"


@pytest.mark.parametrize("val_list", with_role("val"))
def test_val_list_is_disjoint_from_holdout_and_test(val_list):
    for other in with_role("holdout") + with_role("test"):
        overlap = sorted(ids(val_list) & ids(other))
        assert not overlap, f"{val_list} overlaps {other}: {overlap}"


def test_holdout_and_test_are_disjoint():
    for h in with_role("holdout"):
        for t in with_role("test"):
            overlap = sorted(ids(h) & ids(t))
            assert not overlap, f"{h} overlaps {t}: {overlap}"


def test_allowlist_is_still_accurate():
    """Each allowlisted storm is still in its list and still forbidden, so the allowlist can't go stale."""
    forbidden = set().union(*(ids(f) for f in FORBIDDEN_IN_TRAINING))
    for train_list, allowed in OVERLAP_ALLOWLIST.items():
        assert allowed <= ids(train_list), f"{train_list} no longer lists {sorted(allowed)}"
        assert allowed <= forbidden


def test_2025_test_set_is_the_six_storms():
    assert ids("storms_test_2025.txt") == TEST_2025


def test_no_2025_test_storm_in_any_other_list():
    for name in all_lists():
        if name == "storms_test_2025.txt":
            continue
        overlap = sorted(ids(name) & TEST_2025)
        assert not overlap, f"{name} names 2025 test storms: {overlap}"


def test_no_2026_processed_dir(main_root):
    bad = sorted(p.name for p in (main_root / "data" / "processed").glob("*2026*"))
    assert not bad, f"data/processed has 2026 entries: {bad}"
