"""Storm-list invariants I3, I4 and I5 (FEATURES.md), through hurricane_intensity.splits."""
import importlib.util
from pathlib import Path

import pytest

from hurricane_intensity.splits import load_splits, load_storm_list

TEST_2025 = {"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"}

# I5 (approved by John 2026-10-18): the training set load_splits(train, val)[0] of every storms_train*.txt
# never includes a val_expanded, H9, 2025 or H-EP storm. The only exception is below. H-EP joined with the first
# v2 train list, storms_train_n5_atlantic.txt (reports/v2_survey/hep/protocol.md gate 3).
FORBIDDEN_LISTS = ("storms_holdout_h9.txt", "storms_val_expanded.txt", "storms_test_2025.txt", "storms_holdout_hep.txt")

# Validation list each training list is used with. A new storms_train*.txt fails until it is added here,
# so a v2 list is never checked against a guessed validation set.
VAL_LIST_FOR = {
    "storms_train.txt": "storms_val.txt",
    "storms_train_baseline.txt": "storms_val.txt",
    "storms_train_control2.txt": "storms_val_expanded.txt",
    "storms_train_expanded.txt": "storms_val_expanded.txt",
    "storms_train_n5_atlantic.txt": "storms_val_expanded.txt",
}

# logs/decisions.md, 2026-10-13 (class-imbalance reassignment): Beryl AL022024 and Lee AL132023, both H9
# storms, were moved into the legacy storms_train.txt on purpose. H9 is valid only for checkpoints trained
# on storms_train_control2.txt / storms_train_expanded.txt. Exactly these two, in exactly that list.
OVERLAP_ALLOWLIST = {"storms_train.txt": {"AL022024", "AL132023"}}

# Cross-basin aliases: one storm, two ATCF IDs (review_b1.md section 2). Applied in both directions before every
# disjointness check, so a storm can't enter training under its other basin's ID.
ALIASES = {"AL022022": "EP042022", "AL132022": "EP182022"}  # Bonnie, Julia
# Priscilla EP162025 (H-EP) overlaps these two under the 650 km box rule (hep_concurrency.csv): never in training.
HEP_CONCURRENT = {"EP152025": "Octave", "EP172025": "Raymond"}
# Never listed anywhere: Bonnie under either ID (its EP ID is the one in use, unlisted), and Julia's EP ID.
NEVER_LISTED = {"AL022022", "EP042022", "EP182022"}

# List-role registry (COORDINATOR.md F-1): every storms_*.txt has exactly one role; an unclassified list fails.
LIST_ROLES = {
    "storms_train.txt": "train",
    "storms_train_baseline.txt": "train",
    "storms_train_control2.txt": "train",
    "storms_train_expanded.txt": "train",
    "storms_train_n5_atlantic.txt": "train",
    "storms_val.txt": "val",
    "storms_val_expanded.txt": "val",
    "storms_holdout.txt": "heldout",
    "storms_holdout_baseline.txt": "heldout",
    "storms_holdout_h6.txt": "heldout",
    "storms_holdout_h9.txt": "heldout",
    "storms_holdout_hep.txt": "heldout_sealed",  # H-EP: built into data/hep_sealed/, scored once in the single pass
    "storms_test_2025.txt": "test_spent",  # blocked in TCDataset (SPENT_2025)
    "storms_gradcam.txt": "analysis",  # Grad-CAM survey of held-out storms; never trained on
    "storms_n5_new.txt": "build",  # build_batch.py --ids-file for the 31 N5 storms; inside a train list
}
ROLES = {"train", "val", "heldout", "heldout_sealed", "test_spent", "analysis", "build"}


def ids(root, name):
    return set(load_storm_list(root / name))


def alias_closed(storms):
    both = {**ALIASES, **{v: k for k, v in ALIASES.items()}}
    return set(storms) | {both[s] for s in storms if s in both}


def repo_lists(pattern):
    return sorted(p.name for p in Path(__file__).resolve().parents[1].glob(pattern))


def train_lists():
    return repo_lists("storms_train*.txt")


def val_lists():
    return repo_lists("storms_val*.txt")


def training_set(root, train_path, val_name, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    return set(load_splits(train_list=train_path, val_list=root / val_name, holdout_list=empty)[0])


def forbidden_overlaps(root, train, allowed=frozenset()):
    """{forbidden list: storms of `train` in it}, aliases included in both directions."""
    out = {}
    for other in FORBIDDEN_LISTS:
        overlap = sorted((alias_closed(train) & alias_closed(ids(root, other))) - alias_closed(allowed))
        if overlap:
            out[other] = overlap
    return out


def alias_pairs_listed(listed):
    return [(a, b) for a, b in ALIASES.items() if a in listed and b in listed]


def test_every_train_list_has_a_val_list(checkout_root):
    unmapped = sorted(set(train_lists()) - set(VAL_LIST_FOR))
    assert not unmapped, f"add these to VAL_LIST_FOR in {__file__}: {unmapped}"


@pytest.mark.parametrize("train_list", train_lists())
def test_training_set_is_disjoint(checkout_root, tmp_path, train_list):
    if train_list not in VAL_LIST_FOR:
        pytest.fail(f"{train_list} has no entry in VAL_LIST_FOR")
    train = training_set(checkout_root, checkout_root / train_list, VAL_LIST_FOR[train_list], tmp_path)
    assert train, f"{train_list} has an empty training set"
    bad = forbidden_overlaps(checkout_root, train, OVERLAP_ALLOWLIST.get(train_list, set()))
    assert not bad, f"training set of {train_list} overlaps (aliases included): {bad}"


# I5 extension (approved by John 2026-10-18): a validation list, i.e. anything train.py's --val-list takes for
# epoch/checkpoint selection, never includes an H9, 2025 or H-EP storm. load_splits takes the validation storms out
# of the training set first, so the training-set test above can't see this: a held-out storm added to a val
# list and to the train lists (Fiona AL072022 in the adversary's case) would pass it.
def held_out_in_val(root, val_path):
    return {other: sorted(alias_closed(load_storm_list(val_path)) & alias_closed(ids(root, other)))
            for other in ("storms_holdout_h9.txt", "storms_test_2025.txt", "storms_holdout_hep.txt")}


@pytest.mark.parametrize("val_list", val_lists())
def test_val_list_never_includes_h9_2025_or_hep(checkout_root, val_list):
    for other, overlap in held_out_in_val(checkout_root, checkout_root / val_list).items():
        assert not overlap, f"{val_list} (checkpoint selection) includes {other} storms: {overlap}"


def test_fiona_al072022_in_val_is_caught(checkout_root, tmp_path):
    """Regression for the adversary's case: Fiona AL072022 (H9) added to a validation list."""
    val = tmp_path / "storms_val_expanded.txt"
    val.write_text((checkout_root / "storms_val_expanded.txt").read_text() + "AL072022\n")
    assert held_out_in_val(checkout_root, val)["storms_holdout_h9.txt"] == ["AL072022"]


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


# ---- list-role registry -------------------------------------------------------------------------------------
def test_every_list_has_a_role(checkout_root):
    unclassified = sorted(set(repo_lists("storms_*.txt")) - set(LIST_ROLES))
    assert not unclassified, f"add these to LIST_ROLES in {__file__}: {unclassified}"
    assert not sorted(set(LIST_ROLES) - set(repo_lists("storms_*.txt"))), "LIST_ROLES names lists that don't exist"
    assert set(LIST_ROLES.values()) <= ROLES


def test_roles_agree_with_the_checks(checkout_root):
    """The registry and the I5 sets can't drift apart: train lists are exactly VAL_LIST_FOR's keys, val lists are
    exactly the storms_val*.txt the val test scans, and every sealed or spent list is forbidden."""
    assert {n for n, r in LIST_ROLES.items() if r == "train"} == set(VAL_LIST_FOR) == set(train_lists())
    assert {n for n, r in LIST_ROLES.items() if r == "val"} == set(val_lists())
    assert {n for n, r in LIST_ROLES.items() if r in ("heldout_sealed", "test_spent")} <= set(FORBIDDEN_LISTS)


def test_build_lists_are_inside_a_train_list(checkout_root):
    for name in (n for n, r in LIST_ROLES.items() if r == "build"):
        built = ids(checkout_root, name)
        assert built and any(built <= ids(checkout_root, t) for t in train_lists()), name


# ---- H-EP and aliases (protocol gate 3, F-10) -----------------------------------------------------------------
def test_hep_is_the_six_storms(checkout_root):
    assert load_storm_list(checkout_root / "storms_holdout_hep.txt") == [
        "EP052025", "EP062025", "EP112025", "EP162025", "EP112023", "EP182023"]


def test_hep_concurrent_storms_in_no_training_list(checkout_root):
    for name in train_lists():
        bad = sorted(alias_closed(ids(checkout_root, name)) & set(HEP_CONCURRENT))
        assert not bad, f"{name} names storms concurrent with Priscilla EP162025: {bad}"


def test_each_alias_pair_is_listed_under_one_id_only(checkout_root):
    for name in repo_lists("storms_*.txt"):
        assert not alias_pairs_listed(ids(checkout_root, name)), f"{name} lists both IDs of one storm"
    everywhere = set().union(*(ids(checkout_root, n) for n in repo_lists("storms_*.txt")))
    assert not everywhere & NEVER_LISTED, "Bonnie is never listed; Julia is AL132022 only"


@pytest.mark.parametrize("planted", ["EP112023", "EP182022", "EP042022", "EP152025", "EP172025"],
                         ids=["hep-jova", "alias-julia", "alias-bonnie", "octave", "raymond"])
def test_planted_storm_in_the_n5_list_is_caught(checkout_root, tmp_path, planted):
    """Each guard above fires on a copy of the N5 list with one storm added (the Fiona-test pattern)."""
    bad = tmp_path / "storms_train_n5_atlantic.txt"
    bad.write_text((checkout_root / "storms_train_n5_atlantic.txt").read_text() + planted + "\n")
    train = training_set(checkout_root, bad, "storms_val_expanded.txt", tmp_path)
    caught = (forbidden_overlaps(checkout_root, train) or alias_pairs_listed(set(load_storm_list(bad)))
              or alias_closed(train) & set(HEP_CONCURRENT) or set(load_storm_list(bad)) & NEVER_LISTED)
    assert caught, f"{planted} added to the N5 list passed every guard"


def test_alias_of_a_forbidden_storm_is_caught(checkout_root, tmp_path):
    """Only the alias closure catches this: Julia held out under her Atlantic ID (in a copy of the forbidden lists)
    and her East Pacific ID in training, or in a validation list."""
    root = tmp_path / "repo"
    root.mkdir()
    for name in FORBIDDEN_LISTS:
        (root / name).write_text((checkout_root / name).read_text())
    with open(root / "storms_holdout_hep.txt", "a") as f:
        f.write("AL132022\n")
    assert forbidden_overlaps(root, {"EP182022"}) == {"storms_holdout_hep.txt": ["AL132022", "EP182022"]}
    val = tmp_path / "storms_val_planted.txt"
    val.write_text("EP182022\n")
    assert held_out_in_val(root, val)["storms_holdout_hep.txt"] == ["AL132022", "EP182022"]


# ---- N5 lists tied to the pinned analysis constants ---------------------------------------------------------
def analyze_n5(root):
    spec = importlib.util.spec_from_file_location("analyze_n5", root / "scripts" / "analyze_n5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # main() is behind the __main__ guard
    return module


def test_n5_lists_match_the_analysis_constants(checkout_root, tmp_path):
    A = analyze_n5(checkout_root)
    atlantic = training_set(checkout_root, checkout_root / "storms_train_n5_atlantic.txt", "storms_val_expanded.txt",
                            tmp_path)
    control = training_set(checkout_root, checkout_root / "storms_train_expanded.txt", "storms_val_expanded.txt",
                           tmp_path)
    assert sorted(atlantic) == A.ARM_IDS and sorted(control) == A.CONTROL_IDS
    assert load_storm_list(checkout_root / "storms_val_expanded.txt") == A.VAL_IDS
    assert sorted(load_storm_list(checkout_root / "storms_n5_new.txt")) == A.NEW_IDS
    assert A.TRAIN_LISTS == {"control": "storms_train_expanded.txt", "atlantic": "storms_train_n5_atlantic.txt"}
    expanded = load_storm_list(checkout_root / "storms_train_expanded.txt")
    assert load_storm_list(checkout_root / "storms_train_n5_atlantic.txt")[:len(expanded)] == expanded
