"""TCDataset refuses the six spent 2025 Atlantic test storms (the 2025 block in src/hurricane_intensity/dataset.py).

Every storm here has a valid fixture manifest and crop in a temporary processed_root, so a ValueError can't
come from a missing or empty file. The tests were written before the block (strict xfail) and the markers
were removed when it landed.

The block covers the six Atlantic IDs only (N1, Option A). Other 2025 storms must still load: the East Pacific
ones are the future H-EP held-out set, and AL012025 is an Atlantic 2025 storm that was never a test storm.

What the tests require of the block (hc-adversary items 5, 6 and 6b):
- a storm ID that is not exactly ^[A-Z]{2}[0-9]{6}$ (ASCII, no normalisation) is refused before any read;
- a spent ID is refused before any read;
- a manifest whose atcf_id or crop_path names a spent storm is refused, whatever directory it sits in;
- __getitem__ refuses a row whose crop_path names a spent storm, even if df was replaced after construction;
- the spent set is one module-level frozenset literal of exactly the six IDs, dataset.py holds no mutable
  module-level container, and nothing (environment, module or class attribute, parameter) turns the block off.
"""
import ast
import inspect
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hurricane_intensity import dataset
from hurricane_intensity.dataset import TCDataset

# Hard-coded, not read from storms_test_2025.txt, so editing the list can't weaken these tests.
SPENT_2025 = ["AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"]
LEGAL = "AL142018"    # Michael, a training storm
EP_2025 = "EP052025"  # Erick, future H-EP
AL_2025 = "AL012025"  # Andrea, Atlantic 2025 but never a test storm

def case_insensitive_fs():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a").touch()
        return (Path(d) / "A").exists()


CASE_INSENSITIVE = case_insensitive_fs()
needs_case_insensitive_fs = pytest.mark.skipif(not CASE_INSENSITIVE, reason="needs a case-insensitive filesystem")
needs_windows = pytest.mark.skipif(sys.platform != "win32", reason="trailing-dot aliasing is a Windows path rule")


def write_manifest(root, dir_id, atcf_id, crop_paths):
    (root / dir_id).mkdir(parents=True, exist_ok=True)
    rows = [{"atcf_id": atcf_id, "wind_kt": 80.0, "crop_path": p} for p in crop_paths]
    pd.DataFrame(rows).to_csv(root / dir_id / "manifest.csv", index=False)


def write_storm(root, storm_id, n=2):
    crops = root / storm_id / "crops"
    crops.mkdir(parents=True)
    paths = []
    for hour in range(n):
        name = f"{storm_id}_20250901T{hour:02d}00Z.npz"
        np.savez(crops / name, bt=np.full((40, 44), 250.0, dtype=np.float32), wind_kt=np.float32(80.0))
        paths.append(f"{storm_id}/crops/{name}")
    write_manifest(root, storm_id, storm_id, paths)
    return paths


@pytest.fixture
def processed_root(tmp_path, monkeypatch):
    # TCDataset reads excluded_crops.txt from the working directory; keep the repo's out of it.
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "processed"
    for storm_id in SPENT_2025 + [LEGAL, EP_2025, AL_2025]:
        write_storm(root, storm_id)
    return root


@pytest.fixture
def read_spy(monkeypatch):
    read = []
    real = dataset.pd.read_csv
    monkeypatch.setattr(dataset.pd, "read_csv", lambda *a, **k: read.append(a) or real(*a, **k))
    return read


def assert_refused(root, storm_ids):
    with pytest.raises(ValueError) as exc:
        TCDataset(str(root), storm_ids)
    # pandas' EmptyDataError and ParserError are ValueErrors too; those would mean a broken fixture.
    assert not isinstance(exc.value, (pd.errors.EmptyDataError, pd.errors.ParserError))


def spent_row(root, storm_id="AL042025", crop_path=None):
    first = pd.read_csv(root / storm_id / "manifest.csv").iloc[0].to_dict()
    if crop_path is not None:
        first["crop_path"] = crop_path
    return first


@pytest.mark.parametrize("storm_id", SPENT_2025)
def test_fixture_is_valid(processed_root, storm_id):
    """Each 2025 fixture reads cleanly, so the block tests below can only pass on the block."""
    df = pd.read_csv(processed_root / storm_id / "manifest.csv")
    assert len(df) == 2
    for p in df.crop_path:
        with np.load(processed_root / p) as z:
            assert z["bt"].shape == (40, 44)


# --- refused by ID, before any read ---------------------------------------------------------------------

@pytest.mark.parametrize("storm_id", SPENT_2025)
def test_2025_storm_refused(processed_root, storm_id):
    assert_refused(processed_root, [storm_id])


@pytest.mark.parametrize("storm_ids", [[LEGAL, "AL092025"], ["AL092025", LEGAL]], ids=["legal-first", "2025-first"])
def test_2025_storm_refused_when_mixed_with_legal(processed_root, storm_ids):
    assert_refused(processed_root, storm_ids)


@needs_case_insensitive_fs
def test_lowercase_2025_id_refused(processed_root):
    # On a case-insensitive filesystem (NTFS) the lowercase ID would load the real storm's crops.
    assert (processed_root / "al042025" / "manifest.csv").exists()
    assert_refused(processed_root, ["al042025"])


@pytest.mark.parametrize("storm_id", [" AL042025", "AL042025 ", "\tAL042025\n"], ids=["lead", "trail", "tab-nl"])
def test_2025_id_with_whitespace_refused(processed_root, storm_id):
    assert_refused(processed_root, [storm_id])


@pytest.mark.parametrize("alias", [
    pytest.param("AL042025.", marks=needs_windows, id="trailing-dot"),
    pytest.param("./AL042025", id="dot-slash"),
    pytest.param("../processed/AL042025", id="parent-processed"),
    pytest.param("ABSOLUTE", id="absolute"),
])
def test_path_alias_of_2025_storm_refused(processed_root, alias):
    if alias == "ABSOLUTE":
        alias = str(processed_root / "AL042025")
    # Without the block each alias loads the real storm's crops.
    assert (processed_root / alias / "manifest.csv").exists()
    assert_refused(processed_root, [alias])


def test_refused_before_any_manifest_is_read(processed_root, read_spy):
    # The legal storm comes first, so a check made inside the manifest loop would read its manifest.
    assert_refused(processed_root, [LEGAL, "AL132025"])
    assert read_spy == []


# Non-spent IDs, so only the format rule can refuse them. "\n" checks for re.fullmatch (or \Z), not "$";
# the Arabic-Indic digits check for an ASCII-only digit class.
MALFORMED = ["al142018", "Al142018", " AL142018", "AL142018 ", "AL142018\n", "AL142018.", "./AL142018",
             "AL14201", "AL1420188", "", "A1142018", "AL١٤٢٠١٨", str(Path("x") / LEGAL)]


@pytest.mark.parametrize("storm_id", MALFORMED, ids=[repr(s) for s in MALFORMED])
def test_malformed_id_refused_before_any_read(processed_root, read_spy, storm_id):
    assert_refused(processed_root, [LEGAL, storm_id])
    assert read_spy == []


# --- refused by manifest content ------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["copied-manifest", "crop-path-into-spent", "spent-atcf-id", "crop-path-parent-lower"])
def test_manifest_naming_spent_storm_refused(processed_root, case):
    spent_crops = pd.read_csv(processed_root / "AL042025" / "manifest.csv").crop_path.tolist()
    if case == "copied-manifest":  # the whole manifest copied into a legal-looking directory
        write_manifest(processed_root, "ZZ042025", "AL042025", spent_crops)
    elif case == "crop-path-into-spent":  # harmless atcf_id, rows point at the spent crops
        write_manifest(processed_root, "ZZ042025", "ZZ042025", spent_crops)
    elif case == "spent-atcf-id":  # own crops, but the manifest says it is a spent storm
        own = write_storm(processed_root, "ZZ042025")
        write_manifest(processed_root, "ZZ042025", "al042025", own)
    else:  # crop_path written to dodge a first-component check
        write_manifest(processed_root, "ZZ042025", "ZZ042025",
                       [f"../processed/al042025/{p.split('/', 1)[1]}" for p in spent_crops])
    assert_refused(processed_root, ["ZZ042025"])


# --- __getitem__ guard ----------------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", ["AL042025/", "al042025\\", "./AL042025/", "../processed/AL042025/",
                                    pytest.param("AL042025./", marks=needs_windows)],
                         ids=["plain", "lower-backslash", "dot-slash", "parent", "trailing-dot"])
def test_getitem_refuses_spent_row_after_df_replaced(processed_root, prefix):
    ds = TCDataset(str(processed_root), [LEGAL])
    row = spent_row(processed_root)
    row["crop_path"] = prefix + row["crop_path"].split("/", 1)[1]
    assert (processed_root / row["crop_path"]).exists()
    ds.df = pd.concat([ds.df, pd.DataFrame([row])], ignore_index=True)
    with pytest.raises(ValueError):
        ds[len(ds.df) - 1]


# --- no way around it -----------------------------------------------------------------------------------

def test_no_override_parameter():
    """The signature stays exactly as it was before the block."""
    params = inspect.signature(TCDataset.__init__).parameters
    assert [(p.name, p.kind, p.default) for p in params.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("processed_root", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("storm_ids", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("circular_mask", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("rotate", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("generator", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ]


def test_dataset_reads_no_environment():
    """dataset.py never reads the environment, so no variable can switch the block off."""
    tree = ast.parse(Path(dataset.__file__).read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert not names & {"environ", "environb", "getenv", "getenvb", "putenv"}


def test_spent_set_is_one_module_frozenset_of_the_six():
    sets = {name: v for name, v in vars(dataset).items()
            if isinstance(v, (set, frozenset, list, tuple, dict)) and any(s in v for s in SPENT_2025)}
    assert len(sets) == 1, f"spent IDs in {sorted(sets)}"
    (value,) = sets.values()
    assert type(value) is frozenset and value == frozenset(SPENT_2025)


def test_spent_set_is_a_literal_of_the_six():
    """The frozenset is written out in dataset.py as a literal of the six strings, not computed or read."""
    tree = ast.parse(Path(dataset.__file__).read_text(encoding="utf-8"))
    literals = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(stmt.value, ast.Call)                 and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "frozenset":
            (arg,) = stmt.value.args
            assert isinstance(arg, (ast.Set, ast.List, ast.Tuple)), ast.dump(arg)
            assert all(isinstance(e, ast.Constant) and type(e.value) is str for e in arg.elts), ast.dump(arg)
            literals.append(sorted(e.value for e in arg.elts))
    assert literals == [sorted(SPENT_2025)]


def test_no_mutable_module_container():
    """A module-level list, dict, set or bytearray could be edited at runtime to widen or empty the block."""
    mutable = sorted(n for n, v in vars(dataset).items()
                     if not n.startswith("__") and isinstance(v, (list, dict, set, bytearray)))
    assert mutable == []


def test_no_attribute_turns_the_block_off(processed_root, read_spy, monkeypatch):
    """Flip every simple module and class attribute: AL042025 must still be refused each time, before any read
    (the manifest check alone would hide a flag that disables the ID check). Any exception counts as a refusal
    here, so flipping a pattern string to True (a TypeError in re) is fine."""
    targets = [(dataset, n, v) for n, v in vars(dataset).items()] + [(TCDataset, n, v) for n, v in vars(TCDataset).items()]
    simple = [(obj, n) for obj, n, v in targets
              if not n.startswith("__") and (v is None or type(v) in (bool, int, str))]
    for obj, name in simple:
        for value in (True, False, None, 0, 1, "1", "yes"):
            with monkeypatch.context() as m:
                m.setattr(obj, name, value)
                with pytest.raises(Exception):
                    TCDataset(str(processed_root), ["AL042025"])
                assert read_spy == [], f"{name}={value!r} let the ID check through"
    assert_refused(processed_root, ["AL042025"])  # also with no attribute flipped (e.g. none exist)
    assert read_spy == []


# --- positive controls ----------------------------------------------------------------------------------

@pytest.mark.parametrize("storm_id", [LEGAL, EP_2025, AL_2025], ids=["legal", "ep-2025", "al012025"])
def test_non_spent_storm_still_loads(processed_root, storm_id):
    """EP052025 is future H-EP and AL012025 was never a test storm: a block on any (Atlantic) 2025 ID fails here."""
    ds = TCDataset(str(processed_root), [storm_id])
    assert len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224)
    assert y.item() == pytest.approx(80.0 / 185.0)


# --- F-18 -----------------------------------------------------------------------------------------------

def test_storm_ids_are_the_checked_list(processed_root):
    """storm_ids is the list that was checked, even if the caller passed a one-shot iterator."""
    ds = TCDataset(str(processed_root), iter([LEGAL, EP_2025]))
    assert ds.storm_ids == [LEGAL, EP_2025]
    assert len(ds) == 4


def test_each_crop_resolved_once_but_checked_every_item(processed_root, monkeypatch):
    calls = []
    real = dataset.resolved_crop_storm
    monkeypatch.setattr(dataset, "resolved_crop_storm", lambda root, p: calls.append(p) or real(root, p))
    ds = TCDataset(str(processed_root), [LEGAL])
    for _ in range(3):
        for i in range(len(ds)):
            ds[i]
    assert sorted(calls) == sorted(ds.df.crop_path)  # once per crop, at construction
    # The check itself still runs on every item: a spent row added afterwards is resolved and refused.
    ds.df = pd.concat([ds.df, pd.DataFrame([spent_row(processed_root)])], ignore_index=True)
    for _ in range(3):  # refused every time, not only the first (when its resolution is not yet memoised)
        with pytest.raises(ValueError):
            ds[len(ds.df) - 1]
    assert len(calls) == len(ds.df)


@needs_windows
def test_reassigned_processed_root_is_resolved_again(processed_root, tmp_path):
    """The memo is keyed by processed_root too: pointing the dataset at another root re-resolves each crop there.
    Under the other root, LEGAL's directory is a junction into the spent AL042025, so a stale memo (LEGAL, from
    the first root) would let the crop through."""
    import _winapi
    ds = TCDataset(str(processed_root), [LEGAL])
    ds[0]
    other = tmp_path / "other"
    other.mkdir()
    _winapi.CreateJunction(str(processed_root / "AL042025"), str(other / LEGAL))
    ds.processed_root = str(other)
    with pytest.raises(ValueError):
        ds[0]
    ds.processed_root = str(processed_root)
    assert tuple(ds[0][0].shape) == (3, 224, 224)


@needs_windows
def test_missing_crop_is_not_memoised(processed_root):
    """hc-adversary item 12: a crop that does not exist yet must not leave a memo behind. Otherwise a crops
    directory swapped for a junction into a spent storm afterwards would load the spent bytes."""
    import _winapi
    ds = TCDataset(str(processed_root), [LEGAL])
    row = ds.df.iloc[0].to_dict()
    row["crop_path"] = f"{LEGAL}\\crops\\AL042025_20250901T0000Z.npz"
    ds.df = pd.concat([ds.df, pd.DataFrame([row])], ignore_index=True)
    with pytest.raises(FileNotFoundError):
        ds[len(ds.df) - 1]
    shutil.rmtree(processed_root / LEGAL / "crops")
    _winapi.CreateJunction(str(processed_root / "AL042025" / "crops"), str(processed_root / LEGAL / "crops"))
    assert (processed_root / row["crop_path"]).is_file()
    with pytest.raises(ValueError):
        ds[len(ds.df) - 1]


def test_docstring_names_the_path_based_limit():
    assert "copied or hard-linked" in " ".join(TCDataset.__init__.__doc__.split())
