"""TCDataset refuses the six spent 2025 Atlantic test storms (John's block in src/hurricane_intensity/dataset.py).

Written before the block. The block tests are strict xfail, so they XPASS and fail once the block lands,
and the marker has to be removed. Every storm here has a valid fixture manifest and crop in a temporary
processed_root, so a ValueError can't come from a missing or empty file.

The block covers the six Atlantic IDs only (N1, Option A). The East Pacific 2025 storms are the future H-EP
held-out set and must still load, so a block on "any 2025 ID" fails test_ep_2025_storm_still_loads.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from hurricane_intensity import dataset
from hurricane_intensity.dataset import TCDataset

# Hard-coded, not read from storms_test_2025.txt, so editing the list can't weaken these tests.
SPENT_2025 = ["AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"]
LEGAL = "AL142018"   # Michael, a training storm
EP_2025 = "EP052025"  # Erick, future H-EP

# raises= limits each expected failure to "DID NOT RAISE ValueError", so a broken fixture or a crash is a
# real failure, not a hidden xfail.
AWAITING = pytest.mark.xfail(strict=True, reason="awaiting John's TCDataset 2025 block", raises=pytest.fail.Exception)
# Today an ID with whitespace finds no directory (FileNotFoundError, or OSError for a tab or newline).
AWAITING_PATH_ERROR = pytest.mark.xfail(strict=True, reason="awaiting John's TCDataset 2025 block",
                                        raises=(pytest.fail.Exception, OSError))


def write_storm(root, storm_id, n=2):
    crops = root / storm_id / "crops"
    crops.mkdir(parents=True)
    rows = []
    for hour in range(n):
        name = f"{storm_id}_20250901T{hour:02d}00Z.npz"
        bt = np.full((40, 44), 250.0, dtype=np.float32)
        np.savez(crops / name, bt=bt, wind_kt=np.float32(80.0))
        rows.append({"atcf_id": storm_id, "wind_kt": 80.0, "crop_path": f"{storm_id}/crops/{name}"})
    pd.DataFrame(rows).to_csv(root / storm_id / "manifest.csv", index=False)


@pytest.fixture
def processed_root(tmp_path, monkeypatch):
    # TCDataset reads excluded_crops.txt from the working directory; keep the repo's out of it.
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "processed"
    for storm_id in SPENT_2025 + [LEGAL, EP_2025]:
        write_storm(root, storm_id)
    return root


def assert_refused(root, storm_ids):
    with pytest.raises(ValueError) as exc:
        TCDataset(str(root), storm_ids)
    # pandas' EmptyDataError and ParserError are ValueErrors too; those would mean a broken fixture.
    assert not isinstance(exc.value, (pd.errors.EmptyDataError, pd.errors.ParserError))


@pytest.mark.parametrize("storm_id", SPENT_2025)
def test_fixture_is_valid(processed_root, storm_id):
    """Not xfail: each 2025 fixture reads cleanly, so the block tests below can only pass on the block."""
    df = pd.read_csv(processed_root / storm_id / "manifest.csv")
    assert len(df) == 2
    for p in df.crop_path:
        with np.load(processed_root / p) as z:
            assert z["bt"].shape == (40, 44)


@AWAITING
@pytest.mark.parametrize("storm_id", SPENT_2025)
def test_2025_storm_refused(processed_root, storm_id):
    assert_refused(processed_root, [storm_id])


@AWAITING
@pytest.mark.parametrize("storm_ids", [[LEGAL, "AL092025"], ["AL092025", LEGAL]], ids=["legal-first", "2025-first"])
def test_2025_storm_refused_when_mixed_with_legal(processed_root, storm_ids):
    assert_refused(processed_root, storm_ids)


@AWAITING
def test_lowercase_2025_id_refused(processed_root):
    # NTFS is case-insensitive, so the lowercase ID would load the real storm's crops.
    assert (processed_root / "al042025" / "manifest.csv").exists()
    assert_refused(processed_root, ["al042025"])


@AWAITING_PATH_ERROR
@pytest.mark.parametrize("storm_id", [" AL042025", "AL042025 ", "\tAL042025\n"], ids=["lead", "trail", "tab-nl"])
def test_2025_id_with_whitespace_refused(processed_root, storm_id):
    assert_refused(processed_root, [storm_id])


@AWAITING
def test_refused_before_any_manifest_is_read(processed_root, monkeypatch):
    read = []
    real = dataset.pd.read_csv
    monkeypatch.setattr(dataset.pd, "read_csv", lambda *a, **k: read.append(a) or real(*a, **k))
    # The legal storm comes first, so a check made inside the manifest loop would read its manifest.
    assert_refused(processed_root, [LEGAL, "AL132025"])
    assert read == []


def test_no_override_parameter():
    """Not xfail: the block adds no way around it. The signature stays exactly as it is now."""
    params = inspect.signature(TCDataset.__init__).parameters
    assert [(p.name, p.kind, p.default) for p in params.values()] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("processed_root", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("storm_ids", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("circular_mask", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("rotate", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("generator", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ]


@pytest.mark.parametrize("storm_id", [LEGAL, EP_2025], ids=["legal", "ep-2025"])
def test_non_spent_storm_still_loads(processed_root, storm_id):
    """Positive control, not xfail. EP052025 is future H-EP: a block on any 2025 ID would fail here."""
    ds = TCDataset(str(processed_root), [storm_id])
    assert len(ds) == 2
    x, y = ds[0]
    assert tuple(x.shape) == (3, 224, 224)
    assert y.item() == pytest.approx(80.0 / 185.0)
