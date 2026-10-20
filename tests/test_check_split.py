"""F-34: scripts/check.py's FAST default and --full.

FULL's pytest command must collect exactly what F1's pre-split command (`pytest tests`, no deselection)
collects on the same checkout, FAST must be FULL minus the tests marked slow, and the protected hashes
must run in both modes. The collection tests run
pytest --collect-only in a subprocess on this checkout, so they are slow themselves and run under --full.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check", ROOT / "scripts" / "check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

# F1's pytest command before the split (check.py at ae80cae). --full adds only --durations to it.
PRE_SPLIT = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
F1_BEFORE = ["hashes", "pytest", "evaluate.py smoke", "hashes after smoke", "git I3/I6", "gitignore"]


def names(full):
    return [n for n, _ in check.f1_sections(Path("main"), Path("COORDINATOR.md"), full)]


def test_full_runs_every_section_f1_ran_before():
    assert names(True) == F1_BEFORE


def test_fast_drops_only_the_smoke_run_and_the_hashes_after_it():
    assert names(False) == ["hashes", "pytest", "git I3/I6", "gitignore"]


def test_protected_hashes_run_first_in_both_modes():
    for full in (True, False):
        assert names(full)[0] == "hashes"


def test_pytest_commands():
    assert check.pytest_command(True) == PRE_SPLIT
    assert check.pytest_command(False) == PRE_SPLIT + ["-m", "not slow"]


DURATIONS = """\
============================= slowest durations ==============================
12.50s call     tests/test_a.py::test_marked
3.00s setup    tests/test_b.py::test_unmarked[AL132020-outside H9]
1.50s teardown tests/test_b.py::test_at_the_threshold
1.49s call     tests/test_b.py::test_just_under
1.00s call     tests/test_b.py::test_jitter_at_the_marking_rule
2.00s call     tests/test_b.py::test_unmarked[AL132020-outside H9]
"""


def test_warn_threshold_is_above_the_1s_marking_rule():
    assert check.WARN_S == 1.5


def test_unmarked_slow_names_only_unmarked_tests_of_1_5s_or_more():
    # 1.50 s warns; 1.49 s and 1.00 s (jitter at the 1 s marking rule) stay silent.
    late = check.unmarked_slow(DURATIONS, {"tests/test_a.py::test_marked"})
    assert late == {"tests/test_b.py::test_unmarked[AL132020-outside H9]": 3.0,
                    "tests/test_b.py::test_at_the_threshold": 1.5}


def test_unmarked_slow_is_empty_when_everything_slow_is_marked():
    marked = {"tests/test_a.py::test_marked", "tests/test_b.py::test_unmarked[AL132020-outside H9]",
              "tests/test_b.py::test_at_the_threshold"}
    assert check.unmarked_slow(DURATIONS, marked) == {}


def collect(cmd, *extra):
    r = subprocess.run([*cmd, "--collect-only", *extra], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return {line.strip() for line in r.stdout.splitlines() if "::" in line}


@pytest.mark.slow
def test_full_command_collects_same_ids_as_pre_split_command_and_markers_registered():
    """On this checkout, --full's pytest command selects exactly what F1's pre-split command selects (no
    deselection), and --strict-markers is clean. It does not compare against another commit: the
    cross-commit proof (482a501: 490 ids -> this branch: 499, nothing dropped) is evidence in PR #24."""
    full = collect(check.pytest_command(True), "--strict-markers")
    assert full == collect(PRE_SPLIT)
    assert len(full) > 400


@pytest.mark.slow
def test_fast_is_full_minus_slow():
    full = collect(check.pytest_command(True))
    fast = collect(check.pytest_command(False))
    slow = collect(PRE_SPLIT, "-m", "slow")
    assert slow and fast and not (fast & slow)
    assert fast == full - slow
