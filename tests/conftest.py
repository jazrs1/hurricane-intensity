"""Shared fixtures: the checkout under test and the main checkout.

The tests import hurricane_intensity from this checkout's src/, not from the venv's editable install
(which points at the main checkout), so a PR that changes splits.py is tested with its own code.
Storm lists are tracked, so they are read from this checkout too. data/ and models/ are gitignored
and exist only in the main checkout, which scripts/check.py passes in as HC_MAIN_ROOT; run directly,
it is found from git's common dir.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHECKOUT_ROOT / "src"))

import hurricane_intensity  # noqa: E402

PACKAGE_FILE = Path(hurricane_intensity.__file__).resolve()
if not PACKAGE_FILE.is_relative_to(CHECKOUT_ROOT / "src"):
    raise ImportError(f"hurricane_intensity was imported from {PACKAGE_FILE}, not {CHECKOUT_ROOT / 'src'}")


def pytest_report_header(config):
    return f"hurricane_intensity: {PACKAGE_FILE}"


def find_main_root():
    env = os.environ.get("HC_MAIN_ROOT")
    if env:
        return Path(env).resolve()
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=CHECKOUT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(common).parent


@pytest.fixture(scope="session")
def checkout_root():
    return CHECKOUT_ROOT


@pytest.fixture(scope="session")
def main_root():
    root = find_main_root()
    if not (root / "data" / "processed").is_dir():
        raise FileNotFoundError(f"{root / 'data' / 'processed'} does not exist; is {root} the main checkout?")
    return root
