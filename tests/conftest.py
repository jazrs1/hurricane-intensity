"""Shared fixtures: the checkout under test and the main checkout.

Storm lists are tracked, so they are read from the checkout under test (this
tree). data/ and models/ are gitignored and exist only in the main checkout,
which scripts/check.py passes in as HC_MAIN_ROOT; run directly, it is found
from git's common dir.
"""
import os
import subprocess
from pathlib import Path

import pytest

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]


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
