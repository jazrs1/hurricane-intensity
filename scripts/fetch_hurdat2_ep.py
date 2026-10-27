#!/usr/bin/env python
"""Fetch the pinned East Pacific (nepac) HURDAT2 into its own cache (B-v2-2).

    python scripts/fetch_hurdat2_ep.py

Downloads config.HURDAT2_EP_URL once into config.HURDAT2_EP_CACHE_PATH, and
only if its SHA-256 equals config.HURDAT2_EP_SHA256. A cached copy is
re-hashed instead. It never reads, writes or refreshes the Atlantic cache
(data/interim/hurdat2_atlantic.txt). Then build East Pacific storms with
build_dataset.py --hurdat2-path <the printed path>.
"""

from __future__ import annotations

from hurricane_intensity import config, hurdat2


def main() -> None:
    path = hurdat2.fetch_ep_hurdat2()
    print(f"{path}  SHA-256 {config.HURDAT2_EP_SHA256} (verified)")


if __name__ == "__main__":
    main()
