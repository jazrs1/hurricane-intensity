import argparse
import os
import re
import subprocess
import sys
import tempfile
import torch
import pandas as pd
from pathlib import Path
from hurricane_intensity.dataset import TCDataset

EXCLUDED = Path("excluded_crops.txt")
# Spent 2025 Atlantic test storms: never loaded; their existing entries are carried forward unchanged.
# Other 2025 storms (East Pacific, the future H-EP set) are scanned like any other storm.
SPENT_2025 = frozenset({"AL042025", "AL052025", "AL072025", "AL082025", "AL092025", "AL132025"})
STORM_ID = re.compile(r"[A-Z]{2}[0-9]{6}")

# Every refusal exits 1 before excluded_crops.txt is written. Entries of storms that are not scanned (the
# spent six, and storms with no directory under this root, e.g. those under the other root) are carried
# forward unchanged. An entry of a scanned storm is never dropped: if its crop or manifest row is gone,
# every run is refused until the entry is removed by a reviewed, committed manual edit of excluded_crops.txt.


def refuse(reason, lines=()):
    print(f"refusing: {reason}; {EXCLUDED} is unchanged", file=sys.stderr)
    for line in lines:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)


def storm_of(crop_path):
    return crop_path.replace("\\", "/").split("/")[0].strip().upper()


def check_dirs(processed, storms):
    """Each storm directory is named exactly like an ATCF ID and is a real directory, not an alias."""
    malformed = [s for s in storms if not STORM_ID.fullmatch(s)]
    if malformed:
        refuse("storm directories not named like an ATCF ID", malformed)
    root = processed.resolve()
    linked = [s for s in storms
              if (processed / s).is_symlink() or (processed / s).is_junction() or (processed / s).resolve() != root / s]
    if linked:
        refuse("storm directories that are links or junctions", linked)


def check_manifests(processed, storms):
    """Every manifest row belongs to its own directory (crop_path and atcf_id), read before any crop loads."""
    wrong = []
    for s in storms:
        m = pd.read_csv(processed / s / "manifest.csv")
        m = m[m["crop_path"].notna()]
        for _, row in m.iterrows():
            atcf = str(row["atcf_id"]).strip().upper() if "atcf_id" in m else s
            if storm_of(str(row["crop_path"])) != s or atcf != s:
                wrong.append(f"{s}/manifest.csv: crop_path {row['crop_path']}, atcf_id {row.get('atcf_id')}")
    if wrong:
        refuse("manifest rows that belong to another storm", wrong)


def scan(processed, storms):
    """(crop_path of every frame with a NaN pixel or label, frames scanned), ignoring current exclusions."""
    root = str(processed.resolve())
    # TCDataset drops the frames listed in ./excluded_crops.txt; build it from an empty directory instead of
    # deleting the file, so every frame is scanned and the file survives any exception.
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as empty:
        os.chdir(empty)
        try:
            ds = TCDataset(root, storms)
        finally:
            os.chdir(cwd)
    stray = [p for p in ds.df["crop_path"] if storm_of(p) not in storms or storm_of(p) in SPENT_2025]
    if stray:
        refuse("dataset rows outside the scanned storms", stray)
    bad = []
    for i in range(len(ds)):
        img, label = ds[i]
        if torch.isnan(img).any() or torch.isnan(label).any():
            bad.append(ds.df.iloc[i]["crop_path"])
    return bad, len(ds)


def read_old():
    if not EXCLUDED.exists():
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", EXCLUDED.name],
                                 cwd=EXCLUDED.resolve().parent, capture_output=True).returncode == 0
        if tracked:
            refuse(f"{EXCLUDED} is missing but tracked by git; restore it first (git checkout -- {EXCLUDED})")
        return []
    old = [line for line in EXCLUDED.read_text().splitlines() if line.strip()]
    seen, dups = set(), []
    for line in old:
        if line.rstrip() in seen:
            dups.append(line)
        seen.add(line.rstrip())
    if dups:
        print(f"notice: {len(dups)} duplicate entries in {EXCLUDED}; scanned storms' entries are written once:")
        for line in dups:
            print(f"  {line}")
    return old


def main(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate excluded_crops.txt (crops with a NaN pixel or label).")
    parser.add_argument("--processed-root", default="data/processed",
                        help="storm directories to scan; entries are relative to it, as TCDataset expects "
                             "(default data/processed). The output is always ./excluded_crops.txt")
    processed = Path(parser.parse_args(argv).processed_root)

    # Every storm with crops on disk, whichever list it is in, so regenerating
    # the exclusion file can't drop a storm that was added after this was written.
    all_storms = sorted(p.parent.name for p in processed.glob("*/manifest.csv"))
    old = read_old()
    check_dirs(processed, all_storms)
    scanned = [s for s in all_storms if s not in SPENT_2025]
    check_manifests(processed, scanned)

    found, n = scan(processed, scanned)
    kept = [line for line in old if storm_of(line) not in scanned]
    absent = [line for line in kept if storm_of(line) not in SPENT_2025]
    # Same order as before: storm by storm (sorted), frames in manifest order.
    storms = sorted({storm_of(p) for p in found + kept})
    bad = [p for s in storms for p in kept + found if storm_of(p) == s]

    new = {p.rstrip() for p in bad}
    removed = [line for line in old if line.rstrip() not in new]
    if removed:
        refuse(f"the new list would drop {len(removed)} existing entries", removed)

    fd, tmp = tempfile.mkstemp(prefix=EXCLUDED.name + ".", suffix=".tmp", dir=EXCLUDED.resolve().parent)
    os.close(fd)
    try:
        Path(tmp).write_text("\n".join(bad))
        os.replace(tmp, EXCLUDED)
    except BaseException as error:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError as cleanup:
            print(f"also could not remove {tmp}: {cleanup!r}", file=sys.stderr)
            error.add_note(f"also could not remove {tmp}: {cleanup!r}")
        raise
    print(f"{len(bad)} of {n} excluded")
    added = [p for p in bad if p.rstrip() not in {line.rstrip() for line in old}]
    print(f"{len(added)} added" + "".join(f"\n  {p}" for p in added))
    if absent:
        print(f"{len(absent)} carried forward for storms not under {processed}" + "".join(f"\n  {p}" for p in absent))


if __name__ == "__main__":
    main()
