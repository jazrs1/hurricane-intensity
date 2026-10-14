"""Storm lists for the train / validation / held-out split.

The lists live in text files at the repo root, one ATCF id per line ('#'
comments allowed). Every script reads them through here, so there is one
copy of the split.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_LIST = REPO_ROOT / "storms_train.txt"
VAL_LIST = REPO_ROOT / "storms_val.txt"
HOLDOUT_LIST = REPO_ROOT / "storms_holdout.txt"


def load_storm_list(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.upper())
    return ids


def load_splits(train_list=TRAIN_LIST, val_list=VAL_LIST, holdout_list=HOLDOUT_LIST):
    """Return (train, val, holdout) storm ids, each in file order.

    The train list names every storm a checkpoint sees, validation storms
    included (it is also the list the dataset is built from). The validation
    storms are taken out of it here, so train and val never share a storm.
    """
    listed = load_storm_list(train_list)
    val = load_storm_list(val_list)
    holdout = load_storm_list(holdout_list)

    for name, ids in ((train_list, listed), (val_list, val), (holdout_list, holdout)):
        dupes = sorted({s for s in ids if ids.count(s) > 1})
        if dupes:
            raise ValueError(f"{name} lists a storm more than once: {dupes}")
    missing = [s for s in val if s not in listed]
    if missing:
        raise ValueError(f"validation storms {missing} are not in {train_list}")
    leaked = sorted(set(listed) & set(holdout))
    if leaked:
        raise ValueError(f"{holdout_list} overlaps {train_list}: {leaked}")

    train = [s for s in listed if s not in val]
    return train, val, holdout
