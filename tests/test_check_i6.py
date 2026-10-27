"""I6 in scripts/check.py (PR #47, #48): site/ at HEAD is the last REVIEWED_SITE_SHAS entry's tree, and main /
origin/main are MAIN_SHA or a descendant of it that is in HEAD's history and has any listed entry's site/. Each case
runs check_git on a throwaway repo.
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check", ROOT / "scripts" / "check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def test_pins():
    assert (check.MAIN_SHA, check.REVIEWED_SITE_SHAS) == ("fbb39c4", ("0f54536", "41f8d94"))


class Repo:
    def __init__(self, root):
        self.root = root

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, check=True).stdout.strip()

    def commit(self, files, parent=None, msg="c"):
        """A commit with exactly `files` ({path: text}) on `parent` (None: a root commit). Returns its SHA."""
        self.git("read-tree", "--empty")
        for path, text in files.items():
            blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=self.root, input=text,
                                  capture_output=True, text=True, check=True).stdout.strip()
            self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        tree = self.git("write-tree")
        return self.git("commit-tree", tree, *(["-p", parent] if parent else []), "-m", msg)

    def point(self, head, main, origin_main=None):
        self.git("update-ref", "--no-deref", "HEAD", head)
        self.git("update-ref", "refs/heads/main", main)
        self.git("update-ref", "refs/remotes/origin/main", origin_main or main)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    for k in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{k}_NAME", "t")
        monkeypatch.setenv(f"GIT_{k}_EMAIL", "t@example.invalid")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    r = Repo(tmp_path)
    r.base = r.commit({"site/index.html": "old site", "a.txt": "1"})
    r.mid = r.commit({"site/index.html": "old site", "a.txt": "2"}, r.base)  # a later main with the old site
    r.site = r.commit({"site/index.html": "new site", "a.txt": "2"}, r.mid)  # the reviewed site
    r.after = r.commit({"site/index.html": "new site", "a.txt": "3"}, r.site)  # later non-site work
    monkeypatch.setattr(check, "CHECKOUT_ROOT", tmp_path)
    monkeypatch.setattr(check, "MAIN_SHA", r.base)
    monkeypatch.setattr(check, "REVIEWED_SITE_SHAS", (r.site,))
    return r


def test_main_at_main_sha_passes(repo):
    repo.point(head=repo.after, main=repo.base)
    check.check_git(repo.root)


def test_main_merged_with_the_reviewed_site_passes(repo):
    repo.point(head=repo.after, main=repo.site)
    check.check_git(repo.root)


def test_site_edit_after_site_sha_fails(repo):
    edited = repo.commit({"site/index.html": "edited site", "a.txt": "3"}, repo.after)
    repo.point(head=edited, main=repo.base)
    with pytest.raises(AssertionError, match="site/ at HEAD .* differs from"):
        check.check_git(repo.root)


def test_site_added_file_fails(repo):
    edited = repo.commit({"site/index.html": "new site", "site/extra.js": "x", "a.txt": "3"}, repo.after)
    repo.point(head=edited, main=repo.base)
    with pytest.raises(AssertionError, match="site/ at HEAD .* differs from"):
        check.check_git(repo.root)


@pytest.mark.parametrize("which", ["main", "origin/main"])
def test_main_at_unrelated_commit_fails(repo, which):
    unrelated = repo.commit({"site/index.html": "new site", "a.txt": "2"})  # same site, but no shared history
    repo.point(head=repo.after, main=unrelated if which == "main" else repo.base,
               origin_main=unrelated if which == "origin/main" else repo.base)
    with pytest.raises(AssertionError, match=rf"^{which} is .*descends=False"):
        check.check_git(repo.root)


def test_main_unrelated_but_merged_into_head_fails(repo):
    unrelated = repo.commit({"site/index.html": "new site", "a.txt": "u"})  # in HEAD's history, not MAIN_SHA's child
    tree = repo.git("rev-parse", f"{repo.after}^{{tree}}")
    head = repo.git("commit-tree", tree, "-p", repo.after, "-p", unrelated, "-m", "merge")
    repo.point(head=head, main=unrelated)
    with pytest.raises(AssertionError, match=r"main is .*descends=False, in HEAD=True, site matches=True"):
        check.check_git(repo.root)


def test_main_whose_site_differs_fails(repo):
    repo.point(head=repo.after, main=repo.mid)  # descends from MAIN_SHA and is in HEAD's history, old site/
    with pytest.raises(AssertionError, match="main is .*site matches=False"):
        check.check_git(repo.root)


def test_main_not_in_heads_history_fails(repo):
    side = repo.commit({"site/index.html": "new site", "a.txt": "side"}, repo.site)  # reviewed site, other branch
    repo.point(head=repo.after, main=side)
    with pytest.raises(AssertionError, match="main is .*in HEAD=False"):
        check.check_git(repo.root)


def test_main_with_an_older_reviewed_site_passes(repo, monkeypatch):
    monkeypatch.setattr(check, "REVIEWED_SITE_SHAS", (repo.mid, repo.site))  # mid carries the older reviewed site
    repo.point(head=repo.after, main=repo.mid)
    check.check_git(repo.root)


def test_main_with_an_unlisted_site_fails(repo, monkeypatch):
    monkeypatch.setattr(check, "REVIEWED_SITE_SHAS", (repo.mid, repo.site))
    unlisted = repo.commit({"site/index.html": "unlisted site", "a.txt": "u"}, repo.base)
    tree = repo.git("rev-parse", f"{repo.after}^{{tree}}")
    head = repo.git("commit-tree", tree, "-p", repo.after, "-p", unlisted, "-m", "merge")
    repo.point(head=head, main=unlisted)  # descends from MAIN_SHA and is in HEAD's history
    with pytest.raises(AssertionError, match=r"main is .*descends=True, in HEAD=True, site matches=False"):
        check.check_git(repo.root)


def test_head_with_an_older_listed_site_fails(repo, monkeypatch):
    monkeypatch.setattr(check, "REVIEWED_SITE_SHAS", (repo.mid, repo.site))
    reverted = repo.commit({"site/index.html": "old site", "a.txt": "3"}, repo.after)  # back to the older site
    repo.point(head=reverted, main=repo.base)
    with pytest.raises(AssertionError, match="site/ at HEAD .* differs from"):
        check.check_git(repo.root)


def test_results_final_2025_porcelain_still_checked(repo):
    repo.point(head=repo.after, main=repo.base)
    (repo.root / "results" / "final_2025").mkdir(parents=True)
    (repo.root / "results" / "final_2025" / "x.csv").write_text("x")
    with pytest.raises(AssertionError, match="results/final_2025 has changes"):
        check.check_git(repo.root)
