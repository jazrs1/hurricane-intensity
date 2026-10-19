"""N6: every path argument of train.py and evaluate.py is required and has no default.

Both scripts parse arguments at module level with no __main__ guard, so importing them would run them.
These tests read the source with ast and never import or execute either script. The parser test executes
only the parser construction (the ArgumentParser assignment and the add_argument calls), extracted with ast.
"""
import argparse
import ast

import pytest

REQUIRED = {
    "scripts/train.py": ["--train-list", "--val-list", "--out"],
    "scripts/evaluate.py": ["--checkpoint", "--holdout-list", "--train-list", "--out"],
}
# Path arguments whose default None means "off". They stay optional.
OPTIONAL_OFF = {"scripts/train.py": ["--epoch-dir", "--frame-list"]}
# Each of these can supply a value the caller didn't give: nargs="?" with const for a bare flag, type for
# a transformed value (e.g. lambda v: v or "best_model.pt").
BANNED = {"default", "const", "nargs", "type"}


def add_argument_calls(path):
    calls = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            assert node.args[0].value not in calls, f"{path}: {node.args[0].value} added twice"
            calls[node.args[0].value] = {k.arg: k.value for k in node.keywords}
    return calls


def build_parser(path):
    """The script's ArgumentParser, built by executing only its module-level construction statements."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keep = [stmt for stmt in tree.body
            if isinstance(stmt, (ast.Assign, ast.Expr)) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr in ("ArgumentParser", "add_argument")]
    names = [t.id for s in keep if isinstance(s, ast.Assign) for t in s.targets]
    assert len(names) == 1, f"{path}: expected one parser assignment, found {names}"
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[names[0]]


def dest(flag):
    return flag[2:].replace("-", "_")


@pytest.mark.parametrize("script,flag", [(s, f) for s, flags in REQUIRED.items() for f in flags])
def test_path_argument_is_required_without_default(checkout_root, script, flag):
    kwargs = add_argument_calls(checkout_root / script).get(flag)
    assert kwargs is not None, f"{script} has no add_argument({flag!r})"
    assert not BANNED & set(kwargs), f"{script} {flag} has {sorted(BANNED & set(kwargs))}"
    assert "required" in kwargs and ast.literal_eval(kwargs["required"]) is True, \
        f"{script} {flag} is not required=True"


@pytest.mark.parametrize("script,flag", [(s, f) for s, flags in OPTIONAL_OFF.items() for f in flags])
def test_off_by_default_arguments_unchanged(checkout_root, script, flag):
    kwargs = add_argument_calls(checkout_root / script)[flag]
    assert "required" not in kwargs and ast.literal_eval(kwargs["default"]) is None


@pytest.mark.parametrize("script", REQUIRED)
def test_parser_refuses_missing_paths(checkout_root, script, capsys):
    flags = REQUIRED[script]
    parser = build_parser(checkout_root / script)
    # Given values come through unchanged, including an empty string.
    args = parser.parse_args([a for f in flags for a in (f, f"x{f}")])
    assert [getattr(args, dest(f)) for f in flags] == [f"x{f}" for f in flags]
    args = parser.parse_args([a for f in flags for a in (f, "")])
    assert [getattr(args, dest(f)) for f in flags] == [""] * len(flags)

    attempts = [[]]
    for flag in flags:
        others = [a for f in flags if f != flag for a in (f, f"x{f}")]
        attempts += [others, others + [flag]]  # the flag left out, and given with no value
    for argv in attempts:
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(argv)
        assert exc.value.code == 2, argv
    capsys.readouterr()
