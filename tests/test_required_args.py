"""N6: every path argument of train.py and evaluate.py is required and has no default.

Both scripts parse arguments at module level with no __main__ guard, so importing them would run them.
These tests read the source with ast and never import or execute either script.
"""
import ast

import pytest

REQUIRED = {
    "scripts/train.py": ["--train-list", "--val-list", "--out"],
    "scripts/evaluate.py": ["--checkpoint", "--holdout-list", "--train-list", "--out"],
}
# Path arguments whose default None means "off". They stay optional.
OPTIONAL_OFF = {"scripts/train.py": ["--epoch-dir", "--frame-list"]}


def add_argument_calls(path):
    calls = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            assert node.args[0].value not in calls, f"{path}: {node.args[0].value} added twice"
            calls[node.args[0].value] = {k.arg: k.value for k in node.keywords}
    return calls


@pytest.mark.parametrize("script,flag", [(s, f) for s, flags in REQUIRED.items() for f in flags])
def test_path_argument_is_required_without_default(checkout_root, script, flag):
    kwargs = add_argument_calls(checkout_root / script).get(flag)
    assert kwargs is not None, f"{script} has no add_argument({flag!r})"
    assert "default" not in kwargs, f"{script} {flag} still has a default"
    assert "required" in kwargs and ast.literal_eval(kwargs["required"]) is True, \
        f"{script} {flag} is not required=True"


@pytest.mark.parametrize("script,flag", [(s, f) for s, flags in OPTIONAL_OFF.items() for f in flags])
def test_off_by_default_arguments_unchanged(checkout_root, script, flag):
    kwargs = add_argument_calls(checkout_root / script)[flag]
    assert "required" not in kwargs and ast.literal_eval(kwargs["default"]) is None
