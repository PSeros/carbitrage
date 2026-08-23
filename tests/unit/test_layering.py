"""The scopes are the architecture, so the dependencies between them are checked.

A module may import from its own scope, or from a scope below it, never above.
The layer of each scope is declared here rather than inferred from the directory
tree, because the tree encodes the *concepts* — which scope a name belongs to —
and the two questions deserve separate answers.

The one deliberate exception is the six convenience delegators on
``ComparisonResult`` that reach up into ``sensitivity``; those imports are
function-local, so they do not appear in the module-level scan below, and that
is what keeps ``comparison`` importable without ``sensitivity``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import carbitrage

PACKAGE = Path(carbitrage.__file__).parent

#: errors -> rates, cashflow -> the modelled world -> comparison -> study -> reporting
LAYERS = {
    "errors": 0,
    "rates": 1,
    "cashflow": 1,
    "vehicle": 2,
    "energy": 2,
    "residual": 2,
    "tax": 2,
    "incentives": 2,
    "acquisition": 2,
    "context": 2,
    "comparison": 3,
    "params": 4,
    "sensitivity": 4,
    "scenario": 4,
    "reporting": 5,
}


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("carbitrage", *parts))


def layer_of(dotted: str) -> int | None:
    parts = dotted.split(".")
    return LAYERS.get(parts[1]) if len(parts) > 1 else None


def module_level_imports(tree: ast.Module) -> list[ast.ImportFrom]:
    """Top-level `from . import` statements, skipping `if TYPE_CHECKING:` blocks."""
    return [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level]


def resolve(node: ast.ImportFrom, dotted: str) -> str:
    base = dotted.split(".")[: -node.level] or ["carbitrage"]
    return ".".join((*base, node.module)) if node.module else ".".join(base)


SOURCES = sorted(PACKAGE.rglob("*.py"))


def test_every_scope_has_a_declared_layer() -> None:
    """A scope added without a layer would silently escape the scan."""
    assert set(LAYERS) == {name for name in carbitrage.__all__ if name in LAYERS} | set(LAYERS)
    scopes = {p.name.removesuffix(".py") for p in PACKAGE.iterdir() if p.name != "__init__.py"}
    scopes -= {"py.typed", "__pycache__"}
    assert scopes == set(LAYERS), scopes ^ set(LAYERS)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_imports_from_a_scope_above_its_own(path: Path) -> None:
    dotted = module_name(path)
    own = layer_of(dotted)
    if own is None:  # the root facade re-exports everything, by design
        return
    tree = ast.parse(path.read_text())
    for node in module_level_imports(tree):
        target = resolve(node, dotted)
        above = layer_of(target)
        if above is None:
            continue
        assert above <= own, f"{dotted} (layer {own}) imports {target} (layer {above})"


def test_the_scan_actually_sees_something() -> None:
    """A guard against the walk silently matching nothing."""
    seen = sum(len(module_level_imports(ast.parse(p.read_text()))) for p in SOURCES)
    assert seen > 60, seen


SCOPE_INITS = sorted(PACKAGE.glob("*/__init__.py"))


def assigns_to(node: ast.stmt, predicate: object) -> bool:
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and predicate(target.id)  # type: ignore[operator]
        for target in node.targets
    )


def is_dunder_all(node: ast.stmt) -> bool:
    return assigns_to(node, lambda name: name == "__all__")


def is_private_constant(node: ast.stmt) -> bool:
    """A lazy facade needs one, to say which names it is willing to resolve."""
    return assigns_to(node, lambda name: name.startswith("_") and not name.startswith("__"))


@pytest.mark.parametrize("path", SCOPE_INITS, ids=lambda p: p.parent.name)
def test_a_scope_init_holds_nothing_but_re_exports(path: Path) -> None:
    """The concept's surface, never its implementation."""
    body = ast.parse(path.read_text()).body
    assert isinstance(body[0], ast.Expr), f"{path.parent.name} has no module docstring"
    for node in body[1:]:
        allowed = (ast.ImportFrom, ast.Import, ast.If, ast.FunctionDef)
        ok = isinstance(node, allowed) or is_dunder_all(node) or is_private_constant(node)
        assert ok, f"{path.parent.name}/__init__.py has a {type(node).__name__}, not a re-export"
    assert any(is_dunder_all(node) for node in body), f"{path.parent.name} declares no __all__"


def test_every_scope_package_was_found() -> None:
    """A guard against the glob silently matching nothing."""
    assert len(SCOPE_INITS) == 10, [p.parent.name for p in SCOPE_INITS]
