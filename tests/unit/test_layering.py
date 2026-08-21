"""The directory layout is the architecture, so it is checked.

A module may import from its own package or one below it, never above.  The one
deliberate exception is the six convenience delegators on ``ComparisonResult``
that reach up into ``study.sensitivity``; those imports are function-local, so
they do not appear in the module-level scan below, and that is what keeps
``engine`` importable without ``study``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import carbitrage

PACKAGE = Path(carbitrage.__file__).parent
LAYERS = {"errors": 0, "core": 1, "domain": 2, "engine": 3, "study": 4, "reporting": 5}


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


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_module_imports_from_a_layer_above_its_own(path: Path) -> None:
    dotted = module_name(path)
    own = layer_of(dotted)
    if own is None:  # the facade re-exports everything, by design
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
    assert seen > 30, seen
