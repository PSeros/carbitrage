"""Documentation must not drift from the library.

Documentation that no longer runs is worse than no documentation, so the quick
start is executed and its headline numbers are checked against the figures the
README itself quotes.  The remaining blocks are illustrative fragments that
cannot be executed standalone; those are parsed, and every ``result.<attr>``
they mention is checked against the real class.
"""

from __future__ import annotations

import ast
import re

import pytest

from carbitrage import ComparisonResult
from tests.conftest import DOCS_DIR, README

DOC_FILES = [README, *sorted(DOCS_DIR.glob("*.md"))]


def python_blocks(path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), re.S)


def quick_start_source() -> str:
    blocks = python_blocks(README)
    assert blocks, "the README has no python blocks"
    return blocks[0]


def run_quick_start() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(compile(quick_start_source(), "README quick start", "exec"), namespace)
    return namespace


def test_the_quick_start_runs() -> None:
    assert "result" in run_quick_start()


def test_the_quick_start_reproduces_the_numbers_it_prints() -> None:
    """The README shows a ranking table; those figures must still be produced."""
    result = run_quick_start()["result"]
    text = README.read_text()
    for label in ("Buy the EV now", "Lease the EV", "Repair now, replace in 2 years"):
        quoted = re.search(rf"\| {re.escape(label)}\s*\|\s*([\d,]+\.\d+) \|", text)
        assert quoted, f"the README no longer quotes a PV for {label}"
        expected = float(quoted.group(1).replace(",", ""))
        actual = result[label].pv_of_outflows  # type: ignore[index]
        assert actual == pytest.approx(expected, abs=0.01), label


def test_the_documented_winner_is_still_the_winner() -> None:
    assert run_quick_start()["result"].best().name == "Buy the EV now"  # type: ignore[attr-defined]


def test_the_documented_switch_point_is_still_true() -> None:
    found = run_quick_start()["result"].switch_point(  # type: ignore[attr-defined]
        "annual_km", ("Buy the EV now", "Lease the EV")
    )
    assert found is not None
    quoted = re.search(r"annual_km switches the answer at ([\d,]+\.\d+)", README.read_text())
    assert quoted, "the README no longer quotes the switch point"
    assert found.value == pytest.approx(float(quoted.group(1).replace(",", "")), abs=1.0)


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_every_documented_python_block_parses(doc) -> None:
    for index, source in enumerate(python_blocks(doc)):
        compile(source, f"{doc.name}#{index}", "exec")


def test_every_result_attribute_shown_in_the_api_tour_exists() -> None:
    """The fragments cannot be run, but the names they use must be real."""
    for source in python_blocks(DOCS_DIR / "api-tour.md"):
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "result"
            ):
                assert hasattr(ComparisonResult, node.attr), node.attr


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
def test_every_relative_link_resolves(doc) -> None:
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", doc.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (doc.parent / target).resolve().exists(), f"{doc.name} -> {target}"
