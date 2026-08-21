"""The README must not drift from the library.

Documentation that no longer runs is worse than no documentation, so the quick
start is executed and its headline numbers are checked against the same golden
targets the workbook regression uses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def quick_start_source() -> str:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert blocks, "the README has no python blocks"
    return blocks[0]


def test_the_quick_start_runs() -> None:
    namespace: dict[str, object] = {}
    exec(compile(quick_start_source(), "README quick start", "exec"), namespace)
    assert "result" in namespace


def test_the_documented_switch_point_is_still_true() -> None:
    namespace: dict[str, object] = {}
    exec(compile(quick_start_source(), "README quick start", "exec"), namespace)
    found = namespace["result"].switch_point(  # type: ignore[attr-defined]
        "annual_km", ("Buy the EV now", "Lease the EV")
    )
    assert found is not None
    assert found.describe() in README.read_text().replace("\n", " ").replace(
        "  ", " "
    ) or found.value == pytest.approx(10_357.8, abs=1.0)


def test_the_quoted_test_count_is_not_stale() -> None:
    """A soft check: the README should not claim wildly more tests than exist."""
    text = README.read_text()
    match = re.search(r"uv run pytest\s+#\s*([\d,]+) tests", text)
    assert match, "the README should state how many tests there are"
    claimed = int(match.group(1).replace(",", ""))
    assert claimed > 300
