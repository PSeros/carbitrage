"""Optional output adapters: an xlsx workbook and matplotlib figures.

The library takes its inputs in Python and needs nothing from this package.
Every name here is resolved lazily, so reaching for one of the plots does not
require the ``excel`` extra and vice versa -- the import only fails when you
actually ask for something whose extra is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .excel import write_excel
    from .viz import (
        difference_plot,
        npv_density_plot,
        one_way_plot,
        ranking_plot,
        tornado_plot,
    )

__all__ = [
    "difference_plot",
    "npv_density_plot",
    "one_way_plot",
    "ranking_plot",
    "tornado_plot",
    "write_excel",
]

_PLOTS = frozenset(
    {"difference_plot", "npv_density_plot", "one_way_plot", "ranking_plot", "tornado_plot"}
)


def __getattr__(name: str) -> object:
    if name == "write_excel":
        from . import excel

        return excel.write_excel
    if name in _PLOTS:
        from . import viz

        return getattr(viz, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
