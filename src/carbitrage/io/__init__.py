"""Optional input and output adapters.

The core takes its inputs in Python and needs nothing from this package.
"""

from __future__ import annotations

__all__ = ["write_excel"]


def __getattr__(name: str) -> object:
    if name == "write_excel":
        from .excel import write_excel

        return write_excel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
