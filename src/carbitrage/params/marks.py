"""Marking a parameter where it is defined, so a study can name it later."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypeAlias

from ..errors import CarbitrageError

if TYPE_CHECKING:  # pragma: no cover
    from ..sensitivity import Distribution, Range

__all__ = [
    "ParamName",
    "Spread",
    "Uncertain",
    "name_of",
]

#: What is known about a parameter beyond its base value: a shape, or just a span.
Spread: TypeAlias = "Distribution | Range"


class Uncertain(float):
    """A float that carries the name a study will address it by::

        repair = Uncertain(1_700, "repair_bill", spread=Triangular(800, 1_700, 4_000))
        result.one_way(repair, [800, 1_700, 3_000])   # or by its label

    Args:
        value: What the parameter is in the base case.
        label: The name studies address it by.  It shadows an alias of the same
            name, and one label may mark several fields, which addresses them
            together.
        spread: A ``Distribution`` or ``Range``, which must be able to produce
            the base value.  A study reads it to enrich an answer, never to
            narrow its question.

    Raises:
        CarbitrageError: on an empty label, or on a base value the spread
            cannot produce.

    Note:
        Two marks sharing a base value are the same dictionary key, so key by
        label where several are named in one mapping.  A field declared ``int``
        or ``bool`` keeps its type and drops the mark.
    """

    __slots__ = ("label", "spread")

    label: str
    spread: Spread | None

    def __new__(cls, value: float, label: str, spread: Spread | None = None) -> Uncertain:
        if not isinstance(label, str) or not label.strip():
            raise CarbitrageError(
                f"an uncertain parameter needs a non-empty label to be addressed by, got {label!r}"
            )
        marked = super().__new__(cls, value)
        marked.label = label
        marked.spread = spread
        if spread is not None:
            _reject_anchor_outside(float(marked), label, spread)
        return marked

    def __init__(self, value: float, label: str, spread: Spread | None = None) -> None:
        """Nothing to construct — ``__new__`` built the float.

        Declared so that editors, which read ``__init__`` first, can show what
        the constructor takes instead of offering a bare ``Uncertain()``.
        """

    def __repr__(self) -> str:
        shown = f"Uncertain({float(self)!r}, {self.label!r}"
        return shown + (")" if self.spread is None else f", {self.spread!r})")

    def __reduce__(self) -> tuple[type[Uncertain], tuple[float, str, Spread | None]]:
        """Copying and pickling have to carry the label, not just the number."""
        return (type(self), (float(self), self.label, self.spread))


#: How a parameter is named: a mark, an alias, or a dotted path.
ParamName: TypeAlias = "str | Uncertain"


def _remark(value: float, mark: Uncertain) -> Uncertain:
    """``mark``'s declaration carried onto a new value, without re-checking it.

    An override is a study probing the parameter, and a study is entitled to
    probe outside what was declared plausible — a sweep that could not leave the
    band would be answering a narrower question than it was asked.  The anchor
    check guards what the modeller wrote, not where a study looks.
    """
    probed = float.__new__(Uncertain, value)
    probed.label = mark.label
    probed.spread = mark.spread
    return probed


def _support_of(spread: Spread) -> tuple[float, float]:
    """Where a spread can produce values at all, as ``(low, high)``.

    The import is function-local because a spread is defined one scope over, in
    `carbitrage.sensitivity`, which is itself built on this one.  Anyone holding
    a spread has already imported it, so the cost is a dictionary lookup.
    """
    from ..sensitivity import Distribution, Range

    if isinstance(spread, Range):
        # A relative range is a pair of multipliers, so it bounds no value.
        return (-math.inf, math.inf) if spread.relative else (spread.low, spread.high)
    if isinstance(spread, Distribution):
        return spread.support()
    raise CarbitrageError(
        f"a spread must be a Distribution or a Range, got {type(spread).__name__}"
    )


def _reject_anchor_outside(value: float, label: str, spread: Spread) -> None:
    """A base value its own spread cannot produce is a contradiction."""
    low, high = _support_of(spread)
    if low <= value <= high:
        return
    raise CarbitrageError(
        f"the base value {value:,.6g} of {label!r} lies outside {spread!r}, which produces "
        f"values between {low:,.6g} and {high:,.6g}.  The base case would then rest on a "
        "number your own uncertainty rules out."
    )


def name_of(name: ParamName) -> str:
    """The string a parameter name resolves under, unwrapping a mark.

    What a result should display, and the one place that knows a mark stands
    for its label rather than for its number.
    """
    return name.label if isinstance(name, Uncertain) else name
