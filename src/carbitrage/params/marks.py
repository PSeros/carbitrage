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
    """A number that also carries the name a study will address it by.

    Putting the mark where the value is *written* removes the need to know
    where it ends up in the case tree::

        keep = Alternative(
            renault,
            Purchase(upfront_extra=Uncertain(1_700, "repair_bill"), already_owned=True),
            life_years=Uncertain(3, "renault_life"),
            label="Renault",
        )

        result.one_way("repair_bill", [800, 1_700, 3_000])

    A third argument declares what is known about the number beyond the base
    value — a distribution when the shape is known, a ``Range`` when only the
    span is::

        Uncertain(1_700, "repair_bill", Triangular(800, 1_700, 4_000))
        Uncertain(0.99, "lpg_price", Range(0.85, 1.35))

    Declaring it once is what stops a tornado and a Monte Carlo over the same
    parameter from resting on two different beliefs.  A study reads it to
    *enrich* its answer and never to narrow its question: a switch point is
    still searched for across the whole domain, and reported with where it
    falls relative to what was declared.

    The mark itself can be kept and passed around, which is the form that
    survives a rename::

        REPAIR = Uncertain(1_700, "repair_bill")
        ...
        result.one_way(REPAIR, [800, 1_700, 3_000])

    It *is* a float: the base case evaluates on the value given, and
    arithmetic, formatting and numpy all see an ordinary number.  A mark
    therefore costs nothing until something asks for it by name.

    The base value is required for that reason.  A case with a hole in it could
    not be run, there would be nothing for a tornado's relative range to scale,
    and a sweep would have no base case to be a perturbation *of*.

    One label may mark several fields, and then it addresses all of them at
    once, the way an alias does.  That is how you say "these two move
    together".

    Being a float has one sharp edge: two marks that share a base value are the
    same dictionary key.  Where several parameters are named in one mapping — a
    tornado's ranges, a Monte Carlo's distributions — key them by label rather
    than by mark if their base values could coincide.

    A mark survives an override, so a case that has been swept can be swept
    again.  The exception is a field declared ``int`` or ``bool``, where
    keeping the value in its declared type wins and the mark is dropped.

    Args:
        value: What the parameter is in the base case.
        label: The name studies address it by.  It shadows an alias of the same
            name, since a mark is specific to this case and an alias is not.
        spread: What is known about the value beyond the base case.  Optional,
            and it must be able to produce the base value: an anchor its own
            uncertainty rules out is a contradiction, not a base case.

    Raises:
        CarbitrageError: on an empty label, or on a base value the spread
            cannot produce.
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
