"""Marking a parameter where it is defined, so a study can name it later."""

from __future__ import annotations

from typing import TypeAlias

from ..errors import CarbitrageError

__all__ = [
    "ParamName",
    "Uncertain",
    "name_of",
]


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
    """

    __slots__ = ("label",)

    label: str

    def __new__(cls, value: float, label: str) -> Uncertain:
        if not isinstance(label, str) or not label.strip():
            raise CarbitrageError(
                f"an uncertain parameter needs a non-empty label to be addressed by, got {label!r}"
            )
        marked = super().__new__(cls, value)
        marked.label = label
        return marked

    def __repr__(self) -> str:
        return f"Uncertain({float(self)!r}, {self.label!r})"

    def __reduce__(self) -> tuple[type[Uncertain], tuple[float, str]]:
        """Copying and pickling have to carry the label, not just the number."""
        return (type(self), (float(self), self.label))


#: How a parameter is named: a mark, an alias, or a dotted path.
ParamName: TypeAlias = "str | Uncertain"


def name_of(name: ParamName) -> str:
    """The string a parameter name resolves under, unwrapping a mark.

    What a result should display, and the one place that knows a mark stands
    for its label rather than for its number.
    """
    return name.label if isinstance(name, Uncertain) else name
