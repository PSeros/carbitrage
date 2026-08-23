"""Reading and overriding a parameter by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import CarbitrageError
from .aliases import ALIASES, FieldOf
from .marks import ParamName, name_of
from .paths import _get_path, _marks, _paths_matching, _set_path, _walk

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from ..comparison import Case

__all__ = [
    "describe_parameters",
    "find",
    "get_param",
    "resolve",
    "scale_param",
    "set_param",
    "set_params",
    "uncertainties",
]


def resolve(case: Case, name: ParamName) -> tuple[str, ...]:
    """Every concrete dotted path ``name`` addresses, in tree order.

    A name is one of three things, tried in that order: the label of an
    :class:`~carbitrage.params.Uncertain` mark somewhere in the case, an entry
    in :data:`~carbitrage.params.ALIASES`, or a dotted path.  A mark shadows an
    alias of the same name, because a mark was written for *this* case and an
    alias was not.

    Raises:
        CarbitrageError: if the name matches nothing.
    """
    label = name_of(name)
    marked = tuple(path for found, path in _marks(case) if found == label)
    if marked:
        return marked
    spec = ALIASES.get(label, label)
    if isinstance(spec, FieldOf):
        paths = tuple(_paths_matching(case, spec))
        if not paths:
            raise CarbitrageError(
                f"alias {label!r} matches no {spec.owner.__name__} in this case.  "
                f"Available parameters: {', '.join(describe_parameters(case)[:12])} ..."
            )
        return paths
    try:
        _get_path(case, spec)  # raises if the path is wrong
    except CarbitrageError as exc:
        raise _unknown(case, label, exc) from None
    return (spec,)


def _unknown(case: Case, name: str, reason: CarbitrageError) -> CarbitrageError:
    """The error for a name that is no mark, no alias and no path.

    A name with no separator in it was never meant as a path, so the walker's
    complaint about a missing field is the wrong thing to show; what the caller
    needs is the list of names that would have worked.
    """
    if "." in name or "[" in name:
        return reason
    parts = [
        f"{name!r} is not a parameter of this case.  Mark it where it is built — "
        f"Uncertain(value, {name!r}) — or name it by alias or by dotted path.",
        f"Marked in this case: {', '.join(uncertainties(case)) or 'none'}.",
    ]
    near = find(case, name)[:6]
    if near:
        parts.append(f"Paths containing {name!r}: {', '.join(near)}.")
    parts.append(f"Aliases: {', '.join(sorted(ALIASES))}.")
    return CarbitrageError("  ".join(parts))


def get_param(case: Case, name: ParamName) -> float:
    """The current value of a parameter.

    Raises:
        CarbitrageError: if the name resolves to several paths whose values
            disagree, since there would be no single "current value" to perturb
            from.
    """
    paths = resolve(case, name)
    values = [_get_path(case, path) for path in paths]
    first = values[0]
    if any(value != first for value in values):
        listed = ", ".join(f"{p}={v!r}" for p, v in zip(paths, values, strict=True))
        raise CarbitrageError(
            f"{name_of(name)!r} resolves to several parameters that currently disagree "
            f"({listed}), so it has no single base value.  Address them individually by "
            "their dotted paths."
        )
    return float(first)


def set_param(case: Case, name: ParamName, value: float) -> Case:
    """A copy of ``case`` with every path ``name`` addresses set to ``value``."""
    updated = case
    for path in resolve(case, name):
        updated = _set_path(updated, path, value)
    return updated


def scale_param(case: Case, name: ParamName, factor: float) -> Case:
    """A copy of ``case`` with every path ``name`` addresses multiplied by ``factor``.

    Scaling is the meaningful operation when an alias covers several parameters
    that legitimately differ.  ``residual_rate`` addresses the electric car's
    15 % and the petrol car's 13 %; forcing both to one number would erase a
    modelled difference, whereas scaling both by 1.2 preserves it.
    """
    updated = case
    for path in resolve(case, name):
        current = _get_path(updated, path)
        updated = _set_path(updated, path, float(current) * factor)
    return updated


def set_params(case: Case, values: Mapping[ParamName, float]) -> Case:
    """A copy of ``case`` with several parameters overridden at once."""
    updated = case
    for name, value in values.items():
        updated = set_param(updated, name, value)
    return updated


def describe_parameters(case: Case, *, limit: int | None = None) -> list[str]:
    """Every numeric parameter in the case, as dotted paths.

    Useful when a name does not resolve and the caller needs to see what is
    actually addressable.
    """
    paths = [path for path, _ in _walk(case, "")]
    return paths if limit is None else paths[:limit]


def find(case: Case, text: str) -> list[str]:
    """Every parameter path containing ``text``, ignoring case.

    The way to locate a parameter nobody marked: ``find(case, "life")`` beats
    reading all of :func:`describe_parameters` looking for it.
    """
    needle = text.lower()
    return [path for path in describe_parameters(case) if needle in path.lower()]


def uncertainties(case: Case) -> dict[str, tuple[str, ...]]:
    """Every :class:`~carbitrage.params.Uncertain` mark in the case.

    Maps each label to the paths it addresses, in tree order.  A label with
    more than one path marks several fields that move together.
    """
    out: dict[str, list[str]] = {}
    for label, path in _marks(case):
        out.setdefault(label, []).append(path)
    return {label: tuple(paths) for label, paths in out.items()}
