"""Reading and overriding a parameter by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import CarbitrageError
from .aliases import ALIASES, FieldOf
from .paths import _get_path, _paths_matching, _set_path, _walk

if TYPE_CHECKING:  # pragma: no cover
    from ..comparison import Case

__all__ = [
    "describe_parameters",
    "get_param",
    "resolve",
    "scale_param",
    "set_param",
    "set_params",
]


def resolve(case: Case, name: str) -> tuple[str, ...]:
    """Every concrete dotted path ``name`` addresses, in tree order.

    Raises:
        CarbitrageError: if the name matches nothing.
    """
    spec = ALIASES.get(name, name)
    if isinstance(spec, FieldOf):
        paths = tuple(_paths_matching(case, spec))
        if not paths:
            raise CarbitrageError(
                f"alias {name!r} matches no {spec.owner.__name__} in this case.  "
                f"Available parameters: {', '.join(describe_parameters(case)[:12])} ..."
            )
        return paths
    _get_path(case, spec)  # raises if the path is wrong
    return (spec,)


def get_param(case: Case, name: str) -> float:
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
            f"{name!r} resolves to several parameters that currently disagree ({listed}), so it "
            "has no single base value.  Address them individually by their dotted paths."
        )
    return float(first)


def set_param(case: Case, name: str, value: float) -> Case:
    """A copy of ``case`` with every path ``name`` addresses set to ``value``."""
    updated = case
    for path in resolve(case, name):
        updated = _set_path(updated, path, value)
    return updated


def scale_param(case: Case, name: str, factor: float) -> Case:
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


def set_params(case: Case, values: dict[str, float]) -> Case:
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
