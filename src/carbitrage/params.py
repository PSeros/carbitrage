"""Addressing and overriding parameters inside an immutable model.

Sensitivity analysis has to answer "what if the LPG price were 1.20" without the
caller rebuilding the model by hand.  Since every object in the library is a
frozen dataclass, that means walking the tree, rebuilding the branch that
changed, and leaving everything else shared.

Two ways to name a parameter:

* an explicit **dotted path**, e.g. ``"timeline.rate"`` or
  ``"alternatives[A1 EV bought new].vehicle.price"``;
* a registered **alias**, e.g. ``"discount_rate"``, ``"annual_km"`` or
  ``"lpg_price"``.

An alias may resolve to several paths at once, and that is deliberate: the LPG
price appears in every alternative that burns it, and a sensitivity run that
moved only one of them would be answering a different question than the one
asked.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .energy import LPG, Diesel, Electricity, Hydrogen, Petrol
from .errors import CarbitrageError
from .residual import FirstYearDropThenGeometric, GeometricDecline

if TYPE_CHECKING:  # pragma: no cover
    from .analysis import Case

__all__ = [
    "ALIASES",
    "FieldOf",
    "describe_parameters",
    "get_param",
    "resolve",
    "scale_param",
    "set_param",
    "set_params",
]

_INDEX = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[(?P<key>.*)\]$")


@dataclass(frozen=True)
class FieldOf:
    """An alias that matches one field on every instance of a type in the tree.

    ``FieldOf(LPG, "price")`` addresses the autogas price wherever it occurs,
    which is what a user means by "the LPG price".
    """

    owner: type
    field: str

    def matches(self, obj: object, field: str) -> bool:
        return isinstance(obj, self.owner) and field == self.field


#: Human-friendly names for the parameters people actually vary.
ALIASES: dict[str, str | FieldOf] = {
    # Capital market and horizon
    "discount_rate": "timeline.rate",
    "horizon_years": "timeline.horizon_years",
    "energy_escalation": "timeline.energy_escalation",
    "vehicle_price_escalation": "timeline.vehicle_price_escalation",
    "cost_escalation": "timeline.cost_escalation",
    # Usage and household
    "annual_km": "usage.annual_km",
    "taxable_income": "household.taxable_income",
    # Energy prices, wherever they occur
    "lpg_price": FieldOf(LPG, "price"),
    "petrol_price": FieldOf(Petrol, "price"),
    "diesel_price": FieldOf(Diesel, "price"),
    "hydrogen_price": FieldOf(Hydrogen, "price"),
    "home_electricity_price": FieldOf(Electricity, "home_price"),
    "public_electricity_price": FieldOf(Electricity, "public_price"),
    "home_share": FieldOf(Electricity, "home_share"),
    "real_world_factor": FieldOf(Electricity, "real_world_factor"),
    # Residual value assumptions
    "residual_rate": FieldOf(GeometricDecline, "rate"),
    "first_year_drop": FieldOf(FirstYearDropThenGeometric, "drop"),
}


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


# --------------------------------------------------------------------- walking


def _is_node(obj: object) -> bool:
    """Whether the walker should descend into ``obj``."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _children(obj: Any) -> Iterator[tuple[str, Any]]:
    """Field name (or indexed name) and value for everything below ``obj``."""
    for f in dataclasses.fields(obj):
        if not f.init and f.name.startswith("_"):
            continue
        value = getattr(obj, f.name)
        if isinstance(value, tuple):
            for i, item in enumerate(value):
                key = getattr(item, "name", None)
                label = key if isinstance(key, str) else str(i)
                yield f"{f.name}[{label}]", item
        else:
            yield f.name, value


def _walk(obj: Any, prefix: str) -> Iterator[tuple[str, float]]:
    """Yield ``(path, value)`` for every numeric leaf reachable from ``obj``."""
    for name, value in _children(obj):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            yield path, float(value)
        elif _is_node(value):
            yield from _walk(value, path)


def _paths_matching(obj: Any, spec: FieldOf, prefix: str = "") -> Iterator[str]:
    """Every path whose owner and field name satisfy ``spec``."""
    for name, value in _children(obj):
        path = f"{prefix}.{name}" if prefix else name
        bare = name.split("[", 1)[0]
        if spec.matches(obj, bare):
            yield path
        elif _is_node(value):
            yield from _paths_matching(value, spec, path)


# -------------------------------------------------------------------- access


def _split(path: str) -> list[str]:
    if not path:
        raise CarbitrageError("an empty parameter path addresses nothing")
    return path.split(".")


def _child(obj: Any, segment: str) -> Any:
    """Resolve one path segment, whether plain or indexed."""
    match = _INDEX.match(segment)
    if match is None:
        if not hasattr(obj, segment):
            raise CarbitrageError(
                f"{type(obj).__name__} has no field {segment!r}; "
                f"it has {[f.name for f in dataclasses.fields(obj)]}"
            )
        return getattr(obj, segment)
    name, key = match.group("name"), match.group("key")
    container = getattr(obj, name, None)
    if not isinstance(container, tuple):
        raise CarbitrageError(f"{type(obj).__name__}.{name} is not an indexable sequence")
    return _index(container, key, f"{type(obj).__name__}.{name}")


def _index(container: tuple[Any, ...], key: str, where: str) -> Any:
    if key.isdigit():
        i = int(key)
        if i >= len(container):
            raise CarbitrageError(f"{where} has {len(container)} items; index {i} is out of range")
        return container[i]
    for item in container:
        if getattr(item, "name", None) == key:
            return item
    available = [getattr(item, "name", str(i)) for i, item in enumerate(container)]
    raise CarbitrageError(f"{where} has no item named {key!r}; available: {available}")


def _get_path(obj: Any, path: str) -> Any:
    node = obj
    for segment in _split(path):
        node = _child(node, segment)
    return node


def _rebuild(obj: Any, field: str, value: Any) -> Any:
    """``obj`` with one field replaced, honouring any custom constructor.

    Classes whose ``__init__`` does not take their fields by name — such as
    :class:`~carbitrage.chain.ReplacementChain` — provide ``__replace__``.
    """
    replacer = getattr(type(obj), "__replace__", None)
    if replacer is not None and "__replace__" in vars(type(obj)):
        return replacer(obj, **{field: value})
    return dataclasses.replace(obj, **{field: value})


def _declared_type(owner: Any, field_name: str) -> str | None:
    """The *declared* type of a field, as written in the annotation."""
    fields = getattr(type(owner), "__dataclass_fields__", None)
    if not fields or field_name not in fields:
        return None
    declared = fields[field_name].type
    return declared if isinstance(declared, str) else getattr(declared, "__name__", None)


def _coerce(owner: Any, field_name: str, current: Any, value: Any) -> Any:
    """Keep a field's declared type when a numeric override is written into it.

    Scenario overlays are written as plain numbers — ``{"...available": 0}`` to
    switch a flag off — but a field annotated ``bool`` should not end up holding
    a float, and a count of children should stay an integer.

    The *declared* type decides, never the current value's runtime type.  A field
    annotated ``float`` that happens to hold ``12000`` because it was
    constructed from an integer literal is still a continuous parameter, and
    rounding it would turn a smooth differential into a step function and
    silently defeat the root finder.
    """
    declared = _declared_type(owner, field_name)
    if declared == "bool" or (declared is None and isinstance(current, bool)):
        return bool(value)
    if declared == "int":
        return round(float(value))
    return value


def _set_path(obj: Any, path: str, value: Any) -> Any:
    """A copy of ``obj`` with ``path`` set to ``value``, sharing untouched branches."""
    segments = _split(path)
    head, rest = segments[0], segments[1:]
    match = _INDEX.match(head)
    if match is None:
        child = _child(obj, head)
        new_child = (
            _coerce(obj, head, child, value)
            if not rest
            else _set_path(child, ".".join(rest), value)
        )
        return _rebuild(obj, head, new_child)
    name, key = match.group("name"), match.group("key")
    container = getattr(obj, name)
    if not isinstance(container, tuple):
        raise CarbitrageError(f"{type(obj).__name__}.{name} is not an indexable sequence")
    target = _index(container, key, f"{type(obj).__name__}.{name}")
    new_target = (
        _coerce(obj, name, target, value)
        if not rest
        else _set_path(target, ".".join(rest), value)
    )
    replaced = tuple(new_target if item is target else item for item in container)
    return _rebuild(obj, name, replaced)
