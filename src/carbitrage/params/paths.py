"""Walking a frozen case tree, and rebuilding it with one value changed."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ..errors import CarbitrageError
from .aliases import FieldOf
from .marks import Uncertain, _remark

if TYPE_CHECKING:  # pragma: no cover
    pass

_INDEX = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[(?P<key>.*)\]$")

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


def _marks(obj: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(label, path)`` for every marked value reachable from ``obj``."""
    for name, value in _children(obj):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, Uncertain):
            yield value.label, path
        elif _is_node(value):
            yield from _marks(value, path)


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
    :class:`~carbitrage.engine.chain.ReplacementChain` — provide ``__replace__``.
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

    A mark is kept, so that a case which has been swept can be swept again —
    except on those fields, where the declared type wins.

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
    if isinstance(current, Uncertain):
        return _remark(value, current)
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
        _coerce(obj, name, target, value) if not rest else _set_path(target, ".".join(rest), value)
    )
    replaced = tuple(new_target if item is target else item for item in container)
    return _rebuild(obj, name, replaced)
