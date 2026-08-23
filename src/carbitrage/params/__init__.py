"""Addressing a parameter inside a case by name.

Everything in a `Case` is frozen, so perturbing one means naming a value and
getting a new case back.  Paths are dotted and may be indexed; `ALIASES` maps
short human names like ``"discount_rate"`` onto them.  This is the addressing
layer `carbitrage.sensitivity` and `carbitrage.scenario` are built on.
"""

from __future__ import annotations

from .access import describe_parameters, get_param, resolve, scale_param, set_param, set_params
from .aliases import ALIASES, FieldOf

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
