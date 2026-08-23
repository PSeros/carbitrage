"""Addressing a parameter inside a case by name.

Everything in a `Case` is frozen, so perturbing one means naming a value and
getting a new case back.  There are three ways to name one, and `resolve` tries
them in this order:

* an `Uncertain` mark, written where the value itself is written, which is the
  way that needs no knowledge of where the value sits in the tree;
* an entry in `ALIASES`, the short human names for the parameters most cases
  share, such as ``"discount_rate"``;
* a dotted path, which may be indexed by name or position, such as
  ``"alternatives[Renault].legs[0].acquisition.upfront_extra"``.

`find` and `uncertainties` are the discovery half: what is addressable, and
what this case has already marked.  This is the addressing layer
`carbitrage.sensitivity` and `carbitrage.scenario` are built on.
"""

from __future__ import annotations

from .access import (
    describe_parameters,
    find,
    get_param,
    resolve,
    scale_param,
    set_param,
    set_params,
    spread_of,
    uncertainties,
)
from .aliases import ALIASES, FieldOf
from .marks import ParamName, Spread, Uncertain, name_of

__all__ = [
    "ALIASES",
    "FieldOf",
    "ParamName",
    "Spread",
    "Uncertain",
    "describe_parameters",
    "find",
    "get_param",
    "name_of",
    "resolve",
    "scale_param",
    "set_param",
    "set_params",
    "spread_of",
    "uncertainties",
]
