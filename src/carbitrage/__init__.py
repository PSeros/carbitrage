"""Capital budgeting for mutually exclusive asset alternatives with unequal lives.

The question this library answers is not "what does this cost".  It is: given
several mutually exclusive options with different lifetimes, different
acquisition modes, different energy carriers and different subsidy regimes,
which one minimises the present value of outflows, and at what parameter value
does that answer flip?

Vehicles are the first-class use case; nothing in the core knows about cars.

The public API is organised into scopes, imported from here::

    import carbitrage as cb
    from carbitrage import acquisition, energy, incentives, residual

Six names that open every script stay at the top level -- :class:`Timeline`,
:class:`Vehicle`, :class:`Alternative`, :class:`Case`, :func:`compare` and
:class:`ComparisonResult`.  Everything else lives in a scope:

=========================  =====================================================
:mod:`~carbitrage.rates`         period grid, rate conventions, Fisher identities
:mod:`~carbitrage.cashflow`      one-off, recurring and terminal payments
:mod:`~carbitrage.vehicle`       the asset itself
:mod:`~carbitrage.energy`        energy carriers and cost per 100 km
:mod:`~carbitrage.residual`      what the asset is worth at the end
:mod:`~carbitrage.tax`           private and business tax treatments
:mod:`~carbitrage.incentives`    grants, quota credits, tax exemptions
:mod:`~carbitrage.acquisition`   purchase, loan, lease
:mod:`~carbitrage.context`       the buyer, the mileage, the incumbent
:mod:`~carbitrage.comparison`    building a comparison and reading the result
:mod:`~carbitrage.params`        addressing a parameter inside a case by name
:mod:`~carbitrage.sensitivity`   at what value does the answer flip
:mod:`~carbitrage.scenario`      named override bundles, expected NPV, regret
:mod:`~carbitrage.reporting`     optional xlsx and matplotlib output
:mod:`~carbitrage.errors`        the error and warning hierarchy
=========================  =====================================================

Scopes are resolved on first access, so ``import carbitrage`` does not pay for
:mod:`~carbitrage.sensitivity` (and therefore scipy) unless you reach for it.

Each scope is a directory holding the concept's implementation, or a single
module where the concept is small enough not to need one.  The dependency
layering between scopes is declared in ``tests/unit/test_layering.py`` and
enforced on every run.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .comparison.alternative import Alternative
from .comparison.case import Case, compare
from .comparison.result import ComparisonResult
from .rates.timeline import Timeline
from .vehicle import Vehicle

if TYPE_CHECKING:
    from types import ModuleType

    from . import (
        acquisition,
        cashflow,
        comparison,
        context,
        energy,
        errors,
        incentives,
        params,
        rates,
        reporting,
        residual,
        scenario,
        sensitivity,
        tax,
        vehicle,
    )

__all__ = [
    "Alternative",
    "Case",
    "ComparisonResult",
    "Timeline",
    "Vehicle",
    "acquisition",
    "cashflow",
    "compare",
    "comparison",
    "context",
    "energy",
    "errors",
    "incentives",
    "params",
    "rates",
    "reporting",
    "residual",
    "scenario",
    "sensitivity",
    "tax",
    "vehicle",
]

_SCOPES = frozenset(
    {
        "acquisition",
        "cashflow",
        "comparison",
        "context",
        "energy",
        "errors",
        "incentives",
        "params",
        "rates",
        "reporting",
        "residual",
        "scenario",
        "sensitivity",
        "tax",
        "vehicle",
    }
)


def __getattr__(name: str) -> ModuleType:
    """Resolve a scope on first access, so importing the package stays cheap."""
    if name in _SCOPES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
