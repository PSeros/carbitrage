"""At what value does the answer flip.

Sweep one parameter or two, solve for the value at which the ranking changes,
rank the drivers by how much they move the margin, or sample the joint
uncertainty and report how often each option wins.
"""

from __future__ import annotations

from .distributions import Distribution, LogNormal, Normal, Triangular, Uniform
from .grids import OneWayGrid, TwoWayGrid, one_way, two_way
from .metrics import Metric, advantage, best_margin, npv_of
from .montecarlo import MonteCarlo, monte_carlo
from .spec import Range
from .switching import NoSwitchPoint, SwitchPoint, switch_point, switch_point_report
from .tornado import Tornado, TornadoBar, tornado

__all__ = [
    "Distribution",
    "LogNormal",
    "Metric",
    "MonteCarlo",
    "NoSwitchPoint",
    "Normal",
    "OneWayGrid",
    "Range",
    "SwitchPoint",
    "Tornado",
    "TornadoBar",
    "Triangular",
    "TwoWayGrid",
    "Uniform",
    "advantage",
    "best_margin",
    "monte_carlo",
    "npv_of",
    "one_way",
    "switch_point",
    "switch_point_report",
    "tornado",
    "two_way",
]
