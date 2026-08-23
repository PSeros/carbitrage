"""Grants, quota credits and tax exemptions, and who is eligible for them.

An incentive is not a discount: it has eligibility conditions, it lands at a
particular time, and several of them interact.  The rate tables are exposed so
that a changed subsidy regime can be modelled without forking the class.
"""

from __future__ import annotations

from .bafa import BAFA2026, BAFA_2026_BEV, BAFA_2026_PHEV
from .base import Incentive
from .thg import ThgQuote
from .vehicle_tax import VehicleTaxExemption

__all__ = [
    "BAFA2026",
    "BAFA_2026_BEV",
    "BAFA_2026_PHEV",
    "Incentive",
    "ThgQuote",
    "VehicleTaxExemption",
]
