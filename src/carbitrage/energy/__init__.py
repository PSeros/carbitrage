"""Energy carriers, and what they cost per 100 km.

Each carrier turns a consumption figure and a unit price into a running cost,
applying whatever correction it needs on the way: charging losses and a
home/public split for electricity, a volumetric penalty for LPG, a real-world
factor for all of them.
"""

from __future__ import annotations

from .base import EnergySource
from .bivalent import BivalentSource
from .electricity import Electricity
from .fuels import LPG, Diesel, Hydrogen, Petrol

__all__ = [
    "LPG",
    "BivalentSource",
    "Diesel",
    "Electricity",
    "EnergySource",
    "Hydrogen",
    "Petrol",
]
