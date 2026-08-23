"""How the asset is paid for.

The same vehicle bought outright, financed on an annuity loan, or leased is
three different cash-flow profiles against the same service, which is what makes
the comparison non-trivial.
"""

from __future__ import annotations

from .base import Acquisition
from .financed import Financed
from .lease import Lease
from .purchase import Purchase

__all__ = [
    "Acquisition",
    "Financed",
    "Lease",
    "Purchase",
]
