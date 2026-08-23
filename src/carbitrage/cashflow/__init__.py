"""Payments, and the labelled series they aggregate into.

Every cost in a comparison is a `OneOff`, a `Recurring` stream or a `Terminal`
value, and each carries a `Component` label.  The labels are what make a present
value decomposable afterwards rather than one opaque number.
"""

from __future__ import annotations

from .base import CashFlow, Component, Frequency
from .flows import OneOff, Recurring, Terminal
from .series import CashFlowSeries, ComponentPV

__all__ = [
    "CashFlow",
    "CashFlowSeries",
    "Component",
    "ComponentPV",
    "Frequency",
    "OneOff",
    "Recurring",
    "Terminal",
]
