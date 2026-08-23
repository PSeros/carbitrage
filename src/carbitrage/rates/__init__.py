"""Rate conventions and the period grid every flow is resolved against.

A rate quoted per year has to become a rate per period before anything can be
discounted, and a growth rate quoted per year has to be spread over the periods
inside that year.  Those two conversions are the whole of this scope, with the
`Timeline` that carries them and the Fisher identities that move between
nominal and real.
"""

from __future__ import annotations

from .conventions import (
    Escalation,
    Periodisation,
    RateBasis,
    fisher_inflation,
    fisher_nominal,
    fisher_real,
)
from .timeline import Timeline

__all__ = [
    "Escalation",
    "Periodisation",
    "RateBasis",
    "Timeline",
    "fisher_inflation",
    "fisher_nominal",
    "fisher_real",
]
