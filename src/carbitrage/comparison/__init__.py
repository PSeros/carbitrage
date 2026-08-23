"""Building a comparison, running it, and reading what comes back.

A `Case` is the whole comparison as one immutable value, which is what lets
`carbitrage.sensitivity` re-run it under perturbation.  `compare` is the
one-shot form.  Both return a `ComparisonResult`.  When alternatives have
unequal lives, a `ReplacementChain` repeats them until the lives match.
"""

from __future__ import annotations

from .age import ReplacementAgeTable, optimal_replacement_age
from .alternative import Alternative, Evaluable
from .case import Case, compare
from .chain import ReplacementChain
from .evaluation import Evaluation, Incremental
from .result import ComparisonResult

__all__ = [
    "Alternative",
    "Case",
    "ComparisonResult",
    "Evaluable",
    "Evaluation",
    "Incremental",
    "ReplacementAgeTable",
    "ReplacementChain",
    "compare",
    "optimal_replacement_age",
]
