"""The facade is the public API.

The 86 names below are what ``from carbitrage import ...`` promises.  Internal
layout may move freely underneath; this list may not, without a deliberate edit
here.
"""

from __future__ import annotations

import carbitrage

EXPECTED = [
    "BAFA2026",
    "LPG",
    "Acquisition",
    "Alternative",
    "BivalentSource",
    "BusinessAssets",
    "CarbitrageError",
    "Case",
    "CashFlow",
    "CashFlowSeries",
    "ComparisonResult",
    "Component",
    "Context",
    "Diesel",
    "Distribution",
    "DoubleCountingWarning",
    "Electricity",
    "EligibilityError",
    "EnergySource",
    "Escalation",
    "Evaluable",
    "Evaluation",
    "Financed",
    "FirstYearDropThenGeometric",
    "Frequency",
    "GeometricDecline",
    "Household",
    "Hydrogen",
    "Incentive",
    "InconsistentRateBasisError",
    "Incremental",
    "Incumbent",
    "Lease",
    "LogNormal",
    "MonteCarlo",
    "Normal",
    "OneOff",
    "OneWayGrid",
    "Periodisation",
    "Petrol",
    "PrivateHousehold",
    "Propulsion",
    "Purchase",
    "Range",
    "RateBasis",
    "Recurring",
    "ReplacementAgeTable",
    "ReplacementChain",
    "ResidualValueModel",
    "Scenario",
    "ScenarioAnalysis",
    "ScenarioSet",
    "SwitchPoint",
    "TabulatedResiduals",
    "TaxTreatment",
    "Terminal",
    "ThgQuote",
    "Timeline",
    "TimelineError",
    "Tornado",
    "Triangular",
    "TwoWayGrid",
    "UnequalLivesError",
    "Uniform",
    "Usage",
    "Vehicle",
    "VehicleCategory",
    "VehicleTaxExemption",
    "advantage",
    "best_margin",
    "compare",
    "fisher_inflation",
    "fisher_nominal",
    "fisher_real",
    "get_param",
    "monte_carlo",
    "npv_of",
    "one_way",
    "optimal_replacement_age",
    "resolve",
    "scale_param",
    "set_param",
    "set_params",
    "switch_point",
    "tornado",
    "two_way",
]


def test_the_public_names_are_exactly_these() -> None:
    assert list(carbitrage.__all__) == EXPECTED


def test_every_exported_name_resolves() -> None:
    missing = [n for n in carbitrage.__all__ if not hasattr(carbitrage, n)]
    assert not missing
