"""Capital budgeting for mutually exclusive asset alternatives with unequal lives.

The question this library answers is not "what does this cost".  It is: given
several mutually exclusive options with different lifetimes, different
acquisition modes, different energy carriers and different subsidy regimes,
which one minimises the present value of outflows, and at what parameter value
does that answer flip?

Vehicles are the first-class use case; nothing in the core knows about cars.
"""

from __future__ import annotations

from .core.cashflow import (
    CashFlow,
    CashFlowSeries,
    Component,
    Frequency,
    OneOff,
    Recurring,
    Terminal,
)
from .core.timeline import (
    Escalation,
    Periodisation,
    RateBasis,
    Timeline,
    fisher_inflation,
    fisher_nominal,
    fisher_real,
)
from .domain.acquisition import Acquisition, Financed, Lease, Purchase
from .domain.context import Context, Household, Incumbent, Usage
from .domain.energy import (
    LPG,
    BivalentSource,
    Diesel,
    Electricity,
    EnergySource,
    Hydrogen,
    Petrol,
)
from .domain.incentive import BAFA2026, Incentive, ThgQuote, VehicleTaxExemption
from .domain.residual import (
    FirstYearDropThenGeometric,
    GeometricDecline,
    ResidualValueModel,
    TabulatedResiduals,
)
from .domain.tax import BusinessAssets, PrivateHousehold, TaxTreatment
from .domain.vehicle import Propulsion, Vehicle, VehicleCategory
from .engine.alternative import Alternative, Evaluable
from .engine.chain import ReplacementAgeTable, ReplacementChain, optimal_replacement_age
from .engine.comparison import Case, compare
from .engine.result import ComparisonResult, Evaluation, Incremental
from .errors import (
    CarbitrageError,
    DoubleCountingWarning,
    EligibilityError,
    InconsistentRateBasisError,
    TimelineError,
    UnequalLivesError,
)
from .study.params import get_param, resolve, scale_param, set_param, set_params
from .study.scenario import Scenario, ScenarioAnalysis, ScenarioSet
from .study.sensitivity import (
    Distribution,
    LogNormal,
    MonteCarlo,
    Normal,
    OneWayGrid,
    Range,
    SwitchPoint,
    Tornado,
    Triangular,
    TwoWayGrid,
    Uniform,
    advantage,
    best_margin,
    monte_carlo,
    npv_of,
    one_way,
    switch_point,
    tornado,
    two_way,
)

__all__ = [
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
