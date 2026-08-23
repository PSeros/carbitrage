"""The scopes are the public API.

``carbitrage`` exposes six spine names and fifteen scopes.  The tables below are
what ``from carbitrage import ...`` promises: internal layout may move freely
underneath, this file may not, without a deliberate edit here.

The last test is the one that earns its keep -- it fails if a public name is
added to a module inside the package and never given a scope to live in.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest

import carbitrage

PACKAGE = Path(carbitrage.__file__).parent

SPINE = [
    "Alternative",
    "Case",
    "ComparisonResult",
    "Timeline",
    "Vehicle",
    "compare",
]

SCOPES: dict[str, list[str]] = {
    "acquisition": ["Acquisition", "Financed", "Lease", "Purchase"],
    "cashflow": [
        "CashFlow",
        "CashFlowSeries",
        "Component",
        "ComponentPV",
        "Frequency",
        "OneOff",
        "Recurring",
        "Terminal",
    ],
    "comparison": [
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
    ],
    "context": ["Context", "Household", "Incumbent", "Usage"],
    "energy": [
        "LPG",
        "BivalentSource",
        "Diesel",
        "Electricity",
        "EnergySource",
        "Hydrogen",
        "Petrol",
    ],
    "errors": [
        "CarbitrageError",
        "DoubleCountingWarning",
        "EligibilityError",
        "InconsistentRateBasisError",
        "TimelineError",
        "UnequalLivesError",
    ],
    "incentives": [
        "BAFA2026",
        "BAFA_2026_BEV",
        "BAFA_2026_PHEV",
        "Incentive",
        "ThgQuote",
        "VehicleTaxExemption",
    ],
    "params": [
        "ALIASES",
        "FieldOf",
        "ParamName",
        "Spread",
        "Uncertain",
        "describe_parameters",
        "find",
        "get_param",
        "name_of",
        "resolve",
        "scale_param",
        "set_param",
        "set_params",
        "spread_of",
        "spreads",
        "uncertainties",
    ],
    "rates": [
        "Escalation",
        "Periodisation",
        "RateBasis",
        "Timeline",
        "fisher_inflation",
        "fisher_nominal",
        "fisher_real",
    ],
    "reporting": [
        "monte_carlo_plot",
        "one_way_plot",
        "ranking_plot",
        "tornado_plot",
        "write_excel",
    ],
    "residual": [
        "FirstYearDropThenGeometric",
        "GeometricDecline",
        "ResidualValueModel",
        "TabulatedResiduals",
    ],
    "scenario": ["Scenario", "ScenarioAnalysis", "ScenarioSet"],
    "sensitivity": [
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
    ],
    "tax": [
        "DEGRESSIVE_2026",
        "STRAIGHT_LINE_6Y",
        "BusinessAssets",
        "PrivateHousehold",
        "TaxTreatment",
    ],
    "vehicle": ["Propulsion", "Vehicle", "VehicleCategory"],
}

EXPECTED = sorted([*SPINE, *SCOPES])


def scope(name: str) -> ModuleType:
    return importlib.import_module(f"carbitrage.{name}")


def declared_all(path: Path) -> list[str]:
    """The module's `__all__`, read from source so no optional extra is needed."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [ast.literal_eval(e) for e in node.value.elts]  # type: ignore[attr-defined]
    return []


#: The file backing each scope: `<scope>.py`, or `<scope>/__init__.py` for a package.
SCOPE_FILES = {PACKAGE / f"{name}.py" for name in SCOPES} | {
    PACKAGE / name / "__init__.py" for name in SCOPES
}

#: Every module in the package that is neither the root nor a scope facade.
INTERNAL = sorted(
    path
    for path in PACKAGE.rglob("*.py")
    if path != PACKAGE / "__init__.py" and path not in SCOPE_FILES
)


def test_the_public_names_are_exactly_these() -> None:
    assert list(carbitrage.__all__) == EXPECTED


@pytest.mark.parametrize("name", SPINE)
def test_every_spine_name_resolves(name: str) -> None:
    assert hasattr(carbitrage, name)


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_every_scope_is_a_module_exporting_exactly_what_it_promises(name: str) -> None:
    module = getattr(carbitrage, name)
    assert isinstance(module, ModuleType)
    assert module is scope(name)
    assert list(module.__all__) == SCOPES[name]


@pytest.mark.parametrize("name", sorted(SCOPES))
def test_every_exported_name_resolves(name: str) -> None:
    module = scope(name)
    missing = [n for n in module.__all__ if not hasattr(module, n)]
    assert not missing


@pytest.mark.parametrize("name", SPINE)
def test_the_spine_is_the_same_object_as_its_scope_member(name: str) -> None:
    """The spine is a shortcut, never a second implementation."""
    owners = [s for s, names in SCOPES.items() if name in names]
    assert owners, f"{name} is on the spine but in no scope"
    for owner in owners:
        assert getattr(carbitrage, name) is getattr(scope(owner), name)


def test_reaching_for_something_that_is_not_a_scope_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = carbitrage.does_not_exist  # type: ignore[attr-defined]


@pytest.mark.parametrize("path", INTERNAL, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_no_public_name_inside_the_package_is_left_without_a_scope(path: Path) -> None:
    """A name added to an internal `__all__` must be given a home here."""
    scoped = {n for names in SCOPES.values() for n in names}
    assert not [n for n in declared_all(path) if n not in scoped]


def test_the_orphan_scan_actually_sees_something() -> None:
    """A guard against the walk silently matching nothing."""
    seen = sum(len(declared_all(path)) for path in INTERNAL)
    assert seen > 80, seen
