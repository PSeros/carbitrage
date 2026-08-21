"""End-to-end walkthrough of the reference workbook's decision.

Run with::

    uv run python examples/workbook_case.py

It prints the ranking, the traceable breakdown of the winner, the switch points,
the tornado, a Monte Carlo run and a scenario table, then writes an .xlsx report.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The workbook base case lives with the tests, which are the single source of
# truth for it.  Adding the repository root to the path keeps this example from
# becoming a second, drifting copy of the same assumptions.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carbitrage import Case, Range, Triangular, advantage, monte_carlo, optimal_replacement_age
from carbitrage.scenario import Scenario, ScenarioSet
from tests.fixtures import workbook_base_case as wb

DEFERRED_SUBSIDY = (
    "alternatives[A4 Incumbent then EV].legs[Hyundai Inster (deferred)].incentives[0].available"
)
REPAIR = "alternatives[A4 Incumbent then EV].legs[LPG incumbent].acquisition.upfront_extra"


def main() -> None:
    case = Case(
        alternatives=tuple(wb.all_alternatives()),
        timeline=wb.timeline(),
        usage=wb.usage(),
        household=wb.household(),
        incumbent=wb.incumbent(),
    )
    result = case.run()

    print("RANKING\n")
    print(result.to_markdown(baseline=wb.A4))

    print("\n\nWHERE THE WINNER'S NUMBER COMES FROM\n")
    print(result.breakdown_markdown(result.best().name))

    print("\n\nSWITCH POINTS: buy now vs. keep the incumbent and defer\n")
    for param in ("annual_km", "lpg_price", "discount_rate", REPAIR):
        print(" -", result.switch_point_report(param, (wb.A1, wb.A4)).describe())
    print(" -", result.switch_point_report("annual_km", (wb.A1, wb.A2)).describe())

    print("\n\nWHAT DRIVES THE MARGIN\n")
    print(
        result.tornado(
            {
                "annual_km": Range(5_000, 30_000),
                "lpg_price": Range(0.85, 1.35),
                "home_electricity_price": Range(0.20, 0.45),
                "discount_rate": Range(0.0, 0.08),
                "residual_rate": Range(0.7, 1.4, relative=True),
            },
            metric=advantage(wb.A1, wb.A4),
        ).to_markdown()
    )

    print("\n\nUNCERTAINTY, WITH CORRELATED ENERGY PRICES\n")
    simulation = monte_carlo(
        case,
        {
            "lpg_price": Triangular(0.80, 0.99, 1.40),
            "home_electricity_price": Triangular(0.22, 0.30, 0.45),
        },
        between=(wb.A1, wb.A4),
        n=1_000,
        correlation=[[1.0, 0.6], [0.6, 1.0]],
        seed=20260821,
    )
    print(simulation.describe())

    print("\n\nSCENARIOS\n")
    overlays = ScenarioSet(
        [
            Scenario("fuel collapse", {"lpg_price": 0.65, "petrol_price": 1.40}, probability=0.15),
            Scenario("subsidy expires", {DEFERRED_SUBSIDY: 0}, probability=0.30),
            Scenario("incumbent dies early", {REPAIR: 6_000.0}, probability=0.25),
        ]
    )
    analysis = overlays.run(case)
    print(analysis.to_markdown())

    print("\n\nHOW LONG TO KEEP THE EV (equivalent annual cost by holding period)\n")
    table = optimal_replacement_age(
        wb.alternative_a1(),
        timeline=wb.timeline(),
        candidates=[3, 4, 5, 6, 8, 10, 12],
        usage=wb.usage(),
        household=wb.household(),
    )
    for age, _pv, eac in table.as_rows():
        print(f"  {age:>4.0f} years   EAC {eac:>10,.0f}")
    print(f"  -> cheapest at {table.best_age:g} years")

    destination = Path("carbitrage-report.xlsx")
    try:
        from carbitrage.io import write_excel

        written = write_excel(result, destination, baseline=wb.A4, scenarios=analysis)
        print(f"\n\nWrote {written}")
    except ImportError:
        print("\n\n(install the 'excel' extra to also write an .xlsx report)")


if __name__ == "__main__":
    main()
