"""End-to-end walkthrough of the decision the README quick start sets up.

Run with::

    uv run python examples/workbook_case.py

It prints the ranking, the traceable breakdown of the winner, the switch points,
the tornado, a Monte Carlo run and a scenario table, then writes an .xlsx report.

The case is built inline rather than imported: an example that cannot be read
top to bottom is not an example.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from carbitrage import Alternative, Case, Timeline, Vehicle
from carbitrage.acquisition import Lease, Purchase
from carbitrage.comparison import ReplacementChain, optimal_replacement_age
from carbitrage.context import Household, Incumbent, Usage
from carbitrage.energy import LPG, BivalentSource, Electricity, Petrol
from carbitrage.incentives import BAFA2026, ThgQuote, VehicleTaxExemption
from carbitrage.params import Uncertain
from carbitrage.residual import GeometricDecline, TabulatedResiduals
from carbitrage.scenario import Scenario, ScenarioSet
from carbitrage.sensitivity import Range, Triangular, advantage, monte_carlo

BUY = "Buy the EV now"
LEASE = "Lease the EV"
DEFER = "Repair now, replace in 2 years"

#: The repair bill is the number this decision turns on, so it is marked where
#: it is written and addressed by name everywhere below.  The subsidy flag shows
#: the other way of naming a parameter, by its dotted path.
REPAIR = Uncertain(2_500, "repair_bill")
DEFERRED_SUBSIDY = f"alternatives[{DEFER}].legs[Buy the EV in 2 years].incentives[0].available"


def build_case() -> Case:
    timeline = Timeline(
        horizon_years=6,
        periods_per_year=12,
        rate=0.03,
        energy_escalation=0.02,
        vehicle_price_escalation=0.015,
    )
    household = Household(taxable_income=55_000, children=0)
    usage = Usage(annual_km=12_000)

    ev = Vehicle(
        "Hyundai Inster",
        price=23_900,
        energy=Electricity(
            15.1, real_world_factor=1.18, home_price=0.30, public_price=0.55, home_share=0.8
        ),
        residual=GeometricDecline(0.15),
        insurance=750,
        maintenance=300,
        setup_cost=1_500,  # the wallbox
        first_registration=date(2026, 1, 1),
    )
    incentives = (BAFA2026(), ThgQuote(300), VehicleTaxExemption())

    buy = Alternative(ev, Purchase(), incentives, label=BUY)
    lease = Alternative(
        ev,
        Lease(
            239,
            term_months=36,
            included_km=10_000,
            excess_km_rate=0.12,
            renewal_escalation=0.05,
        ),
        incentives,
        label=LEASE,
    )

    old = Vehicle(
        "LPG incumbent",
        price=1_500,  # its current market value
        energy=BivalentSource(
            LPG(7.5, price=0.99, volumetric_penalty=1.2),
            Petrol(7.5, price=2.10),
            primary_share=0.90,
        ),
        residual=TabulatedResiduals.from_values(1_500, {2: 800}),
        insurance=550,
        maintenance=900,
        annual_tax=160,
    )
    defer = ReplacementChain(
        Alternative(
            old,
            Purchase(upfront_extra=REPAIR, already_owned=True),
            life_years=2,
            disposes_incumbent=False,
            label="Keep the incumbent",
        ),
        Alternative(ev, Purchase(), incentives, label="Buy the EV in 2 years"),
        label=DEFER,
    )

    return Case(
        alternatives=(buy, lease, defer),
        timeline=timeline,
        usage=usage,
        household=household,
        incumbent=Incumbent(old, market_value=1_500),
    )


def main() -> None:
    case = build_case()
    result = case.run()

    print("RANKING\n")
    print(result.to_markdown(baseline=DEFER))

    print("\n\nWHERE THE WINNER'S NUMBER COMES FROM\n")
    print(result.breakdown_markdown(result.best().name))

    print("\n\nSWITCH POINTS: buy now vs. keep the incumbent and defer\n")
    for param in ("annual_km", "lpg_price", "discount_rate", REPAIR):
        print(" -", result.switch_point_report(param, (BUY, DEFER)).describe())
    print(" -", result.switch_point_report("annual_km", (BUY, LEASE)).describe())

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
            metric=advantage(BUY, DEFER),
        ).to_markdown()
    )

    print("\n\nUNCERTAINTY, WITH CORRELATED ENERGY PRICES\n")
    simulation = monte_carlo(
        case,
        {
            "lpg_price": Triangular(0.80, 0.99, 1.40),
            "home_electricity_price": Triangular(0.22, 0.30, 0.45),
        },
        between=(BUY, DEFER),
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
        case.alternatives[0],
        timeline=case.timeline,
        candidates=[3, 4, 5, 6, 8, 10, 12],
        usage=case.usage,
        household=case.household,
    )
    for age, _pv, eac in table.as_rows():
        print(f"  {age:>4.0f} years   EAC {eac:>10,.0f}")
    print(f"  -> cheapest at {table.best_age:g} years")

    destination = Path("carbitrage-report.xlsx")
    try:
        from carbitrage.reporting import write_excel

        written = write_excel(result, destination, baseline=DEFER, scenarios=analysis)
        print(f"\n\nWrote {written}")
    except ImportError:
        print("\n\n(install the 'excel' extra to also write an .xlsx report)")


if __name__ == "__main__":
    main()
