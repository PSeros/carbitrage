"""The reference workbook's base case, rebuilt in Python, plus its target numbers.

Source: ``E-Auto_Kapitalwertmodell_2026_v3.xlsx``, sheets ``Annahmen`` (inputs),
``Cashflow`` (timing), ``Foerdermatrix`` (subsidy schedule) and ``Ergebnis``
(results).  The targets below were extracted from that workbook once; the tests
never open it, so the suite stays fast and the oracle stays fixed.

Workbook conventions reproduced here:

* monthly grid, conformal rate ``i_m = (1 + i_a) ** (1/12) - 1``;
* acquisition, setup cost and incumbent disposal at t=0, running costs in
  arrears, terminal values in the final month;
* the energy price index steps annually, holding the whole first year at 1;
* the greenhouse-gas credit falls at every whole year;
* in the chain, the swap happens in month 24 and the premium is paid in month 28.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from carbitrage import (
    LPG,
    Alternative,
    BivalentSource,
    Electricity,
    GeometricDecline,
    Household,
    Incumbent,
    Lease,
    Petrol,
    Purchase,
    ReplacementChain,
    TabulatedResiduals,
    ThgQuote,
    Timeline,
    Usage,
    Vehicle,
    VehicleTaxExemption,
)
from carbitrage.incentive import BAFA2026

# --------------------------------------------------------------------- inputs

HORIZON_YEARS = 6
DISCOUNT_RATE = 0.03  # Annahmen C9
ENERGY_ESCALATION = 0.02  # Annahmen C11
VEHICLE_PRICE_ESCALATION = 0.015  # Annahmen C12
ANNUAL_KM = 12_000  # Annahmen C6

TAXABLE_INCOME = 55_000.0  # Annahmen C15
CHILDREN = 0  # Annahmen C16
SUBSIDY_LAG_MONTHS = 4  # Annahmen C19
THG_ANNUAL = 300.0  # Annahmen C20

HOME_PRICE = 0.30  # Annahmen C24
HOME_SHARE = 0.8  # Annahmen C25
PUBLIC_PRICE = 0.55  # Annahmen C26
PETROL_PRICE = 2.10  # Annahmen C28
LPG_PRICE = 0.99  # Annahmen C29

EV_PRICE = 23_900.0  # Annahmen C33
EV_WLTP = 15.1  # Annahmen C34
REAL_WORLD_FACTOR = 1.18  # Annahmen C35
EV_DEPRECIATION = 0.15  # Annahmen C37
EV_INSURANCE = 750.0  # Annahmen C39
EV_MAINTENANCE = 300.0  # Annahmen C40
WALLBOX = 1_500.0  # Annahmen C41

LEASE_MONTHLY = 239.0  # Annahmen C46
LEASE_TERM = 36  # Annahmen C47
LEASE_RENEWAL_ESCALATION = 0.05  # Annahmen C48
LEASE_INITIAL_PAYMENT = 0.0  # Annahmen C49
LEASE_INCLUDED_KM = 10_000.0  # Annahmen C50
LEASE_EXCESS_RATE = 0.12  # Annahmen C51
LEASE_INSURANCE = 850.0  # Annahmen C53
LEASE_MAINTENANCE = 200.0  # Annahmen C54

ICE_PRICE = 22_000.0  # Annahmen C57
ICE_CONSUMPTION = 6.0  # Annahmen C58
ICE_DEPRECIATION = 0.13  # Annahmen C59
ICE_INSURANCE = 650.0  # Annahmen C61
ICE_MAINTENANCE = 550.0  # Annahmen C62
ICE_TAX = 150.0  # Annahmen C63

REPAIR_COST = 2_500.0  # Annahmen C66
INCUMBENT_REMAINING_YEARS = 2.0  # Annahmen C67
INCUMBENT_PETROL_CONSUMPTION = 7.5  # Annahmen C68
LPG_VOLUMETRIC_PENALTY = 1.2  # Annahmen C69
LPG_SHARE = 0.90  # Annahmen C71
INCUMBENT_INSURANCE = 550.0  # Annahmen C73
INCUMBENT_MAINTENANCE = 900.0  # Annahmen C74
INCUMBENT_TAX = 160.0  # Annahmen C75
INCUMBENT_VALUE_IN_2Y = 800.0  # Annahmen C76
INCUMBENT_VALUE_NOW = 1_500.0  # Annahmen C77

USED_PRICE = 18_000.0  # Annahmen C86
USED_AGE = 3.0  # Annahmen C87
USED_WLTP = 16.0  # Annahmen C88
USED_DEPRECIATION = 0.11  # Annahmen C90
USED_INSURANCE = 700.0  # Annahmen C92
USED_MAINTENANCE = 450.0  # Annahmen C93
USED_REPAIR_RESERVE = 400.0  # Annahmen C94

# --------------------------------------------------------------------- targets


@dataclass(frozen=True)
class Target:
    """The workbook's computed figures for one alternative (sheet ``Ergebnis``)."""

    name: str
    npv: float
    pv_of_outflows: float
    eac: float
    cost_per_km: float
    undiscounted: float


A1 = "A1 EV bought new"
A2 = "A2 EV leased"
A3 = "A3 Petrol car bought new"
A4 = "A4 Incumbent then EV"
A5 = "A5 Used EV bought"

TARGETS: dict[str, Target] = {
    A1: Target(A1, -20844.0262343616, 20844.0262343616, 3847.75514218106, 0.320646261848422,
               -20106.846748099),
    A2: Target(A2, -21880.1650495457, 21880.1650495457, 4039.02377758347, 0.336585314798622,
               -24298.9201715365),
    A3: Target(A3, -28638.1236242331, 28638.1236242331, 5286.5260386166, 0.440543836551383,
               -28598.1024741604),
    A4: Target(A4, -22774.3895195561, 22774.3895195561, 4204.09537958876, 0.350341281632397,
               -22723.9692066771),
    A5: Target(A5, -21964.645208226, 21964.645208226, 4054.61860371347, 0.337884883642789,
               -21556.4243616811),
}

CAPITAL_RECOVERY_FACTOR = 0.184597500450177  # Ergebnis C8
WINNER = A1  # Ergebnis C13
MARGIN_OVER_RUNNER_UP = 1036.13881518417  # Ergebnis C14

#: Ergebnis rows 25-31: the A1 breakdown, which must sum to its NPV.
A1_BREAKDOWN = {
    "acquisition_and_setup": -25_400.0,
    "incumbent_disposal": 1_500.0,
    "subsidy": 3960.78178818167,
    "energy": -4313.09822689114,
    "insurance_maintenance_tax": -5765.84431268361,
    "thg_credits": 1625.15743316345,
    "residual": 7548.97708386805,
}

#: Annahmen C81/C82: the successor's escalated price and its residual at age 4.
SUCCESSOR_PRICE = 24622.3775
SUCCESSOR_RESIDUAL = 12853.0349448594

# ------------------------------------------------------------------- the case


def timeline() -> Timeline:
    """The workbook's monthly grid."""
    return Timeline(
        horizon_years=HORIZON_YEARS,
        periods_per_year=12,
        rate=DISCOUNT_RATE,
        energy_escalation=ENERGY_ESCALATION,
        vehicle_price_escalation=VEHICLE_PRICE_ESCALATION,
    )


def usage() -> Usage:
    return Usage(annual_km=ANNUAL_KM)


def household() -> Household:
    return Household(taxable_income=TAXABLE_INCOME, children=CHILDREN)


def ev_electricity(consumption: float = EV_WLTP) -> Electricity:
    """Charging at the workbook's home/public mix.

    The workbook folds charging losses into its 18 % real-world uplift, so
    ``charging_loss`` stays at zero here rather than double counting them.
    """
    return Electricity(
        consumption=consumption,
        real_world_factor=REAL_WORLD_FACTOR,
        home_price=HOME_PRICE,
        public_price=PUBLIC_PRICE,
        home_share=HOME_SHARE,
    )


def ev_vehicle(*, setup_cost: float = WALLBOX) -> Vehicle:
    return Vehicle(
        name="Hyundai Inster",
        price=EV_PRICE,
        energy=ev_electricity(),
        residual=GeometricDecline(EV_DEPRECIATION),
        insurance=EV_INSURANCE,
        maintenance=EV_MAINTENANCE,
        annual_tax=0.0,
        setup_cost=setup_cost,
        first_registration=date(2026, 1, 1),
    )


def ev_incentives(*, available: bool = True) -> tuple[BAFA2026, ThgQuote, VehicleTaxExemption]:
    return (
        BAFA2026(disbursement_lag_months=SUBSIDY_LAG_MONTHS, available=available),
        ThgQuote(annual_amount=THG_ANNUAL),
        VehicleTaxExemption(),
    )


def alternative_a1() -> Alternative:
    """Buy the EV now."""
    return Alternative(
        vehicle=ev_vehicle(),
        acquisition=Purchase(),
        incentives=ev_incentives(),
        label=A1,
    )


def alternative_a2() -> Alternative:
    """Lease the same EV, two chained 36-month contracts."""
    leased = Vehicle(
        name="Hyundai Inster",
        price=EV_PRICE,
        energy=ev_electricity(),
        residual=GeometricDecline(EV_DEPRECIATION),
        insurance=LEASE_INSURANCE,
        maintenance=LEASE_MAINTENANCE,
        annual_tax=0.0,
        setup_cost=WALLBOX,
        first_registration=date(2026, 1, 1),
    )
    return Alternative(
        vehicle=leased,
        acquisition=Lease(
            monthly_rate=LEASE_MONTHLY,
            term_months=LEASE_TERM,
            initial_payment=LEASE_INITIAL_PAYMENT,
            included_km=LEASE_INCLUDED_KM,
            excess_km_rate=LEASE_EXCESS_RATE,
            renewal_escalation=LEASE_RENEWAL_ESCALATION,
        ),
        incentives=ev_incentives(),
        label=A2,
    )


def alternative_a3() -> Alternative:
    """Buy a comparable petrol car new."""
    return Alternative(
        vehicle=Vehicle(
            name="Petrol equivalent",
            price=ICE_PRICE,
            energy=Petrol(consumption=ICE_CONSUMPTION, price=PETROL_PRICE),
            residual=GeometricDecline(ICE_DEPRECIATION),
            insurance=ICE_INSURANCE,
            maintenance=ICE_MAINTENANCE,
            annual_tax=ICE_TAX,
        ),
        acquisition=Purchase(),
        label=A3,
    )


def incumbent_vehicle() -> Vehicle:
    """The bivalent LPG car already owned: 90 % gas, 10 % petrol."""
    return Vehicle(
        name="LPG incumbent",
        price=INCUMBENT_VALUE_NOW,
        energy=BivalentSource(
            primary=LPG(
                consumption=INCUMBENT_PETROL_CONSUMPTION,
                price=LPG_PRICE,
                volumetric_penalty=LPG_VOLUMETRIC_PENALTY,
            ),
            secondary=Petrol(consumption=INCUMBENT_PETROL_CONSUMPTION, price=PETROL_PRICE),
            primary_share=LPG_SHARE,
        ),
        residual=TabulatedResiduals.from_values(
            INCUMBENT_VALUE_NOW, {INCUMBENT_REMAINING_YEARS: INCUMBENT_VALUE_IN_2Y}
        ),
        insurance=INCUMBENT_INSURANCE,
        maintenance=INCUMBENT_MAINTENANCE,
        annual_tax=INCUMBENT_TAX,
    )


def incumbent() -> Incumbent:
    return Incumbent(vehicle=incumbent_vehicle(), market_value=INCUMBENT_VALUE_NOW)


def alternative_a4() -> ReplacementChain:
    """Repair the incumbent, run out its remaining life, then buy the EV."""
    keep = Alternative(
        vehicle=incumbent_vehicle(),
        acquisition=Purchase(upfront_extra=REPAIR_COST, already_owned=True),
        life_years=INCUMBENT_REMAINING_YEARS,
        disposes_incumbent=False,
        label="LPG incumbent",
    )
    successor = Alternative(
        vehicle=ev_vehicle(),
        acquisition=Purchase(),
        incentives=ev_incentives(),
        label="Hyundai Inster (deferred)",
    )
    return ReplacementChain(keep, successor, label=A4)


def alternative_a5() -> Alternative:
    """Buy a three-year-old EV.  No purchase premium: used vehicles are excluded."""
    return Alternative(
        vehicle=Vehicle(
            name="VW ID.3 Pro, 3 years old",
            price=USED_PRICE,
            energy=ev_electricity(USED_WLTP),
            residual=GeometricDecline(USED_DEPRECIATION),
            insurance=USED_INSURANCE,
            maintenance=USED_MAINTENANCE,
            other_annual_cost=USED_REPAIR_RESERVE,
            annual_tax=0.0,
            setup_cost=WALLBOX,
            age_at_acquisition=USED_AGE,
            first_registration=date(2023, 1, 1),
        ),
        acquisition=Purchase(),
        incentives=(BAFA2026(), ThgQuote(annual_amount=THG_ANNUAL), VehicleTaxExemption()),
        label=A5,
    )


def all_alternatives() -> list[object]:
    """The five alternatives the workbook compares, in its own order."""
    return [
        alternative_a1(),
        alternative_a2(),
        alternative_a3(),
        alternative_a4(),
        alternative_a5(),
    ]
