"""Short human names for the paths people actually reach for."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..energy import LPG, Diesel, Electricity, Hydrogen, Petrol
from ..residual import FirstYearDropThenGeometric, GeometricDecline

if TYPE_CHECKING:  # pragma: no cover
    pass

__all__ = [
    "ALIASES",
    "FieldOf",
]


@dataclass(frozen=True)
class FieldOf:
    """An alias that matches one field on every instance of a type in the tree.

    ``FieldOf(LPG, "price")`` addresses the autogas price wherever it occurs,
    which is what a user means by "the LPG price".
    """

    owner: type
    field: str

    def matches(self, obj: object, field: str) -> bool:
        return isinstance(obj, self.owner) and field == self.field


#: Human-friendly names for the parameters people actually vary.
ALIASES: dict[str, str | FieldOf] = {
    # Capital market and horizon
    "discount_rate": "timeline.rate",
    "horizon_years": "timeline.horizon_years",
    "energy_escalation": "timeline.energy_escalation",
    "vehicle_price_escalation": "timeline.vehicle_price_escalation",
    "cost_escalation": "timeline.cost_escalation",
    # Usage and household
    "annual_km": "usage.annual_km",
    "taxable_income": "household.taxable_income",
    # Energy prices, wherever they occur
    "lpg_price": FieldOf(LPG, "price"),
    "petrol_price": FieldOf(Petrol, "price"),
    "diesel_price": FieldOf(Diesel, "price"),
    "hydrogen_price": FieldOf(Hydrogen, "price"),
    "home_electricity_price": FieldOf(Electricity, "home_price"),
    "public_electricity_price": FieldOf(Electricity, "public_price"),
    "home_share": FieldOf(Electricity, "home_share"),
    "real_world_factor": FieldOf(Electricity, "real_world_factor"),
    # Residual value assumptions
    "residual_rate": FieldOf(GeometricDecline, "rate"),
    "first_year_drop": FieldOf(FirstYearDropThenGeometric, "drop"),
}
