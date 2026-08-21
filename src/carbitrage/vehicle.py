"""The asset itself.

A :class:`Vehicle` is pure data about the asset.  It knows nothing about how it
is paid for (that is :mod:`carbitrage.acquisition`), nothing about who buys it
(that is :class:`~carbitrage.context.Household`), and nothing about how far it is
driven (that is :class:`~carbitrage.context.Usage`).  Keeping those apart is what
lets the same vehicle appear in a purchase, a lease and a replacement chain
without being redefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from .energy import BivalentSource, Electricity, EnergySource, Hydrogen
from .errors import CarbitrageError
from .residual import ResidualValueModel

__all__ = ["Propulsion", "Vehicle", "VehicleCategory"]


class VehicleCategory(StrEnum):
    """EU vehicle classes, to the extent subsidy eligibility depends on them."""

    M1 = "M1"
    """Passenger car.  The only class the 2026 purchase premium covers."""

    N1 = "N1"
    """Light commercial vehicle."""

    L6E = "L6e"
    """Light quadricycle (Citroen Ami and similar).  Not eligible."""

    L7E = "L7e"
    """Heavy quadricycle (Microlino and similar).  Not eligible."""

    OTHER = "other"


class Propulsion(StrEnum):
    """How the vehicle is driven, to the extent incentives depend on it."""

    BEV = "bev"
    PHEV = "phev"
    FCEV = "fcev"
    ICE = "ice"


@dataclass(frozen=True)
class Vehicle:
    """Data about one asset.

    Args:
        name: Display name, used as the alternative's default label.
        price: Acquisition price.  For a used asset this is the market price
            actually paid at :attr:`age_at_acquisition`, not the as-new price.
        energy: How it is fuelled and what that costs.
        residual: The curve its terminal value is derived from.
        insurance: Annual insurance premium.
        maintenance: Annual maintenance and wear.
        annual_tax: Annual circulation tax before any exemption.  Model an
            exemption as a :class:`~carbitrage.incentive.VehicleTaxExemption`
            rather than by setting this to zero, so that a used vehicle can
            inherit a partially elapsed exemption.
        other_annual_cost: Any further recurring cost, such as a repair reserve
            for a vehicle out of warranty.
        setup_cost: One-off infrastructure needed to operate the asset, paid at
            acquisition.  A wallbox is the obvious case.  It travels with the
            vehicle, so a chain that acquires this vehicle later pays it later.
        age_at_acquisition: Age in years when acquired.  Greater than zero makes
            it a used vehicle, which matters for subsidy eligibility and for
            placing ``price`` on the residual curve.
        first_registration: Date of first registration.  Tax exemptions are
            keyed on it, not on the purchase date.
        category: EU vehicle class.
        propulsion: Overrides the propulsion inferred from ``energy``.
    """

    name: str
    price: float
    energy: EnergySource
    residual: ResidualValueModel
    insurance: float = 0.0
    maintenance: float = 0.0
    annual_tax: float = 0.0
    other_annual_cost: float = 0.0
    setup_cost: float = 0.0
    age_at_acquisition: float = 0.0
    first_registration: date | None = None
    category: VehicleCategory = VehicleCategory.M1
    propulsion: Propulsion | None = None

    def __post_init__(self) -> None:
        if self.price < 0:
            raise CarbitrageError(f"price must not be negative, got {self.price!r}")
        if self.age_at_acquisition < 0:
            raise CarbitrageError(
                f"age_at_acquisition must not be negative, got {self.age_at_acquisition!r}"
            )
        if self.setup_cost < 0:
            raise CarbitrageError(f"setup_cost must not be negative, got {self.setup_cost!r}")

    @property
    def is_used(self) -> bool:
        """Whether the asset was already in service when acquired."""
        return self.age_at_acquisition > 0.0

    @property
    def drivetrain(self) -> Propulsion:
        """The propulsion type, inferred from the energy source unless overridden."""
        if self.propulsion is not None:
            return self.propulsion
        return _infer_propulsion(self.energy)

    @property
    def fixed_annual_cost(self) -> float:
        """Insurance, maintenance, circulation tax and any other recurring cost."""
        return self.insurance + self.maintenance + self.annual_tax + self.other_annual_cost

    def residual_value(self, years_held: float) -> float:
        """Terminal value after ``years_held``, derived from :attr:`residual`."""
        return self.residual.value_after(
            self.price, years_held=years_held, age_at_acquisition=self.age_at_acquisition
        )


def _infer_propulsion(energy: EnergySource) -> Propulsion:
    if isinstance(energy, Electricity):
        return Propulsion.BEV
    if isinstance(energy, Hydrogen):
        return Propulsion.FCEV
    if isinstance(energy, BivalentSource):
        legs = (energy.primary, energy.secondary)
        if any(isinstance(leg, Electricity) for leg in legs):
            return Propulsion.PHEV
        if any(isinstance(leg, Hydrogen) for leg in legs):
            return Propulsion.FCEV
    return Propulsion.ICE
