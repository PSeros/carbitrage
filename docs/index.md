# carbitrage documentation

Start with the [quick start](../README.md#quick-start) in the README, then:

| Page | What it covers |
| --- | --- |
| [Method](method.md) | What is and is not monetised, the same-service premise, terminology, conformal periodisation, real vs. nominal |
| [Non-negotiable rules](rules.md) | The correctness invariants, and where each is enforced |
| [API tour](api-tour.md) | What you can ask a `ComparisonResult` |
| [Traceability](traceability.md) | Where a number comes from, and why the components must sum |
| [Validation](validation.md) | What the test suite checks, and the provenance of the figures |
| [Working on the code](development.md) | Setup, the layering rule, what the tests hold in place |

## The public scopes

Six names sit directly on `carbitrage` — `Timeline`, `Vehicle`, `Alternative`,
`Case`, `compare`, `ComparisonResult`. Everything else lives in a scope:

| Scope | Holds |
| --- | --- |
| `carbitrage.rates` | `Timeline`, `RateBasis`, `Periodisation`, `Escalation`, the Fisher identities |
| `carbitrage.cashflow` | `OneOff`, `Recurring`, `Terminal`, `Component`, `CashFlowSeries` |
| `carbitrage.vehicle` | `Vehicle`, `VehicleCategory`, `Propulsion` |
| `carbitrage.energy` | `Electricity`, `Petrol`, `Diesel`, `LPG`, `Hydrogen`, `BivalentSource` |
| `carbitrage.residual` | `GeometricDecline`, `FirstYearDropThenGeometric`, `TabulatedResiduals` |
| `carbitrage.tax` | `PrivateHousehold`, `BusinessAssets`, and the depreciation schedules |
| `carbitrage.incentives` | `BAFA2026`, `ThgQuote`, `VehicleTaxExemption`, and the rate tables |
| `carbitrage.acquisition` | `Purchase`, `Financed`, `Lease` |
| `carbitrage.context` | `Household`, `Usage`, `Incumbent`, `Context` |
| `carbitrage.comparison` | `Case`, `compare`, `Alternative`, `ComparisonResult`, `ReplacementChain` |
| `carbitrage.params` | `Uncertain`, `get_param`, `set_param`, `resolve`, `find`, `describe_parameters` |
| `carbitrage.sensitivity` | `one_way`, `two_way`, `switch_point`, `tornado`, `monte_carlo`, the distributions |
| `carbitrage.scenario` | `Scenario`, `ScenarioSet`, `ScenarioAnalysis` |
| `carbitrage.reporting` | `write_excel`, `tornado_plot`, `one_way_plot`, `monte_carlo_plot`, `ranking_plot` |
| `carbitrage.errors` | The error and warning hierarchy |

```python
import carbitrage as cb
from carbitrage import acquisition, energy, incentives

ev  = cb.Vehicle("Hyundai Inster", price=23_900, energy=energy.Electricity(15.1))
buy = cb.Alternative(ev, acquisition.Purchase(), (incentives.BAFA2026(),))
```

Scopes resolve on first access, so `import carbitrage` does not pay for
`carbitrage.sensitivity` — and therefore scipy — unless you reach for it.

## How a scope is laid out

A scope is a directory holding the concept's implementation, with an
`__init__.py` that is nothing but re-exports and `__all__`:

```
carbitrage/energy/
    __init__.py       the scope's public surface
    base.py           EnergySource — what every carrier has to answer
    electricity.py    Electricity
    fuels.py          Petrol, Diesel, LPG, Hydrogen
    bivalent.py       BivalentSource
```

Where a concept is small enough not to need a directory it stays a single
module — `vehicle.py`, `residual.py`, `tax.py`, `context.py`, `errors.py`.

## The layering

Scopes are layered, and a module may import from its own scope or one below it,
never above:

| Layer | Scopes | What it is |
| ---: | --- | --- |
| 0 | `errors` | The error and warning hierarchy |
| 1 | `rates`, `cashflow` | Discounting machinery that knows nothing about cars |
| 2 | `vehicle`, `energy`, `residual`, `tax`, `incentives`, `acquisition`, `context` | The modelled world |
| 3 | `comparison` | Building a comparison and running it |
| 4 | `params`, `sensitivity`, `scenario` | At what value does the answer flip |
| 5 | `reporting` | Optional output adapters |

The layer of each scope is declared in `tests/unit/test_layering.py` and enforced
on every run. The directory tree answers *which concept a name belongs to*; the
table answers *what may depend on what*. Keeping the two separate is deliberate.
See [working on the code](development.md#the-public-api-is-scopes-not-the-layers).
