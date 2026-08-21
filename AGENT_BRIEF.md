# `carbitrage` — Agent Brief

## What you are building

A small, dependency-light Python library for **capital budgeting of mutually exclusive
asset alternatives** — with vehicles as the first-class use case, but nothing in the core
may hard-code "car".

The question the library answers is never "what does this cost". It is:

> Given several mutually exclusive options with **different lifetimes**, different
> acquisition modes, different energy carriers and different subsidy regimes — which one
> minimises the present value of outflows, and **at what parameter value does that answer
> flip**?

An Excel workbook (`E-Auto_Kapitalwertmodell_2026_v3.xlsx`) ships with this repo. It is a
working reference implementation of the same problem: German EV vs. LPG-bivalent
incumbent vs. lease vs. used EV, 2026 BAFA subsidy matrix, 121-period monthly cash flow
grid. **Read it before writing code.** Its `Annahmen` sheet is the input taxonomy, its
`Cashflow` sheet is the timing convention, its `Ergebnis` and `Sensitivitaet` sheets are
the output contract. Its computed values are your regression oracle (see Testing).

## Project setup

- `uv init --lib`, src layout, package name `carbitrage`, Python ≥ 3.11.
- Core runtime deps: `numpy`, `scipy` (root finding only). Nothing else in the core.
- Optional extras: `excel` → `openpyxl`; `viz` → `matplotlib`; `frames` → `pandas`.
- Dev: `pytest`, `hypothesis`, `ruff`, `mypy --strict`.
- Full type annotations. Every public class is either a `@dataclass(frozen=True)` or an
  ABC. No mutable global state, no config singletons.

## Non-negotiable financial rules

These are correctness requirements, not style preferences. Encode them in the type system
or in runtime checks where possible, and state them in the docstrings.

1. **Benefit equivalence is the premise — not an absence of inflows.**
   Inflows do occur: grants, recurring credits, disposal proceeds of the incumbent at
   t=0, terminal residual value. They are netted inside the stream, never dropped. What
   is absent is *operating revenue attributable to the asset* — the vehicle earns
   nothing.
   What licenses deciding on cost alone is the assumption that every alternative
   delivers **the same service over the same horizon**. That assumption is doing the
   real work here, and it is fragile: a larger vehicle or one with materially different
   range is not the same service, and the comparison is then invalid rather than merely
   incomplete. State this in the README next to the "not monetised" section — the two
   are the same point seen from opposite sides.
2. **Decision rule: maximise NPV of the net stream** (equivalently: minimise the PV of
   net outflows). Do **not** rank by absolute value. A cheap vehicle with a large grant,
   recurring credits and a high residual can produce a positive NPV, and an
   `abs()`-based ranking silently inverts the order when it does. Sign convention:
   outflow negative throughout the cash-flow layer; flip only at presentation.
   The same caveat propagates to EAC and cost-per-km, which are affine transforms of the
   NPV and inherit its sign.
   Terminology note for any German-language docs: this is a *Kapitalwertvergleich* or
   *Barwertminimierung*. "Kostenvergleichsrechnung" denotes the **static**, undiscounted
   method in German cost accounting and is the wrong label for what this library does.
3. **Unequal lives must never be compared by raw NPV.** Two remedies, both supported:
   equalise via a replacement chain to a common horizon, or compare *equivalent annual
   cost* (EAC). If the user passes alternatives with unequal lives, no explicit horizon
   and no chain, **raise** — do not silently truncate.
4. **EAC** = PV × capital recovery factor, CRF = `i(1+i)^n / ((1+i)^n − 1)`.
5. **Conformal periodisation.** Sub-annual rate is `(1+i_a)^(1/m) − 1`, not `i_a/m`.
   Allow the user to opt into nominal/proportional conversion explicitly, but never
   default to it.
6. **Real vs. nominal must not be mixed.** Either escalate cash flows with growth rates
   and discount at a nominal rate, or use real flows and a real rate. Provide a Fisher
   helper (`(1+i_nom) = (1+i_real)(1+π)`) and validate consistency at construction time.
7. **Sunk costs are excluded; opportunity costs are not.** The incumbent asset's current
   market value is an inflow at t=0 for every alternative that disposes of it, and an
   implicit opportunity cost for the alternative that keeps it. Past spend on the
   incumbent never enters.
8. **Terminal value** is discounted at the horizon, and must be derived from a residual
   value model — not entered as a free parameter — so that assets of *different ages*
   (e.g. the replacement vehicle in a chain) are valued consistently.
9. **Taxes**: default is a private household — no depreciation, no VAT recovery, subsidies
   tax-free. Do not bake that in; put it behind a `TaxTreatment` ABC so a business case
   (degressive depreciation, input VAT, company-car benefit-in-kind) can be added later
   without touching the engine.

## Architecture

Composition over inheritance for the domain objects; ABCs only where genuine polymorphism
exists. Every ABC below defines a narrow protocol and gets at least two concrete
implementations, so the abstraction is proven rather than speculative.

### `timeline.py` — the part that will bite you

A `Timeline` owns period indexing, discount factors and escalation indices. Every cash
flow is resolved against it. This is the module that makes the rest simple, so build it
first and test it hardest.

It must handle, in one series: one-off flows at t=0, recurring monthly flows, recurring
annual flows, flows that start or stop mid-horizon (a subsidy paid in month 4; a vehicle
swap in month 24), and flows that escalate at their own growth rate. The user must never
index into arrays by hand.

```python
Timeline(horizon_years=6, periods_per_year=12, rate=0.03)
  .discount_factor(t) -> float
  .escalation(t, g)   -> float
  .index_of(years=2.0) -> int
```

### `cashflow.py`

`CashFlow` protocol with `.amounts(timeline) -> np.ndarray`. Concrete: `OneOff`,
`Recurring(amount, frequency, growth, start, end)`, `Terminal`. A `CashFlowSeries`
aggregates, keeps a **labelled breakdown** (acquisition / energy / insurance /
maintenance / tax / subsidy / residual), and exposes `.pv(timeline)`. The breakdown is
what makes the result explainable; do not collapse it early.

### `energy.py`

```python
class EnergySource(ABC):
    consumption: float                      # per 100 km, in the source's own unit
    def unit_cost(self, t: int, timeline) -> float: ...
    def cost_per_100km(self, t, timeline) -> float: ...
```

Concrete: `Electricity` (home/public price split, share of home charging, **charging-loss
factor applied on top of nominal consumption**), `Petrol`, `Diesel`, `LPG`
(carries its own volumetric penalty factor vs. petrol), `Hydrogen`.

Add a `BivalentSource(primary, secondary, primary_share)` composite — the incumbent in the
reference workbook runs 90 % LPG / 10 % petrol, and this must be expressible without a
bespoke class. Real-world uplift over WLTP/NEDC belongs here too, as a factor on
consumption.

### `residual.py`

```python
class ResidualValueModel(ABC):
    def value(self, price: float, age_years: float) -> float: ...
```

Concrete: `GeometricDecline(rate)`, `FirstYearDropThenGeometric(drop, rate)` — EVs lose
disproportionately in year one and a constant rate understates it — and
`TabulatedResiduals(mapping)` for Schwacke/DAT-style curves. Taking `age`, not `year in
horizon`, is what makes replacement chains consistent.

### `incentive.py`

```python
class Incentive(ABC):
    def is_eligible(self, vehicle, household) -> bool: ...
    def flows(self, vehicle, household, timeline) -> CashFlowSeries: ...
```

One protocol covers one-off grants, recurring credits and tax exemptions. Concrete:

- `BAFA2026` — income × children lookup matrix, means-tested; new vehicles only, so
  `is_eligible` must reject used and non-M1 vehicles; disbursed with a lag after
  registration; carries a `minimum_holding_period` that the engine surfaces as a
  constraint, not a cash flow.
- `ThgQuote` — recurring annual credit, BEV only.
- `VehicleTaxExemption` — suppresses an otherwise-recurring cost, keyed on *first
  registration date*, with a cap on years. Modelling it as an incentive rather than a
  hard-coded zero is what lets a used vehicle inherit a partially-elapsed exemption.

Household means-testing data (`taxable_income`, `children`) belongs in a `Household`
value object, not in the vehicle.

### `vehicle.py`

```python
Vehicle(
    name="Hyundai Inster",
    price=23_900,
    energy=Electricity(consumption=15.1, real_world_factor=1.18, home_share=0.8, ...),
    residual=FirstYearDropThenGeometric(...),
    insurance=750, maintenance=300, annual_tax=0,
    age_at_acquisition=0,          # > 0 makes it a used vehicle
    first_registration=date(2026, 1, 1),
)
```

Pure data about the asset. It knows nothing about how it is paid for and nothing about
who buys it.

### `acquisition.py`

```python
class Acquisition(ABC):
    def flows(self, vehicle, usage, timeline) -> CashFlowSeries: ...
```

Concrete: `Purchase(down_payment=…)`, `Financed(loan_rate, term, …)` — an annuity loan is
not the same as a cash purchase once the discount rate differs from the loan rate, and the
library should be able to show that — and `Lease(monthly_rate, term, initial_payment,
included_km, excess_km_rate, residual_risk_borne_by=lessor)`. `Lease` must support
**contract chaining** with an escalation factor when the horizon exceeds one term, and
must model excess-mileage settlement at each term end.

Guard against the trap documented in the workbook: advertised German lease rates usually
have the subsidy already baked in as a capitalised initial payment. If an `Incentive` and
a subsidised lease rate are both supplied, warn loudly about double counting.

### `alternative.py` and `chain.py`

An `Alternative` binds vehicle + acquisition + usage profile + incentives + the disposal
of the incumbent. A `ReplacementChain` composes alternatives sequentially — "run the
incumbent for its remaining 2 years, then buy the EV" — handling the handover period
correctly: disposal proceeds of the outgoing asset and acquisition of the incoming one
land in the same period, and the successor's residual is computed at *its* age at horizon,
not the horizon length.

Also expose the classical **optimal replacement age**: the holding period that minimises
EAC, found by evaluating an infinite (or n-fold) replacement chain over candidate ages.
That is the textbook-correct version of "how long should I keep it", and it falls out of
machinery you already have.

### `analysis.py` / `result.py`

`compare(alternatives, timeline, ...) -> ComparisonResult`, which must offer:

- `.npv()`, `.pv_of_outflows()`, `.eac()`, `.cost_per_km()` — per alternative.
- `.ranking()` and `.best()`, plus a **materiality flag**: if the winner's margin is
  within a configurable tolerance (default 3 % of PV) of the runner-up, say so. A
  spurious €200 lead over a six-year horizon is noise, and the library should not
  present it as a decision.
- `.incremental(a, b)` — the differential cash flow, its PV, and its **IRR where a sign
  change exists**. Guard against multiple/no IRR; return `None` rather than a wrong root.
  For cost-only streams also expose the differential payback period, clearly labelled as
  a secondary criterion.
- `.breakdown(alternative)` — labelled PV components summing to the total. Assert the sum.

### `sensitivity.py`

- `one_way(param, values)` and `two_way(p1, v1, p2, v2)` → grids matching the workbook's
  layout.
- `switch_point(param, between=(a, b))` — solve for the parameter value where the ranking
  flips, via `scipy.optimize.brentq`. **This is the library's headline feature.** Bracket
  automatically, return `None` with a reason when no crossing exists in the domain.
- `tornado(params)` — rank drivers by swing in PV over each parameter's plausible range.
  Vary one at a time from the base case; report absolute swing, not elasticity, since
  ranges differ in width.
- `monte_carlo(distributions, n, correlation=None)` — return the full distribution of the
  *difference* between alternatives, not just of each alternative. Report P(A beats B).
  Support correlated inputs (Cholesky on a supplied correlation matrix): energy prices and
  residual values are not independent, and pretending otherwise understates the spread.

### `scenario.py`

Named parameter overlays (`"fuel price collapse"`, `"subsidy expires"`, `"incumbent dies
early"`) with optional probability weights, yielding expected values across scenarios and
a per-scenario ranking table. Add **minimax-regret** alongside expected value: when
probabilities are soft — and here they are — the robust choice and the expected-value
choice can differ, and the user deserves to see both.

Stretch goal, only after the core is solid: value the **option to defer**. "Buy now vs.
buy in two years" is a real option, and a binomial lattice on the uncertain input
(subsidy availability, price trajectory) is a defensible way to price the flexibility that
a plain NPV comparison throws away. Ship it behind an explicit import; do not let it leak
into the default path.

### `io/excel.py` (optional extra)

Load a workbook laid out like the reference file into a `Scenario`. One-directional:
Excel → objects. Do not write Excel.

## API shape to aim for

```python
from carbitrage import Household, Timeline, Vehicle, compare
from carbitrage.energy import Electricity, LPG, Petrol, BivalentSource
from carbitrage.incentive import BAFA2026, ThgQuote, VehicleTaxExemption
from carbitrage.acquisition import Purchase, Lease
from carbitrage.residual import FirstYearDropThenGeometric

household = Household(taxable_income=55_000, children=0)
timeline  = Timeline(horizon_years=6, periods_per_year=12, rate=0.03,
                     energy_escalation=0.02)

ev = Alternative(
    Vehicle("Hyundai Inster", price=23_900,
            energy=Electricity(15.1, real_world_factor=1.18,
                               home_price=0.30, public_price=0.55, home_share=0.8),
            residual=FirstYearDropThenGeometric(drop=0.20, rate=0.13),
            insurance=750, maintenance=300),
    acquisition=Purchase(),
    incentives=[BAFA2026(), ThgQuote(300), VehicleTaxExemption()],
)

incumbent = Alternative(
    Vehicle("LPG incumbent", price=0,
            energy=BivalentSource(LPG(9.0, price=0.99), Petrol(7.5, price=2.10),
                                  primary_share=0.90),
            insurance=550, maintenance=900, annual_tax=160),
    acquisition=Purchase(down_payment=2_500),   # the repair bill
    life_years=2,
)

result = compare(
    [ev, ev.leased(Lease(199, term=36, included_km=10_000)),
     ReplacementChain(incumbent, then=ev)],
    timeline, usage=Usage(annual_km=12_000), household=household,
)

result.best()
result.switch_point("annual_km", between=("LPG incumbent chain", "Hyundai Inster"))
result.tornado(["annual_km", "lpg_price", "residual_rate", "discount_rate"])
result.to_markdown()
```

## Testing

- **Golden tests against the workbook.** Reproduce its base case and assert PV, EAC and
  cost/km per alternative to within €1. Extract the target numbers once into a fixture;
  do not read the workbook at test time.
- **Property tests** (`hypothesis`):
  - `rate == 0` ⟹ PV equals the undiscounted sum.
  - EAC × annuity factor round-trips to PV.
  - Two identical alternatives have identical EAC for any horizon; an n-fold chain of an
    alternative with itself has the same EAC as the single alternative.
  - Monotonicity: raising the discount rate must not raise the PV of a pure cost stream.
  - Scaling every monetary input by k scales every PV by k.
- **Regression on the failure modes**: unequal lives without a horizon must raise;
  subsidy double counting must warn; escalation on the wrong side of the real/nominal
  divide must be caught at construction.

## Output and reporting

`ComparisonResult.to_markdown()` and `.to_frame()` (pandas extra). A result must always be
printable to a table that a non-programmer can read: alternative, PV, EAC, cost/km,
delta vs. the status quo, rank. Round only at the presentation layer.

Every number the library reports must be traceable to a labelled component. If a user
cannot ask "where does this €20,844 come from" and get a breakdown that sums correctly,
the library has failed at its actual job, which is not arithmetic but justification.

## Deliberately out of scope

Utility/nuisance factors (range, charging time, comfort) are **not** monetised. The
library quantifies what is quantifiable and stays silent on the rest; conflating the two
is exactly the error that makes spreadsheet TCO comparisons untrustworthy. Say so in the
README.