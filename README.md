# carbitrage

Capital budgeting for **mutually exclusive asset alternatives with unequal
lives**. Vehicles are the first-class use case; nothing in the core knows about
cars.

The question this library answers is never *"what does this cost"*. It is:

> Given several mutually exclusive options with different lifetimes, different
> acquisition modes, different energy carriers and different subsidy regimes —
> which one minimises the present value of outflows, and **at what parameter
> value does that answer flip?**

The second half is the point. A comparison that says *"the electric car wins by
€1,930"* is far less useful than one that says *"the lease wins below 10,358 km
a year and the purchase wins above it"*, because the second tells you which side
of the question you are on.

---

## Install

```bash
uv add carbitrage                  # core: numpy, scipy
uv add "carbitrage[frames,viz,excel]"   # pandas frames, matplotlib plots, xlsx export
```

Python 3.11+. Core dependencies are numpy and scipy (the latter for root
finding only).

## Quick start

```python
from datetime import date

from carbitrage import (
    Alternative, Household, Incumbent, Lease, Purchase, ReplacementChain,
    TabulatedResiduals, Timeline, Usage, Vehicle, compare,
)
from carbitrage.energy import LPG, BivalentSource, Electricity, Petrol
from carbitrage.incentive import BAFA2026, ThgQuote, VehicleTaxExemption
from carbitrage.residual import GeometricDecline

timeline  = Timeline(horizon_years=6, periods_per_year=12, rate=0.03,
                     energy_escalation=0.02, vehicle_price_escalation=0.015)
household = Household(taxable_income=55_000, children=0)
usage     = Usage(annual_km=12_000)

ev = Vehicle(
    "Hyundai Inster", price=23_900,
    energy=Electricity(15.1, real_world_factor=1.18,
                       home_price=0.30, public_price=0.55, home_share=0.8),
    residual=GeometricDecline(0.15),
    insurance=750, maintenance=300, setup_cost=1_500,   # the wallbox
    first_registration=date(2026, 1, 1),
)
incentives = (BAFA2026(), ThgQuote(300), VehicleTaxExemption())

buy   = Alternative(ev, Purchase(), incentives, label="Buy the EV now")
lease = Alternative(ev, Lease(239, term_months=36, included_km=10_000,
                              excess_km_rate=0.12, renewal_escalation=0.05),
                    incentives, label="Lease the EV")

old = Vehicle(
    "LPG incumbent", price=1_500,          # its current market value
    energy=BivalentSource(LPG(7.5, price=0.99, volumetric_penalty=1.2),
                          Petrol(7.5, price=2.10), primary_share=0.90),
    residual=TabulatedResiduals.from_values(1_500, {2: 800}),
    insurance=550, maintenance=900, annual_tax=160,
)
defer = ReplacementChain(
    Alternative(old, Purchase(upfront_extra=2_500, already_owned=True),
                life_years=2, disposes_incumbent=False, label="Keep the incumbent"),
    Alternative(ev, Purchase(), incentives, label="Buy the EV in 2 years"),
    label="Repair now, replace in 2 years",
)

result = compare([buy, lease, defer], timeline, usage=usage, household=household,
                 incumbent=Incumbent(old, market_value=1_500))

print(result.to_markdown())
print(result.switch_point("annual_km", ("Buy the EV now", "Lease the EV")).describe())
```

```
| # | Alternative                     | PV of outflows | EAC / year | Cost / km | vs. …     |
|--:|---------------------------------|---------------:|-----------:|----------:|----------:|
| 1 | Buy the EV now                  |      20,844.03 |   3,847.76 |    0.3206 |  1,930.36 |
| 2 | Lease the EV                    |      21,880.17 |   4,039.02 |    0.3366 |    894.22 |
| 3 | Repair now, replace in 2 years  |      22,774.39 |   4,204.10 |    0.3503 |      0.00 |

Buy the EV now is ahead of Lease the EV by 1,036 in present value (5.0 % of its
own PV): a clear difference.

annual_km switches the answer at 10,357.80 (base case 12,000.00): below it
Lease the EV wins, above it Buy the EV now wins.
```

## Two sides of one assumption

**What is not monetised.** Range, charging time, comfort, noise, the nuisance of
a dead car on a Monday morning — none of it. The library quantifies what is
quantifiable and stays silent on the rest. Conflating the two is exactly the
error that makes spreadsheet TCO comparisons untrustworthy, so where a
difference cannot be defended in euros it does not appear.

**What licenses deciding on cost alone.** That silence is only acceptable
because of an assumption doing a great deal of work: every alternative is taken
to deliver **the same service over the same horizon**. That assumption is
fragile. A larger vehicle, or one with materially different range, is *not* the
same service — and in that case the comparison is **invalid rather than merely
incomplete**. No amount of methodological care downstream repairs a comparison
between things that are not comparable.

These are the same point seen from opposite sides. Check the premise before you
read the number.

Note also that an absence of *operating revenue* is not an absence of inflows.
Grants, recurring credits, disposal proceeds of the incumbent at t=0 and
terminal residual value are all genuine inflows; they are netted inside the
stream and never dropped. What is absent is revenue *attributable to the asset*,
because the vehicle earns nothing.

## Non-negotiable rules

These are correctness requirements, enforced in the types or at runtime.

| Rule | How it is enforced |
|---|---|
| Unequal lives are never compared by raw NPV | `compare()` raises `UnequalLivesError` unless every alternative covers the horizon, or you pass `allow_unequal_lives=True` and read only EAC |
| Rank by NPV, never by `abs()` | Outflows are negative throughout the cash-flow layer; the sign flips only in `pv_of_outflows`, which is a presentation convenience and *not* an absolute value |
| Conformal periodisation | `(1+i_a)**(1/m) - 1`. The proportional convention needs an explicit `Periodisation.PROPORTIONAL` |
| Real and nominal are never mixed | Checked in `Timeline.__post_init__`; `fisher_nominal` / `fisher_real` convert exactly, not by addition |
| Sunk costs out, opportunity costs in | The incumbent enters at its **current market value**, as an inflow to every alternative that disposes of it and a forgone inflow to the one that keeps it. Past spend has no representation at all |
| Terminal value is derived, not entered | `Terminal.from_residual` takes a `ResidualValueModel` and an **age**, so a chain successor four years old at the horizon is valued on the same curve as anything else four years old |
| EAC = PV × CRF | `CRF = i(1+i)^n / ((1+i)^n − 1)` |
| Subsidies are not counted twice | A lease marked `rate_includes_subsidy=True` plus an explicit `Incentive` emits a `DoubleCountingWarning` |
| Taxes are pluggable | Default `PrivateHousehold` (no depreciation, no input VAT, subsidies tax-free); `BusinessAssets` adds input VAT and a depreciation shield without the engine changing |

### Terminology

In German this is a **Kapitalwertvergleich** or **Barwertminimierung**.
*Kostenvergleichsrechnung* denotes the **static, undiscounted** method in German
cost accounting and is the wrong label for what this library does.

## What it can tell you

```python
result.best()                       # the winning Evaluation
result.ranking()                    # every alternative, best first
result.is_material()                # is the lead bigger than 3 % of PV, or noise?
result.breakdown("Buy the EV now")  # labelled components that sum to the total
result.incremental(a, b)            # differential stream, its PV, IRR and payback
result.to_markdown(); result.to_frame()
```

**Sensitivity** — every one of these re-runs the full monthly cash-flow engine:

```python
result.switch_point("lpg_price", (a, b))       # solved with brentq, None if no crossing
result.switch_point_report("lpg_price", (a, b)) # ... and why, when there is none
result.one_way("annual_km", [5_000, 12_000, 30_000])
result.two_way("annual_km", rows, "lpg_price", cols)
result.tornado({"annual_km": Range(5_000, 30_000),
                "residual_rate": Range(0.7, 1.4, relative=True)})
result.monte_carlo({"lpg_price": Triangular(0.80, 0.99, 1.40)},
                   between=(a, b), correlation=[[1, .8], [.8, 1]], n=5_000)
```

`monte_carlo` returns the distribution of the **difference**, not of each
alternative separately, and reports `P(a beats b)`. Two alternatives that share
an energy price move together; the spread of each one alone says nothing about
how often one wins.

**Scenarios** — expected value *and* minimax regret, because when probabilities
are soft the two rules can disagree, and that disagreement is the finding:

```python
ScenarioSet([
    Scenario("subsidy expires", {"...incentives[0].available": 0}, probability=0.3),
    Scenario("incumbent dies early", {"...upfront_extra": 6_000}, probability=0.2),
]).run(case).to_markdown()
```

**Optimal replacement age** — the textbook answer to "how long should I keep
it", by minimising EAC over candidate holding periods:

```python
optimal_replacement_age(alt, timeline=tl, candidates=[3, 4, 6, 8, 10], usage=usage)
```

The classical U-shaped curve only appears when running costs actually grow with
age. With flat costs the annuity falls monotonically and the "optimum" is just
the longest candidate — a statement about your inputs, not about replacement
policy.

## Method note

The alternatives here are **not identical repetitions**: the same car bought
today and bought in two years differ in price vintage, in subsidy availability,
and in *age at the horizon* — six years versus four. So:

- EAC-as-infinite-chain is **not** licensed as the primary method. The
  defensible one is the explicit finite common horizon with age-consistent
  terminal values, which is what `ReplacementChain` builds.
- Consequently **the period at which a leg starts is an economic quantity**, not
  plumbing: it fixes the asset's age and hence its residual, the vintage of its
  price, and whether and when a subsidy disburses.

The "keep it or replace it" question is the classical defender–challenger
problem (Terborgh 1949; Grant/Ireson/Leavenworth), and optimal replacement age
traces to Hotelling (1925) and Preinreich (1940). Deferring under uncertainty is
a real option (McDonald & Siegel 1986; Dixit & Pindyck 1994) — deterministic NPV
systematically *understates* the value of waiting, and this library does not
price that flexibility.

None of these are "proven" the way a theorem is. They are normative decision
rules resting on stated assumptions, chiefly a capital market where you can lend
and borrow at the discount rate. The genuinely uncertain part of any result here
is not the method but the **residual value curve**, and no methodological rigour
repairs a wrong depreciation assumption.

### Real or nominal residuals

A depreciation rate read off observed used-asset prices is a **nominal** rate of
price decline — it already contains inflation. Applied to today's nominal price
it yields a future nominal price, correctly discounted at a nominal rate. On a
real basis, restate the rate as `1 - (1 - nominal) / (1 + inflation)`. The
library cannot tell which one you handed it, so this is documented rather than
guessed at.

## Traceability

Every number reports where it comes from, and the components sum:

```python
print(result.breakdown_markdown("Buy the EV now"))
```

```
| Purchase price of Hyundai Inster | -23,900.00 |
| Setup cost for Hyundai Inster    |  -1,500.00 |
| Residual value of Hyundai Inster |   7,548.98 |
| Energy for Hyundai Inster        |  -4,313.10 |
| Insurance for Hyundai Inster     |  -4,118.46 |
| Maintenance and wear             |  -1,647.38 |
| Disposal of LPG incumbent        |   1,500.00 |
| Purchase premium, paid month 4   |   3,960.78 |
| Greenhouse-gas quota credit      |   1,625.16 |
| **Total**                        | **-20,844.03** |
```

If you cannot ask "where does this €20,844 come from" and get an answer that
adds up, the library has failed at its actual job — which is not arithmetic but
justification. `breakdown()` asserts the sum on every call.

## Validation

The repository ships `E-Auto_Kapitalwertmodell_2026_v3.xlsx`, an independent
implementation of the same problem (German EV vs. LPG-bivalent incumbent vs.
lease vs. used EV; 2026 subsidy matrix; 121-period monthly grid). The library
reproduces all five of its alternatives:

| Alternative | PV of outflows | Workbook | Δ |
|---|---:|---:|---:|
| EV bought new | 20,844.0262 | 20,844.0262 | 4 × 10⁻¹¹ |
| EV leased | 21,880.1650 | 21,880.1650 | 3 × 10⁻¹¹ |
| Petrol car new | 28,638.1236 | 28,638.1236 | 1 × 10⁻¹¹ |
| Incumbent then EV | 22,774.3895 | 22,774.3895 | 4 × 10⁻¹² |
| Used EV | 21,964.6452 | 21,964.6452 | 4 × 10⁻¹¹ |

EAC, cost per kilometre, undiscounted control totals and the component breakdown
all match too. The targets are extracted into a fixture, so the test suite never
opens the workbook.

The sensitivity layer cross-checks independently: the workbook's break-even
repair cost is €594.94 from an annual closed form; solving on the monthly engine
gives €569.60, and the workbook's own note says its approximation differs from
its monthly grid. Its break-even mileage is unreachable; `switch_point`
independently reports no crossing.

Beyond that: property tests (`hypothesis`) for `rate == 0` ⟹ PV equals the
undiscounted sum, EAC round-tripping through the annuity factor, an n-fold
self-chain having the EAC of a single holding, monotonicity in the discount
rate, and scale homogeneity; plus regression tests on every failure mode above.

```bash
uv run pytest                # 349 tests
uv run ruff check src tests
uv run mypy --strict src
uv run python examples/workbook_case.py   # the full walkthrough, end to end
```

## Deviations from the original brief

Two, both deliberate:

1. **Builders take a `Context`**, not the brief's literal
   `flows(vehicle, usage, timeline)`. Those signatures cannot express a
   replacement chain — see the method note above for why the leg's start period
   is an economic quantity rather than an implementation detail.
2. **Excel is an output, not an input.** The brief specified an
   Excel → objects loader. Cases are defined in Python here; `carbitrage.io`
   writes a *result* workbook (ranking, breakdown, full cash-flow grid,
   scenarios) as values rather than formulas, so there is never a second
   implementation of the engine to drift out of step.

The real-option lattice for the value of deferring is out of scope for now.

## Licence

Not yet chosen.
