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

The package is not on an index; install it from the repository.

```bash
# core: numpy, scipy
uv add "carbitrage @ git+https://github.com/PSeros/carbitrage.git"

# with pandas frames, matplotlib plots, xlsx export
uv add "carbitrage[frames,viz,excel] @ git+https://github.com/PSeros/carbitrage.git"
```

`pip install "carbitrage @ git+https://github.com/PSeros/carbitrage.git"` works
the same way. Pin a commit or tag by appending `@<ref>` to the URL.

To work on the library itself, clone it and sync the environment:

```bash
git clone https://github.com/PSeros/carbitrage.git
cd carbitrage
make install          # uv sync --all-extras --group dev
```

Python 3.11+. Core dependencies are numpy and scipy (the latter for root
finding only).

## Quick start

```python
from datetime import date

import carbitrage as cb
from carbitrage import acquisition, comparison, context, energy, incentives, residual

timeline  = cb.Timeline(horizon_years=6, periods_per_year=12, rate=0.03,
                        energy_escalation=0.02, vehicle_price_escalation=0.015)
household = context.Household(taxable_income=55_000, children=0)
usage     = context.Usage(annual_km=12_000)

ev = cb.Vehicle(
    "Hyundai Inster", price=23_900,
    energy=energy.Electricity(15.1, real_world_factor=1.18,
                              home_price=0.30, public_price=0.55, home_share=0.8),
    residual=residual.GeometricDecline(0.15),
    insurance=750, maintenance=300, setup_cost=1_500,   # the wallbox
    first_registration=date(2026, 1, 1),
)
grants = (incentives.BAFA2026(), incentives.ThgQuote(300), incentives.VehicleTaxExemption())

buy   = cb.Alternative(ev, acquisition.Purchase(), grants, label="Buy the EV now")
lease = cb.Alternative(ev, acquisition.Lease(239, term_months=36, included_km=10_000,
                                             excess_km_rate=0.12, renewal_escalation=0.05),
                       grants, label="Lease the EV")

old = cb.Vehicle(
    "LPG incumbent", price=1_500,          # its current market value
    energy=energy.BivalentSource(energy.LPG(7.5, price=0.99, volumetric_penalty=1.2),
                                 energy.Petrol(7.5, price=2.10), primary_share=0.90),
    residual=residual.TabulatedResiduals.from_values(1_500, {2: 800}),
    insurance=550, maintenance=900, annual_tax=160,
)
defer = comparison.ReplacementChain(
    cb.Alternative(old, acquisition.Purchase(upfront_extra=2_500, already_owned=True),
                   life_years=2, disposes_incumbent=False, label="Keep the incumbent"),
    cb.Alternative(ev, acquisition.Purchase(), grants, label="Buy the EV in 2 years"),
    label="Repair now, replace in 2 years",
)

result = cb.compare([buy, lease, defer], timeline, usage=usage, household=household,
                    incumbent=context.Incumbent(old, market_value=1_500))

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

## Naming the parameter you want to vary

Every sensitivity question needs to say which number it varies. Mark that number
where you write it, and it carries its own name:

```python
from carbitrage import params

repair = params.Uncertain(2_500, "repair_bill")     # the repair bill, and its name

cb.Alternative(old, acquisition.Purchase(upfront_extra=repair, already_owned=True),
               life_years=2, disposes_incumbent=False, label="Keep the incumbent")

result.one_way(repair, [800, 2_500, 6_000])         # sweep the mark itself
result.switch_point("repair_bill", (a, b))          # or the name it carries
```

A third argument declares what you know beyond the base case — a distribution
where the shape is known, a `Range` where only the span is — so that a tornado
and a Monte Carlo over one parameter cannot rest on two different beliefs:

```python
repair = params.Uncertain(2_500, "repair_bill", Triangular(1_800, 2_500, 6_000))

print(result.switch_point("repair_bill", (a, b)).describe())
# repair_bill switches the answer at 2,117.02 (base case 2,500.00): below it
# Repair wins, above it Buy wins.  That is inside the 1,800.00 to 6,000.00 you
# called plausible.
```

A declaration **informs an answer and never restricts a question**. The switch
point above is still solved across the whole domain, so a crossing far outside
what you called plausible is still found and reported as such — *can* this flip
and *would* it flip are different findings, and the library will not collapse
them into one.

An `Uncertain` *is* a float — the base case evaluates on it untouched — so the
mark costs nothing until a study asks for it by name. Parameters nobody marked
are named by one of the short aliases (`"annual_km"`, `"lpg_price"`,
`"discount_rate"`, …) or by a dotted path
(`"alternatives[…].legs[…].acquisition.upfront_extra"`); `params.find(case,
"life")` locates one, and `params.uncertainties(case)` lists the marks.

## Documentation

| Page | What it covers |
| --- | --- |
| [Method](docs/method.md) | What is and is not monetised, terminology, conformal periodisation, real vs. nominal |
| [Non-negotiable rules](docs/rules.md) | The correctness invariants, and where each is enforced |
| [API tour](docs/api-tour.md) | What you can ask a `ComparisonResult` |
| [Traceability](docs/traceability.md) | Where a number comes from, and why the components must sum |
| [Validation](docs/validation.md) | What the test suite checks |
| [Working on the code](docs/development.md) | Setup, the layering rule, what the tests hold in place |

The public API is organised into scopes — `energy`, `acquisition`, `incentives`,
`tax`, `residual`, `sensitivity` and nine more — with six spine names
(`Timeline`, `Vehicle`, `Alternative`, `Case`, `compare`, `ComparisonResult`)
directly on `carbitrage`. See [the scope table](docs/index.md#the-public-scopes).

## Licence

GPL-3.0-or-later — see [LICENSE](LICENSE).
