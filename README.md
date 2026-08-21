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

from carbitrage import (
    LPG, BAFA2026, Alternative, BivalentSource, Electricity, GeometricDecline,
    Household, Incumbent, Lease, Petrol, Purchase, ReplacementChain,
    TabulatedResiduals, ThgQuote, Timeline, Usage, Vehicle, VehicleTaxExemption,
    compare,
)

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

## Documentation

| Page | What it covers |
| --- | --- |
| [Method](docs/method.md) | What is and is not monetised, terminology, conformal periodisation, real vs. nominal |
| [Non-negotiable rules](docs/rules.md) | The correctness invariants, and where each is enforced |
| [API tour](docs/api-tour.md) | What you can ask a `ComparisonResult` |
| [Traceability](docs/traceability.md) | Where a number comes from, and why the components must sum |
| [Validation](docs/validation.md) | What the test suite checks |
| [Working on the code](docs/development.md) | Setup, the layering rule, what the tests hold in place |

`carbitrage` is layered — `core` → `domain` → `engine` → `study` → `reporting` —
but everything public is re-exported at the top level, so `from carbitrage import X`
is the supported import in all cases. See [the layout table](docs/index.md#package-layout).

## Licence

GPL-3.0-or-later — see [LICENSE](LICENSE).
