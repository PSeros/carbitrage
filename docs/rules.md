# Non-negotiable rules

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

