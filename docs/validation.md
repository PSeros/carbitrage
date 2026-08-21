# Validation

## What the suite checks

```bash
make test
make lint
make typecheck
make example     # the full walkthrough, end to end
```

**Property tests** (`hypothesis`, `tests/integration/test_properties.py`) assert the
invariants that must hold for every input, not just the ones someone thought to
write down:

* `rate == 0` ⟹ the present value equals the undiscounted sum;
* EAC round-trips through the annuity factor;
* an n-fold self-chain has the EAC of a single holding;
* the PV of a cost stream is monotone in the discount rate;
* scaling every monetary input scales every present value;
* a residual value is never negative and never exceeds the price;
* a bivalent energy cost lies between its two legs.

**Failure-mode regressions** (`tests/integration/test_failure_modes.py`) pin every
error and warning the library is supposed to raise: unequal lives without a
chain, a nominal rate against unescalated flows, a real basis with no stated
inflation, double-counted subsidies, and the guards around IRR.

**The public API** is frozen in `tests/unit/test_public_api.py`. The 86 names
exported from `carbitrage` cannot change without that test failing.

**The layering** is enforced in `tests/unit/test_layering.py`: no module may
import from a package above its own.

## Provenance of the numbers

The engine was originally developed against `E-Auto_Kapitalwertmodell_2026_v3.xlsx`,
an independent implementation of the same problem (German EV vs. LPG-bivalent
incumbent vs. lease vs. used EV; 2026 subsidy matrix; 121-period monthly grid).
It reproduced all five of that workbook's alternatives to within 4 × 10⁻¹¹, and
its sensitivity layer cross-checked the workbook's break-even figures
independently.

That workbook and its golden-number regression have since been removed from the
repository. The figures quoted in the [quick start](../README.md#quick-start)
and in [traceability](traceability.md) are still executed on every test run by
`tests/docs/test_docs.py`, so the documented numbers cannot drift from the code —
but they are no longer checked against an outside implementation.
