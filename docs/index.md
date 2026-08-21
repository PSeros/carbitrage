# carbitrage documentation

Start with the [quick start](../README.md#quick-start) in the README, then:

| Page | What it covers |
| --- | --- |
| [Method](method.md) | What is and is not monetised, the same-service premise, terminology, conformal periodisation, real vs. nominal |
| [Non-negotiable rules](rules.md) | The correctness invariants, and where each is enforced |
| [API tour](api-tour.md) | What you can ask a `ComparisonResult` |
| [Traceability](traceability.md) | Where a number comes from, and why the components must sum |
| [Validation](validation.md) | What the test suite checks, and the provenance of the figures |
| [Deviations from the brief](deviations.md) | Two deliberate departures from the original specification |

## Package layout

`carbitrage` is layered; a module may import only from its own package or one below.

| Package | Holds |
| --- | --- |
| `carbitrage.errors` | The error and warning hierarchy |
| `carbitrage.core` | `timeline`, `cashflow` — discounting machinery that knows nothing about cars |
| `carbitrage.domain` | `vehicle`, `energy`, `residual`, `tax`, `incentive`, `acquisition`, `context` |
| `carbitrage.engine` | `alternative`, `comparison`, `chain`, `result` — building a comparison and running it |
| `carbitrage.study` | `params`, `sensitivity`, `scenario` — at what value does the answer flip |
| `carbitrage.reporting` | `viz`, `excel` — optional output adapters |

Everything in the table above is re-exported from the top level, so
`from carbitrage import Timeline` is the supported import in all cases.
