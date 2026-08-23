# Working on the code

## Getting set up

```bash
make install          # uv sync --all-extras --group dev
uv run pre-commit install
make check            # lint, types, tests, and the end-to-end example
```

`make check` is what CI runs, so a clean run locally means a clean run there.

## Scopes

The unit of structure is a **scope**: one concept, one directory, its
`__init__.py` nothing but re-exports and `__all__`.

```
carbitrage/energy/
    __init__.py       the scope's public surface
    base.py           EnergySource — what every carrier has to answer
    electricity.py    Electricity
    fuels.py          Petrol, Diesel, LPG, Hydrogen
    bivalent.py       BivalentSource
```

A scope stays a single module while it is small enough not to need a directory
(`vehicle.py`, `residual.py`, `tax.py`, `context.py`, `errors.py`), and becomes
one when it grows past roughly 200 lines or holds units that are independently
substantial. Converting one is invisible to users: `carbitrage.energy` is the
import path either way.

Six spine names — `Timeline`, `Vehicle`, `Alternative`, `Case`, `compare`,
`ComparisonResult` — sit directly on `carbitrage` as well, because they open
every script and stutter when qualified.

`carbitrage/__init__.py` resolves scopes lazily through a module `__getattr__`,
so `import carbitrage` costs the spine and nothing more; reaching for
`carbitrage.sensitivity` is what pulls in `scipy`. The `if TYPE_CHECKING:` block
listing every scope is what keeps them resolvable for mypy and editors — keep it
in sync with `_SCOPES`.

## The layering rule

Scopes are layered. **A module may import from its own scope or one below it,
never above.**

```
errors → rates, cashflow → the modelled world → comparison → params,
sensitivity, scenario → reporting
```

| Layer | Scopes | May import |
| ---: | --- | --- |
| 0 | `errors` | nothing |
| 1 | `rates`, `cashflow` | `errors` |
| 2 | `vehicle`, `energy`, `residual`, `tax`, `incentives`, `acquisition`, `context` | layers 0–1 |
| 3 | `comparison` | layers 0–2 |
| 4 | `params`, `sensitivity`, `scenario` | everything below |
| 5 | `reporting` | everything below |

The layer of each scope is **declared** in `tests/unit/test_layering.py`, not
inferred from the directory tree, and enforced on every run. That separation is
the point: the tree answers *which concept a name belongs to*, the table answers
*what may depend on what*, and neither question should distort the other. A scope
added without a layer fails the scan rather than escaping it.

The scan reads *module-level* imports only, which is deliberate: the six
convenience delegators on `ComparisonResult` (`result.one_way(...)`,
`result.tornado(...)`, …) reach up into `sensitivity` from inside the method
body, so `comparison` stays importable without `sensitivity`. Another upward
reference has to be function-local and carry a comment saying why — the test does
not get relaxed.

Type-only imports under `if TYPE_CHECKING:` are exempt and may point anywhere.

## Adding to the public API

Adding a name means editing `__all__` in its own module, in the scope's
`__init__.py`, **and** in the frozen `SCOPES` table in
`tests/unit/test_public_api.py`. That friction is intentional. Leaving out the
last two is not an option: the orphan scan in that file fails on any public name
that no scope exports.

## Documentation

Every ```` ```python ```` block in `README.md` and `docs/*.md` is compiled by
`tests/docs/test_docs.py`, and the README quick start is executed with its
printed figures checked against the code. A number quoted in the docs is a
number that gets verified. Illustrative fragments that cannot run standalone
still have to parse, and any `result.<attribute>` they mention must exist on
`ComparisonResult`.
