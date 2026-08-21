# Contributing

## Getting set up

```bash
make install          # uv sync --all-extras --group dev
uv run pre-commit install
make check            # lint, types, tests, and the end-to-end example
```

## The layering rule

`carbitrage` is layered. **A module may import from its own package or one
below it, never above.**

```
errors  →  core  →  domain  →  engine  →  study  →  reporting
```

| Package | Holds | May import |
| --- | --- | --- |
| `errors` | the error and warning hierarchy | nothing |
| `core` | `timeline`, `cashflow` | `errors` |
| `domain` | `vehicle`, `energy`, `residual`, `tax`, `incentive`, `acquisition`, `context` | `errors`, `core` |
| `engine` | `alternative`, `comparison`, `chain`, `result` | `errors`, `core`, `domain` |
| `study` | `params`, `sensitivity`, `scenario` | everything below |
| `reporting` | `viz`, `excel` | everything below |

`tests/unit/test_layering.py` enforces this on every run. It scans *module-level*
imports only, which is deliberate: the six convenience delegators on
`ComparisonResult` (`result.one_way(...)`, `result.tornado(...)`, …) reach up into
`study.sensitivity` from inside the method body, so `engine` stays importable
without `study`. If you need another upward reference, make it function-local
and say why in a comment — do not relax the test.

Type-only imports under `if TYPE_CHECKING:` are exempt and may point anywhere.

## Adding to the public API

`carbitrage/__init__.py` is the facade and the only supported import path.
Subpackage `__init__.py` files are docstrings only, on purpose: eager
re-exports would make `import carbitrage.study.params` drag in `scipy`, and
under `mypy --strict` (`no_implicit_reexport`) each one would need its own
`__all__` to keep in sync by hand.

Adding a name means editing `__all__` in `carbitrage/__init__.py` **and** the
frozen list in `tests/unit/test_public_api.py`, plus a `CHANGELOG.md` entry.
That friction is intentional.

## Documentation

Every ```` ```python ```` block in `README.md` and `docs/*.md` is compiled by
`tests/docs/test_docs.py`, and the README quick start is executed with its
printed figures checked against the code. If you quote a number in the docs,
it will be verified. Illustrative fragments that cannot run standalone still
have to parse, and any `result.<attribute>` they mention must exist on
`ComparisonResult`.

## Before you push

```bash
make check
```
