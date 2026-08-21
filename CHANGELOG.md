# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

* **The package is now layered into subpackages.** The declared public API is
  unchanged — all 87 names in `carbitrage.__all__` import exactly as before, and
  `tests/unit/test_public_api.py` enforces that. What moved are the *submodule*
  paths, which were never re-exported or documented as public:

  | Was | Is now |
  | --- | --- |
  | `carbitrage.timeline`, `carbitrage.cashflow` | `carbitrage.core.*` |
  | `carbitrage.vehicle`, `.energy`, `.residual`, `.tax`, `.incentive`, `.acquisition`, `.context` | `carbitrage.domain.*` |
  | `carbitrage.alternative`, `.chain`, `.result` | `carbitrage.engine.*` |
  | `carbitrage.analysis` | `carbitrage.engine.comparison` |
  | `carbitrage.params`, `.sensitivity`, `.scenario` | `carbitrage.study.*` |
  | `carbitrage.viz` | `carbitrage.reporting.viz` |
  | `carbitrage.io` | `carbitrage.reporting` |

  `carbitrage.errors` is the only submodule path that kept its name.
  `from carbitrage.io import write_excel` becomes
  `from carbitrage.reporting import write_excel`.

* The `README.md` was split: it keeps the pitch, install and quick start, and
  everything else moved into `docs/`.

### Removed

* The reference workbook `E-Auto_Kapitalwertmodell_2026_v3.xlsx`, the
  `AGENT_BRIEF.md` specification, and `tests/fixtures/workbook_base_case.py`.
* The 172 tests that depended on that fixture, including the entire
  golden-workbook regression (`tests/test_golden_workbook.py`). The suite went
  from 349 tests to 224. See [docs/validation.md](docs/validation.md) for what
  is still checked.

### Added

* `tests/unit/test_layering.py` — enforces that no module imports from a
  package above its own.
* `tests/unit/test_public_api.py` — freezes the 87 exported names.
* `tests/docs/test_docs.py` — executes the documented code blocks, checks the
  figures the docs quote against what the code produces, and verifies that
  every relative link resolves.
* CI (`.github/workflows/ci.yml`), `.pre-commit-config.yaml`, a `Makefile`,
  `LICENSE` (GPL-3.0-or-later), `CONTRIBUTING.md` and this changelog.

## [0.1.0]

Initial implementation: capital budgeting for mutually exclusive asset
alternatives with unequal lives.
