"""The optional output adapters: Excel reporting and plots."""

from __future__ import annotations

from pathlib import Path

import pytest

from carbitrage import ComparisonResult
from carbitrage.analysis import Case
from carbitrage.scenario import Scenario, ScenarioSet
from carbitrage.sensitivity import Range, Uniform, advantage, monte_carlo, one_way, tornado
from tests.fixtures import workbook_base_case as wb


@pytest.fixture(scope="module")
def case() -> Case:
    return Case(
        alternatives=tuple(wb.all_alternatives()),
        timeline=wb.timeline(),
        usage=wb.usage(),
        household=wb.household(),
        incumbent=wb.incumbent(),
    )


@pytest.fixture(scope="module")
def result(case: Case) -> ComparisonResult:
    return case.run()


# ------------------------------------------------------------------- excel


def test_write_excel_produces_a_readable_workbook(
    result: ComparisonResult, tmp_path: Path
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    written = write_excel(result, tmp_path / "comparison", baseline=wb.A4)
    assert written.suffix == ".xlsx"
    assert written.exists()

    book = openpyxl.load_workbook(written)
    assert book.sheetnames == ["Result", "Breakdown", "Cashflow"]


def test_the_result_sheet_carries_the_ranking(result: ComparisonResult, tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    book = openpyxl.load_workbook(write_excel(result, tmp_path / "r.xlsx"))
    sheet = book["Result"]
    assert sheet.cell(row=7, column=2).value == wb.A1
    assert sheet.cell(row=7, column=4).value == pytest.approx(
        wb.TARGETS[wb.A1].pv_of_outflows, abs=1.0
    )


def test_the_cashflow_sheet_reproduces_the_npv(result: ComparisonResult, tmp_path: Path) -> None:
    """The grid is written as values so the present value can be checked by hand."""
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    book = openpyxl.load_workbook(write_excel(result, tmp_path / "cf.xlsx"))
    sheet = book["Cashflow"]
    timeline = wb.timeline()
    total = 0.0
    for t in range(timeline.n_periods + 1):
        factor = sheet.cell(row=5 + t, column=3).value
        amount = sheet.cell(row=5 + t, column=4).value
        total += float(factor) * float(amount)
    assert total == pytest.approx(wb.TARGETS[wb.A1].npv, abs=1.0)


def test_the_breakdown_sheet_totals_are_written(
    result: ComparisonResult, tmp_path: Path
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    book = openpyxl.load_workbook(write_excel(result, tmp_path / "b.xlsx"))
    totals = [
        row[3]
        for row in book["Breakdown"].iter_rows(min_row=5, values_only=True)
        if row[2] == "Total"
    ]
    assert len(totals) == len(result.names)
    assert any(value == pytest.approx(wb.TARGETS[wb.A1].npv, abs=1.0) for value in totals)


def test_scenarios_are_appended_when_supplied(
    case: Case, result: ComparisonResult, tmp_path: Path
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    analysis = ScenarioSet([Scenario("cheap gas", {"lpg_price": 0.60})]).run(case)
    book = openpyxl.load_workbook(
        write_excel(result, tmp_path / "s.xlsx", scenarios=analysis)
    )
    assert "Scenarios" in book.sheetnames
    assert book["Scenarios"].cell(row=3, column=2).value == "base case"


def test_the_cashflow_sheet_can_be_omitted(result: ComparisonResult, tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from carbitrage.io import write_excel

    book = openpyxl.load_workbook(
        write_excel(result, tmp_path / "short.xlsx", include_cashflows=False)
    )
    assert "Cashflow" not in book.sheetnames


def test_the_io_package_exposes_only_the_writer() -> None:
    import carbitrage.io as io_package

    assert callable(io_package.write_excel)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = io_package.read_excel  # type: ignore[attr-defined]


# --------------------------------------------------------------------- viz


def test_plots_render_without_a_display(case: Case, result: ComparisonResult) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from carbitrage.viz import monte_carlo_plot, one_way_plot, ranking_plot, tornado_plot

    grid = one_way(case, "annual_km", [8_000, 12_000, 20_000])
    bars = tornado(case, {"annual_km": Range(8_000, 20_000)}, metric=advantage(wb.A1, wb.A4))
    simulation = monte_carlo(
        case, {"lpg_price": Uniform(0.8, 1.3)}, between=(wb.A1, wb.A4), n=30, seed=1
    )

    assert one_way_plot(grid) is not None
    assert tornado_plot(bars) is not None
    assert monte_carlo_plot(simulation) is not None
    assert ranking_plot(result) is not None


def test_plots_accept_an_existing_axis(result: ComparisonResult) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from carbitrage.viz import ranking_plot

    _, ax = plt.subplots()
    assert ranking_plot(result, ax=ax) is ax
