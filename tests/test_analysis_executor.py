from __future__ import annotations

from analysis.capabilities import analyze_dataset_capability
from analysis.executor import execute_analysis_plan
from analysis.planner import plan_analysis


def test_execute_profile_returns_column_profile_table() -> None:
    columns = ("CustomerID", "Revenue", "Country")
    rows = (
        ("C001", 10.0, "US"),
        ("C002", 20.0, "IN"),
        ("C003", None, "US"),
    )
    capability = analyze_dataset_capability(columns, rows)
    plan = plan_analysis("profile this result", capability)

    result = execute_analysis_plan(plan, capability, columns, rows)

    assert result.status == "success"
    assert result.recipe == "profile"
    assert "Profiled 3 rows and 3 columns" in result.summary
    assert result.tables[0].columns == ("Column", "Type", "Non-null", "Nulls", "Distinct", "Notes")
    assert any(row[0] == "CustomerID" and row[5] == "ID-like" for row in result.tables[0].rows)
    assert any(row[0] == "Revenue" and "target candidate" in row[5] for row in result.tables[0].rows)


def test_execute_correlation_returns_pairwise_pearson_table() -> None:
    columns = ("Revenue", "OrderCount", "Inverse")
    rows = (
        (1, 2, 3),
        (2, 4, 2),
        (3, 6, 1),
    )
    capability = analyze_dataset_capability(columns, rows)
    plan = plan_analysis("correlate numeric columns", capability)

    result = execute_analysis_plan(plan, capability, columns, rows)

    assert result.status == "success"
    assert result.recipe == "correlation"
    assert result.tables[0].columns == ("Column A", "Column B", "Correlation", "Paired rows")
    rows_by_pair = {(row[0], row[1]): row for row in result.tables[0].rows}
    assert rows_by_pair[("Revenue", "OrderCount")][2:] == ("1.000", 3)
    assert rows_by_pair[("Revenue", "Inverse")][2:] == ("-1.000", 3)


def test_execute_correlation_blocks_truncated_artifact() -> None:
    columns = ("Revenue", "OrderCount")
    rows = tuple((index, index * 2) for index in range(10))
    capability = analyze_dataset_capability(columns, rows, truncated=True)
    plan = plan_analysis("correlate numeric columns", capability)

    result = execute_analysis_plan(plan, capability, columns, rows)

    assert result.status == "blocked"
    assert "truncated artifacts" in result.summary
    assert result.tables == ()


def test_execute_correlation_allows_materialized_bounded_rows() -> None:
    columns = ("Revenue", "OrderCount")
    rows = tuple((index, index * 2) for index in range(10))
    capability = analyze_dataset_capability(columns, rows, truncated=True)
    plan = plan_analysis("correlate numeric columns", capability)

    result = execute_analysis_plan(
        plan,
        capability,
        columns,
        rows,
        allow_bounded_rows=True,
    )

    assert result.status == "success"
    assert result.tables[0].rows == (("Revenue", "OrderCount", "1.000", 10),)
