from __future__ import annotations

from analysis.capabilities import analyze_dataset_capability
from analysis.planner import plan_analysis


def _supervised_capability(*, truncated: bool = False):
    columns = ("CustomerID", "TotalRevenue", "OrderCount", "Country", "SignupDate")
    rows = tuple(
        (
            f"C{index:03d}",
            100.0 + index,
            (index % 7) + 1,
            "US" if index % 2 else "IN",
            f"2026-01-{(index % 28) + 1:02d}",
        )
        for index in range(60)
    )
    return analyze_dataset_capability(columns, rows, truncated=truncated)


def test_plan_profile_is_ready_without_confirmation() -> None:
    plan = plan_analysis("profile this result", _supervised_capability())

    assert plan.status == "ready"
    assert plan.recipe == "profile"
    assert plan.target_column is None
    assert plan.row_scope == "artifact_rows"
    assert plan.confirmation_required is False


def test_plan_correlation_uses_numeric_columns_and_refetch_when_truncated() -> None:
    plan = plan_analysis("correlate numeric columns", _supervised_capability(truncated=True))

    assert plan.status == "ready"
    assert plan.recipe == "correlation"
    assert plan.feature_columns == ("TotalRevenue", "OrderCount")
    assert plan.row_scope == "bounded_refetch"
    assert plan.confirmation_required is True
    assert any("bounded re-fetch" in warning for warning in plan.warnings)


def test_plan_predict_resolves_target_and_excludes_unsafe_features() -> None:
    plan = plan_analysis("predict TotalRevenue", _supervised_capability())

    assert plan.status == "ready"
    assert plan.recipe == "predict"
    assert plan.target_column == "TotalRevenue"
    assert plan.feature_columns == ("OrderCount", "Country")
    assert plan.confirmation_required is True
    assert ("CustomerID", "ID-like column") in [
        (item.column, item.reason) for item in plan.excluded_columns
    ]
    assert ("TotalRevenue", "target column") in [
        (item.column, item.reason) for item in plan.excluded_columns
    ]


def test_plan_explain_asks_for_target_when_ambiguous() -> None:
    columns = ("Revenue", "Spend", "Orders", "Country")
    rows = tuple((100 + index, 50 + index, index % 5, "US") for index in range(60))
    capability = analyze_dataset_capability(columns, rows)

    plan = plan_analysis("explain drivers", capability)

    assert plan.status == "needs_clarification"
    assert plan.recipe == "explain"
    assert "Choose a target column" in plan.message
    assert plan.confirmation_required is False


def test_plan_predict_blocks_when_supervised_gates_fail() -> None:
    columns = ("TotalRevenue", "OrderCount", "Country")
    rows = tuple((100 + index, index % 5, "US" if index % 2 else "IN") for index in range(20))
    capability = analyze_dataset_capability(columns, rows)

    plan = plan_analysis("predict TotalRevenue", capability)

    assert plan.status == "blocked"
    assert plan.recipe == "predict"
    assert plan.target_column == "TotalRevenue"
    assert "at least 50" in plan.message
    assert plan.confirmation_required is False


def test_plan_unknown_request_needs_clarification() -> None:
    plan = plan_analysis("forecast this next quarter", _supervised_capability())

    assert plan.status == "needs_clarification"
    assert plan.recipe == "cannot_run"
    assert "profile, correlation, predict, or explain" in plan.message
