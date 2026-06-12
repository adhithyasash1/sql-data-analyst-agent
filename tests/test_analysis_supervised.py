from __future__ import annotations

from analysis.capabilities import analyze_dataset_capability
from analysis.planner import plan_analysis
from analysis.supervised import build_supervised_preflight


def _preflight(columns, rows, request: str, *, truncated: bool = False, **kwargs):
    capability = analyze_dataset_capability(columns, rows, truncated=truncated)
    plan = plan_analysis(request, capability)
    return build_supervised_preflight(plan, capability, columns, rows, **kwargs)


def test_supervised_preflight_passes_regression_with_fixed_baseline_plan() -> None:
    columns = ("TotalRevenue", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            100.0 + index * 3 + (index % 7),
            index % 7,
            (index % 5) / 10,
            "US" if index % 3 == 0 else "IN",
        )
        for index in range(60)
    )

    preflight = _preflight(columns, rows, "predict TotalRevenue")

    assert preflight.status == "ready"
    assert preflight.can_execute is True
    assert preflight.problem_type == "regression"
    assert preflight.eligible_features == ("OrderCount", "DiscountRate", "Country")
    assert "Median baseline" in preflight.baseline_recipe
    assert all(gate.status == "pass" for gate in preflight.gates)


def test_supervised_preflight_passes_classification_with_majority_baseline() -> None:
    columns = ("Status", "OrderCount", "SpendScore", "Country")
    rows = tuple(
        (
            "churned" if index % 3 == 0 else "active",
            index % 7,
            50.0 + (index % 11),
            ("US", "IN", "GB")[index % 3],
        )
        for index in range(60)
    )

    preflight = _preflight(columns, rows, "explain Status")

    assert preflight.status == "ready"
    assert preflight.problem_type == "classification"
    assert "Majority-class baseline" in preflight.baseline_recipe
    assert "permutation importance" in preflight.baseline_recipe
    assert any(gate.name == "class_balance" and gate.status == "pass" for gate in preflight.gates)


def test_supervised_preflight_blocks_small_row_count() -> None:
    columns = ("TotalRevenue", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (100.0 + index * 3 + (index % 7), index % 7, (index % 5) / 10, "US")
        for index in range(40)
    )

    preflight = _preflight(columns, rows, "predict TotalRevenue")

    assert preflight.status == "blocked"
    assert preflight.can_execute is False
    assert any(gate.name == "usable_rows" and gate.status == "block" for gate in preflight.gates)


def test_supervised_preflight_blocks_unsupported_datetime_target() -> None:
    columns = ("CreatedDate", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            f"2026-01-{(index % 28) + 1:02d}",
            index % 7,
            (index % 5) / 10,
            "US" if index % 2 else "IN",
        )
        for index in range(60)
    )

    preflight = _preflight(columns, rows, "predict CreatedDate")

    assert preflight.status == "blocked"
    assert preflight.problem_type == "unsupported"
    assert any(gate.name == "target" and gate.status == "block" for gate in preflight.gates)


def test_supervised_preflight_blocks_evidence_backed_leakage() -> None:
    columns = ("Revenue", "RevenueCopy", "OrderCount", "DiscountRate")
    rows = tuple(
        (
            100.0 + index * 3 + (index % 7),
            100.0 + index * 3 + (index % 7),
            index % 7,
            (index % 5) / 10,
        )
        for index in range(60)
    )

    preflight = _preflight(columns, rows, "predict Revenue")
    assessments = {item.column: item for item in preflight.feature_assessments}

    assert preflight.status == "blocked"
    assert assessments["RevenueCopy"].status == "blocked"
    assert "target values" in assessments["RevenueCopy"].reason
    assert any(gate.name == "target_leakage" and gate.status == "block" for gate in preflight.gates)


def test_supervised_preflight_warns_on_name_similarity_without_blocking() -> None:
    columns = ("Revenue", "RevenueBucket", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            100.0 + index * 3 + (index % 7),
            "high" if index % 4 in {0, 1} else "low",
            index % 7,
            (index % 5) / 10,
            "US" if index % 2 else "IN",
        )
        for index in range(60)
    )

    preflight = _preflight(columns, rows, "predict Revenue")

    assert preflight.status == "ready"
    assert any("RevenueBucket" in warning for warning in preflight.warnings)
    assert all(gate.status == "pass" for gate in preflight.gates)


def test_supervised_preflight_keeps_bounded_refetch_scope_and_warning() -> None:
    columns = ("TotalRevenue", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            100.0 + index * 3 + (index % 7),
            index % 7,
            (index % 5) / 10,
            "US" if index % 3 == 0 else "IN",
        )
        for index in range(60)
    )
    warning = "Preflight used up to MAX_ANALYSIS_ROWS=5000 rows from the validated SQL."

    preflight = _preflight(
        columns,
        rows,
        "predict TotalRevenue",
        row_scope="bounded_refetch",
        extra_warnings=(warning,),
    )

    assert preflight.status == "ready"
    assert preflight.row_scope == "bounded_refetch"
    assert warning in preflight.warnings
