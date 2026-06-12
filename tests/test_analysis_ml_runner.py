from __future__ import annotations

import builtins
import importlib.util

import pytest

from analysis.capabilities import analyze_dataset_capability
from analysis.ml_runner import run_supervised_model
from analysis.planner import plan_analysis
from analysis.supervised import build_supervised_preflight


ML_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("numpy", "pandas", "sklearn")
)


def _preflight(columns, rows, request: str):
    capability = analyze_dataset_capability(columns, rows)
    plan = plan_analysis(request, capability)
    return build_supervised_preflight(plan, capability, columns, rows), capability


def test_run_supervised_model_reports_missing_ml_extra(monkeypatch) -> None:
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
    preflight, capability = _preflight(columns, rows, "predict TotalRevenue")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"numpy", "pandas"} or name.startswith("sklearn"):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = run_supervised_model(preflight, capability, columns, rows)

    assert result.status == "missing_dependency"
    assert "uv sync --extra ml" in result.message


@pytest.mark.skipif(not ML_AVAILABLE, reason="requires optional ml extra")
def test_run_supervised_model_regression_beats_median_baseline_and_audits_rows() -> None:
    columns = ("TotalRevenue", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            None
            if index == 0
            else (
                100.0
                + (index % 7) * 20
                + ((index % 5) / 10) * 50
                + (12 if index % 3 == 0 else -8)
                + (index % 13) * 0.1
            ),
            None if index % 11 == 0 else index % 7,
            (index % 5) / 10,
            "US" if index % 3 == 0 else "IN",
        )
        for index in range(80)
    )
    preflight, capability = _preflight(columns, rows, "predict TotalRevenue")

    result = run_supervised_model(preflight, capability, columns, rows)

    assert result.status == "success"
    assert result.problem_type == "regression"
    assert result.baseline_name == "Median baseline"
    assert result.model_name == "Ridge"
    assert result.metric_name == "MAE"
    assert result.model_metric is not None
    assert result.baseline_metric is not None
    assert result.model_metric < result.baseline_metric
    assert result.rows_dropped_missing_target == 1
    assert result.train_rows + result.test_rows == result.rows_used


@pytest.mark.skipif(not ML_AVAILABLE, reason="requires optional ml extra")
def test_run_supervised_model_classification_uses_logistic_regression() -> None:
    columns = ("Status", "OrderCount", "SpendScore", "Country")
    rows = tuple(
        (
            "churned" if index % 3 == 0 else "active",
            index % 7,
            50.0 + (index % 11),
            ("US", "IN", "GB")[index % 3],
        )
        for index in range(90)
    )
    preflight, capability = _preflight(columns, rows, "predict Status")

    result = run_supervised_model(preflight, capability, columns, rows)

    assert result.problem_type == "classification"
    assert result.baseline_name == "Majority-class baseline"
    assert result.model_name == "LogisticRegression"
    assert result.metric_name == "accuracy"
    assert result.secondary_metric_name == "balanced_accuracy"
    assert result.train_rows + result.test_rows == result.rows_used


@pytest.mark.skipif(not ML_AVAILABLE, reason="requires optional ml extra")
def test_run_supervised_explain_adds_test_set_permutation_importance() -> None:
    columns = ("TotalRevenue", "OrderCount", "DiscountRate", "Country")
    rows = tuple(
        (
            100.0
            + (index % 7) * 20
            + ((index % 5) / 10) * 50
            + (12 if index % 3 == 0 else -8)
            + (index % 13) * 0.1,
            index % 7,
            (index % 5) / 10,
            "US" if index % 3 == 0 else "IN",
        )
        for index in range(80)
    )
    preflight, capability = _preflight(columns, rows, "explain TotalRevenue")

    result = run_supervised_model(preflight, capability, columns, rows)

    assert result.status == "success"
    assert any(table.title == "Permutation Importance" for table in result.tables)
