from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from analysis.capabilities import ColumnCapability, DatasetCapability
from analysis.planner import AnalysisPlan


@dataclass(frozen=True)
class AnalysisResultTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class AnalysisExecutionResult:
    recipe: str
    status: str
    title: str
    summary: str
    tables: tuple[AnalysisResultTable, ...]
    warnings: tuple[str, ...] = ()


def execute_analysis_plan(
    plan: AnalysisPlan,
    capability: DatasetCapability,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    allow_bounded_rows: bool = False,
) -> AnalysisExecutionResult:
    if plan.recipe == "profile":
        return execute_profile(plan, capability)
    if plan.recipe == "correlation":
        return execute_correlation(
            plan,
            columns,
            rows,
            allow_bounded_rows=allow_bounded_rows,
        )
    return AnalysisExecutionResult(
        recipe=plan.recipe,
        status="plan_only",
        title="Analysis Plan Only",
        summary=f"{plan.recipe} execution is not implemented yet.",
        tables=(),
        warnings=plan.warnings,
    )


def execute_profile(
    plan: AnalysisPlan,
    capability: DatasetCapability,
) -> AnalysisExecutionResult:
    if plan.status != "ready":
        return _non_ready_result(plan)

    rows = tuple(_profile_row(column) for column in capability.columns)
    table = AnalysisResultTable(
        title="Column Profile",
        columns=("Column", "Type", "Non-null", "Nulls", "Distinct", "Notes"),
        rows=rows,
    )
    summary = (
        f"Profiled {capability.row_count} rows and {capability.column_count} columns. "
        f"Detected {len(capability.numeric_columns)} numeric, "
        f"{len(capability.categorical_columns)} categorical, and "
        f"{len(capability.datetime_columns)} datetime columns."
    )
    return AnalysisExecutionResult(
        recipe="profile",
        status="success",
        title="Profile Result",
        summary=summary,
        tables=(table,),
        warnings=plan.warnings,
    )


def execute_correlation(
    plan: AnalysisPlan,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    allow_bounded_rows: bool = False,
) -> AnalysisExecutionResult:
    if plan.status != "ready":
        return _non_ready_result(plan)
    if plan.row_scope != "artifact_rows" and not allow_bounded_rows:
        return AnalysisExecutionResult(
            recipe="correlation",
            status="blocked",
            title="Correlation Blocked",
            summary=(
                "Correlation execution is blocked for truncated artifacts unless bounded "
                "rows are materialized first."
            ),
            tables=(),
            warnings=plan.warnings,
        )
    if len(plan.feature_columns) < 2:
        return AnalysisExecutionResult(
            recipe="correlation",
            status="blocked",
            title="Correlation Blocked",
            summary="Need at least two numeric columns for correlation.",
            tables=(),
            warnings=plan.warnings,
        )

    index_by_name = {str(column): index for index, column in enumerate(columns)}
    pairs: list[tuple[str, str, float | None, int]] = []
    for left, right in combinations(plan.feature_columns, 2):
        left_index = index_by_name[left]
        right_index = index_by_name[right]
        paired = [
            (left_value, right_value)
            for row in rows
            if (left_value := _number_at(row, left_index)) is not None
            and (right_value := _number_at(row, right_index)) is not None
        ]
        correlation = _pearson(paired)
        pairs.append((left, right, correlation, len(paired)))

    pairs.sort(
        key=lambda item: (item[2] is None, -(abs(item[2]) if item[2] is not None else 0))
    )
    table = AnalysisResultTable(
        title="Pearson Correlations",
        columns=("Column A", "Column B", "Correlation", "Paired rows"),
        rows=tuple(
            (left, right, _format_correlation(correlation), paired_rows)
            for left, right, correlation, paired_rows in pairs
        ),
    )
    summary = f"Computed {len(pairs)} pairwise Pearson correlation(s)."
    return AnalysisExecutionResult(
        recipe="correlation",
        status="success",
        title="Correlation Result",
        summary=summary,
        tables=(table,),
        warnings=plan.warnings,
    )


def _profile_row(column: ColumnCapability) -> tuple[Any, ...]:
    notes: list[str] = []
    if column.is_id_like:
        notes.append("ID-like")
    if column.is_constant:
        notes.append("constant")
    if column.is_mostly_null:
        notes.append("mostly null")
    if column.is_high_cardinality:
        notes.append("high cardinality")
    if column.usable_as_feature:
        notes.append("feature candidate")
    if column.target_score > 0:
        notes.append("target candidate")
    return (
        column.name,
        column.kind,
        column.non_null_count,
        column.null_count,
        column.distinct_count,
        ", ".join(notes) if notes else "none",
    )


def _number_at(row: Sequence[Any], index: int) -> float | None:
    if index >= len(row):
        return None
    value = row[index]
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _pearson(paired: Sequence[tuple[float, float]]) -> float | None:
    count = len(paired)
    if count < 2:
        return None
    xs = [item[0] for item in paired]
    ys = [item[1] for item in paired]
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    denominator = denom_x * denom_y
    if denominator == 0:
        return None
    return numerator / denominator


def _format_correlation(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _non_ready_result(plan: AnalysisPlan) -> AnalysisExecutionResult:
    return AnalysisExecutionResult(
        recipe=plan.recipe,
        status=plan.status,
        title="Analysis Not Run",
        summary=plan.message,
        tables=(),
        warnings=plan.warnings,
    )
