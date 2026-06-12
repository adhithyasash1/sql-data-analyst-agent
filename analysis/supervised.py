from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from analysis.capabilities import (
    MIN_CLASS_COUNT,
    MIN_FEATURE_COLUMNS,
    MIN_SUPERVISED_ROWS,
    ColumnCapability,
    DatasetCapability,
)
from analysis.planner import AnalysisPlan


NUMERIC_CLASSIFICATION_DISTINCT_LIMIT = 20
MIN_LEAKAGE_PAIRS = 10
PERFECT_CORRELATION_EPSILON = 1e-12


@dataclass(frozen=True)
class PreflightGate:
    name: str
    status: str
    reason: str


@dataclass(frozen=True)
class FeatureAssessment:
    column: str
    status: str
    reason: str


@dataclass(frozen=True)
class SupervisedPreflight:
    recipe: str
    status: str
    target_column: str | None
    problem_type: str
    row_scope: str
    row_count: int
    eligible_features: tuple[str, ...]
    feature_assessments: tuple[FeatureAssessment, ...]
    gates: tuple[PreflightGate, ...]
    baseline_recipe: str
    can_execute: bool
    warnings: tuple[str, ...]
    message: str


def build_supervised_preflight(
    plan: AnalysisPlan,
    capability: DatasetCapability,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    row_scope: str | None = None,
    extra_warnings: Sequence[str] = (),
) -> SupervisedPreflight:
    target = plan.target_column
    scope = row_scope or plan.row_scope
    warnings = list(dict.fromkeys((*plan.warnings, *extra_warnings)))

    if target is None:
        gate = PreflightGate("target", "block", "Choose a target column before supervised ML.")
        return _preflight_result(
            plan=plan,
            capability=capability,
            target=None,
            problem_type="unknown",
            row_scope=scope,
            eligible_features=(),
            assessments=(),
            gates=(gate,),
            warnings=tuple(warnings),
        )

    target_cap = _column_by_name(capability, target)
    if target_cap is None:
        gate = PreflightGate("target", "block", f"Target column {target} is not in the result.")
        return _preflight_result(
            plan=plan,
            capability=capability,
            target=target,
            problem_type="unsupported",
            row_scope=scope,
            eligible_features=(),
            assessments=(),
            gates=(gate,),
            warnings=tuple(warnings),
        )

    problem_type = infer_problem_type(target_cap)
    assessments = _assess_features(capability, target_cap, columns, rows, warnings)
    eligible_features = tuple(
        assessment.column for assessment in assessments if assessment.status == "eligible"
    )
    leakage_blocks = tuple(
        assessment for assessment in assessments if assessment.status == "blocked"
    )
    gates = _build_gates(
        target_cap,
        problem_type=problem_type,
        eligible_feature_count=len(eligible_features),
        leakage_blocks=leakage_blocks,
        rows=rows,
        columns=columns,
    )

    return _preflight_result(
        plan=plan,
        capability=capability,
        target=target,
        problem_type=problem_type,
        row_scope=scope,
        eligible_features=eligible_features,
        assessments=tuple(assessments),
        gates=tuple(gates),
        warnings=tuple(warnings),
    )


def infer_problem_type(target: ColumnCapability) -> str:
    if target.is_id_like or target.is_constant or target.is_mostly_null:
        return "unsupported"
    if target.kind in {"boolean", "categorical"}:
        return "classification"
    if target.kind == "numeric":
        repeated_values = target.distinct_count < target.non_null_count
        if (
            2 <= target.distinct_count <= NUMERIC_CLASSIFICATION_DISTINCT_LIMIT
            and repeated_values
        ):
            return "classification"
        return "regression"
    return "unsupported"


def _build_gates(
    target: ColumnCapability,
    *,
    problem_type: str,
    eligible_feature_count: int,
    leakage_blocks: Sequence[FeatureAssessment],
    rows: Sequence[Sequence[Any]],
    columns: Sequence[str],
) -> list[PreflightGate]:
    gates: list[PreflightGate] = []

    gates.append(
        _gate(
            "target",
            not target.is_id_like
            and not target.is_constant
            and not target.is_mostly_null
            and problem_type != "unsupported",
            f"Target {target.name} is valid for {problem_type}.",
            f"Target {target.name} is not suitable for supervised ML.",
        )
    )
    gates.append(
        _gate(
            "usable_rows",
            target.non_null_count >= MIN_SUPERVISED_ROWS,
            f"{target.non_null_count} target-labeled rows are available.",
            f"Need at least {MIN_SUPERVISED_ROWS} target-labeled rows.",
        )
    )
    gates.append(
        _gate(
            "feature_count",
            eligible_feature_count >= MIN_FEATURE_COLUMNS,
            f"{eligible_feature_count} eligible feature columns are available.",
            f"Need at least {MIN_FEATURE_COLUMNS} eligible feature columns.",
        )
    )

    if problem_type == "classification":
        class_counts = _target_class_counts(columns, rows, target.name)
        gates.append(
            _gate(
                "class_count",
                len(class_counts) >= 2,
                f"{len(class_counts)} target classes were detected.",
                "Classification needs at least two target classes.",
            )
        )
        min_class_count = min(class_counts.values()) if class_counts else 0
        gates.append(
            _gate(
                "class_balance",
                min_class_count >= MIN_CLASS_COUNT,
                f"Smallest class has {min_class_count} rows.",
                f"Each class needs at least {MIN_CLASS_COUNT} rows.",
            )
        )

    gates.append(
        _gate(
            "target_leakage",
            not leakage_blocks,
            "No high-confidence target leakage was detected.",
            "High-confidence target leakage was detected: "
            + ", ".join(item.column for item in leakage_blocks[:5]),
        )
    )
    return gates


def _assess_features(
    capability: DatasetCapability,
    target: ColumnCapability,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    warnings: list[str],
) -> list[FeatureAssessment]:
    assessments: list[FeatureAssessment] = []
    seen_warnings: set[str] = set(warnings)
    for column in capability.columns:
        if column.name == target.name:
            assessments.append(FeatureAssessment(column.name, "excluded", "target column"))
            continue

        reason = _feature_exclusion_reason(column)
        if reason is not None:
            assessments.append(FeatureAssessment(column.name, "excluded", reason))
            continue

        leakage_reason = _leakage_reason(target, column, columns, rows)
        if leakage_reason is not None:
            assessments.append(FeatureAssessment(column.name, "blocked", leakage_reason))
            continue

        if _names_resemble(column.name, target.name):
            warning = (
                f"Feature {column.name} name resembles target {target.name}; "
                "inspect it for leakage before fitting ML."
            )
            if warning not in seen_warnings:
                warnings.append(warning)
                seen_warnings.add(warning)

        assessments.append(FeatureAssessment(column.name, "eligible", "v1 feature candidate"))
    return assessments


def _feature_exclusion_reason(column: ColumnCapability) -> str | None:
    if column.is_id_like:
        return "ID-like column"
    if column.is_constant:
        return "constant column"
    if column.is_mostly_null:
        return "mostly-null column"
    if column.is_high_cardinality:
        return "high-cardinality text/category"
    if column.kind not in {"numeric", "categorical", "boolean"}:
        return f"{column.kind} column is not a v1 feature"
    if not column.usable_as_feature:
        return "not usable as a v1 feature"
    return None


def _leakage_reason(
    target: ColumnCapability,
    feature: ColumnCapability,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> str | None:
    if _normalized_name(feature.name) == _normalized_name(target.name):
        return "target alias"

    pairs = _paired_values(columns, rows, target.name, feature.name)
    if len(pairs) < MIN_LEAKAGE_PAIRS:
        return None
    if _exact_duplicate(pairs):
        return "duplicates target values"
    if target.kind == "numeric" and feature.kind == "numeric" and _perfect_correlation(pairs):
        return "perfectly correlated with target"
    if target.kind in {"categorical", "boolean"} or feature.kind in {"categorical", "boolean"}:
        if _one_to_one_mapping(pairs):
            return "one-to-one mapping with target"
    return None


def _paired_values(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    target: str,
    feature: str,
) -> list[tuple[Any, Any]]:
    index_by_name = {str(column): index for index, column in enumerate(columns)}
    if target not in index_by_name or feature not in index_by_name:
        return []
    target_index = index_by_name[target]
    feature_index = index_by_name[feature]
    pairs: list[tuple[Any, Any]] = []
    for row in rows:
        target_value = row[target_index] if target_index < len(row) else None
        feature_value = row[feature_index] if feature_index < len(row) else None
        if _is_null(target_value) or _is_null(feature_value):
            continue
        pairs.append((target_value, feature_value))
    return pairs


def _exact_duplicate(pairs: Sequence[tuple[Any, Any]]) -> bool:
    return all(_normal_key(target) == _normal_key(feature) for target, feature in pairs)


def _perfect_correlation(pairs: Sequence[tuple[Any, Any]]) -> bool:
    numeric_pairs = [
        (target_number, feature_number)
        for target, feature in pairs
        if (target_number := _coerce_number(target)) is not None
        and (feature_number := _coerce_number(feature)) is not None
    ]
    if len(numeric_pairs) < MIN_LEAKAGE_PAIRS:
        return False
    correlation = _pearson(numeric_pairs)
    return correlation is not None and abs(abs(correlation) - 1.0) <= PERFECT_CORRELATION_EPSILON


def _one_to_one_mapping(pairs: Sequence[tuple[Any, Any]]) -> bool:
    feature_to_target: dict[str, str] = {}
    target_to_feature: dict[str, str] = {}
    for target, feature in pairs:
        target_key = _normal_key(target)
        feature_key = _normal_key(feature)
        if feature_key in feature_to_target and feature_to_target[feature_key] != target_key:
            return False
        if target_key in target_to_feature and target_to_feature[target_key] != feature_key:
            return False
        feature_to_target[feature_key] = target_key
        target_to_feature[target_key] = feature_key
    return len(feature_to_target) >= 2 and len(target_to_feature) >= 2


def _target_class_counts(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    target: str,
) -> Counter[str]:
    index_by_name = {str(column): index for index, column in enumerate(columns)}
    if target not in index_by_name:
        return Counter()
    target_index = index_by_name[target]
    return Counter(
        _normal_key(row[target_index])
        for row in rows
        if target_index < len(row) and not _is_null(row[target_index])
    )


def _preflight_result(
    *,
    plan: AnalysisPlan,
    capability: DatasetCapability,
    target: str | None,
    problem_type: str,
    row_scope: str,
    eligible_features: tuple[str, ...],
    assessments: tuple[FeatureAssessment, ...],
    gates: tuple[PreflightGate, ...],
    warnings: tuple[str, ...],
) -> SupervisedPreflight:
    can_execute = all(gate.status != "block" for gate in gates)
    status = "ready" if can_execute else "blocked"
    baseline = _baseline_recipe(plan.recipe, problem_type)
    if can_execute:
        message = (
            f"Supervised {plan.recipe} preflight passed. "
            "No scikit-learn model was fitted."
        )
    else:
        message = (
            f"Supervised {plan.recipe} preflight is blocked. "
            "No scikit-learn model was fitted."
        )
    return SupervisedPreflight(
        recipe=plan.recipe,
        status=status,
        target_column=target,
        problem_type=problem_type,
        row_scope=row_scope,
        row_count=capability.row_count,
        eligible_features=eligible_features,
        feature_assessments=assessments,
        gates=gates,
        baseline_recipe=baseline,
        can_execute=can_execute,
        warnings=warnings,
        message=message,
    )


def _baseline_recipe(recipe: str, problem_type: str) -> str:
    if problem_type == "regression":
        base = "Median baseline, then Ridge"
    elif problem_type == "classification":
        base = "Majority-class baseline, then LogisticRegression"
    else:
        return "none"
    if recipe == "explain":
        return base + " with permutation importance"
    return base


def _gate(name: str, passes: bool, pass_reason: str, block_reason: str) -> PreflightGate:
    return PreflightGate(name, "pass" if passes else "block", pass_reason if passes else block_reason)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
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


def _normal_key(value: Any) -> str:
    if isinstance(value, float):
        return repr(round(value, 12))
    return str(value).strip().lower()


def _names_resemble(feature: str, target: str) -> bool:
    feature_name = _normalized_name(feature)
    target_name = _normalized_name(target)
    if feature_name == target_name or len(feature_name) < 4 or len(target_name) < 4:
        return False
    return feature_name in target_name or target_name in feature_name


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _column_by_name(capability: DatasetCapability, name: str) -> ColumnCapability | None:
    return next((column for column in capability.columns if column.name == name), None)
