from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any


MIN_SUPERVISED_ROWS = 50
MIN_FEATURE_COLUMNS = 2
MIN_CLASS_COUNT = 5
MOSTLY_NULL_RATIO = 0.8
HIGH_CARDINALITY_RATIO = 0.8


@dataclass(frozen=True)
class ColumnCapability:
    name: str
    kind: str
    non_null_count: int
    null_count: int
    distinct_count: int
    null_ratio: float
    distinct_ratio: float
    is_id_like: bool
    is_constant: bool
    is_mostly_null: bool
    is_high_cardinality: bool
    usable_as_feature: bool
    target_score: int
    target_reasons: tuple[str, ...]
    min_class_count: int | None = None


@dataclass(frozen=True)
class RecipeBlock:
    recipe: str
    reason: str


@dataclass(frozen=True)
class DatasetCapability:
    row_count: int
    column_count: int
    truncated: bool
    columns: tuple[ColumnCapability, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    id_like_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    mostly_null_columns: tuple[str, ...]
    high_cardinality_columns: tuple[str, ...]
    usable_feature_columns: tuple[str, ...]
    target_candidates: tuple[str, ...]
    allowed_recipes: tuple[str, ...]
    blocked_recipes: tuple[RecipeBlock, ...]


@dataclass(frozen=True)
class AnalysisSuggestion:
    recipe: str
    message: str
    reason: str


def analyze_dataset_capability(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    truncated: bool = False,
) -> DatasetCapability:
    row_count = len(rows)
    column_names = tuple(str(column) for column in columns)
    column_caps = tuple(
        _analyze_column(name, _column_values(rows, index), row_count)
        for index, name in enumerate(column_names)
    )

    numeric_columns = tuple(
        cap.name
        for cap in column_caps
        if cap.kind == "numeric"
        and not cap.is_id_like
        and not cap.is_constant
        and not cap.is_mostly_null
    )
    categorical_columns = tuple(
        cap.name
        for cap in column_caps
        if cap.kind in {"categorical", "boolean"}
        and not cap.is_id_like
        and not cap.is_constant
        and not cap.is_mostly_null
        and not cap.is_high_cardinality
    )
    datetime_columns = tuple(cap.name for cap in column_caps if cap.kind == "datetime")
    id_like_columns = tuple(cap.name for cap in column_caps if cap.is_id_like)
    constant_columns = tuple(cap.name for cap in column_caps if cap.is_constant)
    mostly_null_columns = tuple(cap.name for cap in column_caps if cap.is_mostly_null)
    high_cardinality_columns = tuple(cap.name for cap in column_caps if cap.is_high_cardinality)
    usable_feature_columns = tuple(cap.name for cap in column_caps if cap.usable_as_feature)
    target_candidates = tuple(
        cap.name
        for cap in sorted(
            column_caps,
            key=lambda item: (-item.target_score, -item.distinct_count, item.name.lower()),
        )
        if cap.target_score > 0
    )

    blocked = _blocked_recipes(row_count, column_caps, numeric_columns, target_candidates)
    blocked_names = {item.recipe for item in blocked}
    allowed = tuple(
        recipe
        for recipe in ("profile", "correlation", "predict", "explain")
        if recipe not in blocked_names
    )

    return DatasetCapability(
        row_count=row_count,
        column_count=len(column_names),
        truncated=truncated,
        columns=column_caps,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        id_like_columns=id_like_columns,
        constant_columns=constant_columns,
        mostly_null_columns=mostly_null_columns,
        high_cardinality_columns=high_cardinality_columns,
        usable_feature_columns=usable_feature_columns,
        target_candidates=target_candidates,
        allowed_recipes=allowed,
        blocked_recipes=tuple(blocked),
    )


def suggest_analyses(
    capability: DatasetCapability, *, max_suggestions: int = 4
) -> tuple[AnalysisSuggestion, ...]:
    if capability.row_count == 0 or capability.column_count == 0:
        return ()
    suggestions: list[AnalysisSuggestion] = []
    allowed = set(capability.allowed_recipes)

    if "profile" in allowed:
        suggestions.append(
            AnalysisSuggestion(
                recipe="profile",
                message="Profile first: `:analyze profile this result`",
                reason="lightweight profiling is safe for any non-empty artifact",
            )
        )

    visual = _visualization_suggestion(capability)
    if visual is not None:
        suggestions.append(visual)

    if "correlation" in allowed:
        suggestions.append(
            AnalysisSuggestion(
                recipe="correlation",
                message="Then check numeric relationships: `:analyze correlate numeric columns`",
                reason="at least two numeric columns were detected",
            )
        )

    target = capability.target_candidates[0] if capability.target_candidates else None
    if target and {"predict", "explain"} <= allowed:
        suggestions.append(
            AnalysisSuggestion(
                recipe="ml_optional",
                message=(
                    f"Advanced ML is optional: inspect `{target}` with "
                    f"`:analyze predict {target}` or `:analyze explain {target}`; "
                    "add `--run` only after preflight matches a real decision."
                ),
                reason="supervised ML passed coarse capability checks but remains explicit",
            )
        )

    if len(suggestions) < max_suggestions:
        blocked = {item.recipe: item.reason for item in capability.blocked_recipes}
        if "predict" in blocked:
            suggestions.append(
                AnalysisSuggestion(
                    recipe="ml_not_recommended",
                    message=f"Skip supervised ML for now: {blocked['predict']}.",
                    reason="the current artifact does not satisfy supervised ML gates",
                )
            )

    return tuple(suggestions[: max(0, max_suggestions)])


def _visualization_suggestion(capability: DatasetCapability) -> AnalysisSuggestion | None:
    if capability.row_count < 2:
        return None

    numeric = tuple(_column_by_name(capability, name) for name in capability.numeric_columns)
    numeric = tuple(column for column in numeric if column is not None)
    if not numeric:
        return None

    datetime_columns = tuple(
        column
        for column in capability.columns
        if column.kind == "datetime" and not column.is_id_like and _plot_safe(column.name)
    )
    metric = _preferred_metric(capability, numeric)
    if metric is None or not _plot_safe(metric.name):
        return None

    if datetime_columns:
        x_column = datetime_columns[0].name
        return AnalysisSuggestion(
            recipe="visualize",
            message=f"Visualize trend: `:plot line x={x_column} y={metric.name}`",
            reason="a datetime column and numeric metric were detected",
        )

    label = _preferred_label(capability)
    if label is not None and _plot_safe(label.name):
        return AnalysisSuggestion(
            recipe="visualize",
            message=f"Visualize ranking: `:plot bar x={label.name} y={metric.name}`",
            reason="a label-like dimension and numeric metric were detected",
        )

    return AnalysisSuggestion(
        recipe="visualize",
        message=f"Visualize distribution: `:plot hist column={metric.name}`",
        reason="a numeric metric was detected",
    )


def _preferred_metric(
    capability: DatasetCapability,
    numeric: Sequence[ColumnCapability],
) -> ColumnCapability | None:
    by_name = {column.name: column for column in numeric}
    for target in capability.target_candidates:
        column = by_name.get(target)
        if column is not None:
            return column
    return sorted(numeric, key=lambda column: (-column.target_score, column.name.lower()))[0]


def _preferred_label(capability: DatasetCapability) -> ColumnCapability | None:
    label_candidates = [
        column
        for column in capability.columns
        if column.kind in {"categorical", "text", "mixed"}
        and not column.is_id_like
        and not column.is_constant
        and not column.is_mostly_null
        and column.distinct_count > 1
    ]
    if not label_candidates:
        return None
    exact_grain = [
        column
        for column in label_candidates
        if column.distinct_count == capability.row_count and capability.row_count <= 100
    ]
    search_space = exact_grain or label_candidates
    return sorted(
        search_space,
        key=lambda column: (
            not _has_label_hint(column.name),
            column.distinct_count != capability.row_count,
            column.name.lower(),
        ),
    )[0]


def _column_by_name(capability: DatasetCapability, name: str) -> ColumnCapability | None:
    lower = name.lower()
    for column in capability.columns:
        if column.name.lower() == lower:
            return column
    return None


def _plot_safe(name: str) -> bool:
    return bool(name) and re.search(r"\s", name) is None


def _has_label_hint(name: str) -> bool:
    lower = name.lower()
    return any(
        hint in lower
        for hint in (
            "name",
            "title",
            "country",
            "city",
            "genre",
            "category",
            "product",
            "customer",
        )
    )


def _column_values(rows: Sequence[Sequence[Any]], index: int) -> list[Any]:
    return [row[index] if index < len(row) else None for row in rows]


def _analyze_column(name: str, values: Sequence[Any], row_count: int) -> ColumnCapability:
    null_count = sum(1 for value in values if _is_null(value))
    non_null = [value for value in values if not _is_null(value)]
    non_null_count = len(non_null)
    normalized_counts = Counter(_normal_key(value) for value in non_null)
    distinct_count = len(normalized_counts)
    null_ratio = null_count / row_count if row_count else 0.0
    distinct_ratio = distinct_count / non_null_count if non_null_count else 0.0
    kind = _infer_kind(name, non_null)
    is_constant = non_null_count > 0 and distinct_count <= 1
    is_mostly_null = row_count > 0 and null_ratio >= MOSTLY_NULL_RATIO
    is_high_cardinality = (
        non_null_count >= 20
        and distinct_ratio >= HIGH_CARDINALITY_RATIO
        and kind in {"categorical", "text", "mixed"}
    )
    is_id_like = _is_id_like_name(name) or (
        non_null_count >= 10
        and distinct_ratio >= 0.98
        and _has_identifier_hint(name)
    )
    usable_as_feature = (
        non_null_count > 0
        and kind in {"numeric", "categorical", "boolean"}
        and not is_constant
        and not is_mostly_null
        and not is_high_cardinality
        and not is_id_like
    )
    min_class_count = min(normalized_counts.values()) if normalized_counts else None
    target_score, target_reasons = _target_score(
        name,
        kind,
        distinct_count,
        is_id_like=is_id_like,
        is_constant=is_constant,
        is_mostly_null=is_mostly_null,
        is_high_cardinality=is_high_cardinality,
    )

    return ColumnCapability(
        name=name,
        kind=kind,
        non_null_count=non_null_count,
        null_count=null_count,
        distinct_count=distinct_count,
        null_ratio=null_ratio,
        distinct_ratio=distinct_ratio,
        is_id_like=is_id_like,
        is_constant=is_constant,
        is_mostly_null=is_mostly_null,
        is_high_cardinality=is_high_cardinality,
        usable_as_feature=usable_as_feature,
        target_score=target_score,
        target_reasons=target_reasons,
        min_class_count=min_class_count,
    )


def _blocked_recipes(
    row_count: int,
    columns: Sequence[ColumnCapability],
    numeric_columns: Sequence[str],
    target_candidates: Sequence[str],
) -> list[RecipeBlock]:
    blocked: list[RecipeBlock] = []
    if row_count == 0:
        blocked.append(RecipeBlock("profile", "no rows are available"))
    if row_count < 2 or len(numeric_columns) < 2:
        blocked.append(RecipeBlock("correlation", "need at least two numeric columns and two rows"))

    supervised_reason = _supervised_block_reason(row_count, columns, target_candidates)
    if supervised_reason is not None:
        blocked.append(RecipeBlock("predict", supervised_reason))
        blocked.append(RecipeBlock("explain", supervised_reason))
    return blocked


def _supervised_block_reason(
    row_count: int,
    columns: Sequence[ColumnCapability],
    target_candidates: Sequence[str],
) -> str | None:
    if row_count < MIN_SUPERVISED_ROWS:
        return f"need at least {MIN_SUPERVISED_ROWS} usable rows"
    if not target_candidates:
        return "no target-like column was detected"
    by_name = {column.name: column for column in columns}
    target = by_name[target_candidates[0]]
    if target.distinct_count <= 1:
        return f"target column {target.name} has no variation"
    if target.kind in {"categorical", "boolean"}:
        if target.distinct_count < 2:
            return f"target column {target.name} needs at least two classes"
        if target.min_class_count is not None and target.min_class_count < MIN_CLASS_COUNT:
            return (
                f"target column {target.name} needs at least "
                f"{MIN_CLASS_COUNT} rows per class"
            )
    usable_features = [
        column.name
        for column in columns
        if column.usable_as_feature and column.name != target.name
    ]
    if len(usable_features) < MIN_FEATURE_COLUMNS:
        return f"need at least {MIN_FEATURE_COLUMNS} usable feature columns"
    return None


def _infer_kind(name: str, values: Sequence[Any]) -> str:
    if not values:
        return "empty"
    if all(_is_boolean_value(value) for value in values):
        return "boolean"
    numeric_count = sum(1 for value in values if _coerce_number(value) is not None)
    datetime_count = sum(1 for value in values if _is_datetime_value(value))
    total = len(values)
    if numeric_count / total >= 0.9:
        return "numeric"
    if datetime_count / total >= 0.8 or (_has_datetime_hint(name) and datetime_count / total >= 0.6):
        return "datetime"
    normalized = {_normal_key(value) for value in values}
    if len(normalized) <= max(20, int(total * 0.2)):
        return "categorical"
    if all(isinstance(value, str) for value in values):
        return "text"
    return "mixed"


def _target_score(
    name: str,
    kind: str,
    distinct_count: int,
    *,
    is_id_like: bool,
    is_constant: bool,
    is_mostly_null: bool,
    is_high_cardinality: bool,
) -> tuple[int, tuple[str, ...]]:
    if is_id_like or is_constant or is_mostly_null or is_high_cardinality:
        return 0, ()
    if kind not in {"numeric", "categorical", "boolean"}:
        return 0, ()
    if distinct_count <= 1:
        return 0, ()

    score = 0
    reasons: list[str] = []
    words = set(_name_tokens(name))
    if kind == "numeric":
        score += 1
        reasons.append("numeric")
    if kind in {"categorical", "boolean"} and 2 <= distinct_count <= 20:
        score += 1
        reasons.append("low-cardinality target")
    for hint in _TARGET_HINTS:
        if hint in words:
            score += 2
            reasons.append(f"name contains {hint}")
            break
    return score, tuple(reasons)


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


def _is_boolean_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "false", "yes", "no"}
    return False


def _is_datetime_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}", text):
        return False
    normalized = text.replace("/", "-").replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return True


def _normal_key(value: Any) -> str:
    if isinstance(value, float):
        return repr(round(value, 12))
    return str(value).strip().lower()


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _name_tokens(name: str) -> tuple[str, ...]:
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
    return tuple(token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token)


def _is_id_like_name(name: str) -> bool:
    raw = name.strip()
    lower_raw = raw.lower()
    tokens = _name_tokens(name)
    return (
        lower_raw == "id"
        or lower_raw.endswith(("_id", " id", "-id", ".id"))
        or (raw.endswith(("ID", "Id")) and len(raw) > 2)
        or "uuid" in lower_raw
        or "guid" in lower_raw
        or (bool(tokens) and tokens[-1] in {"id", "key"})
    )


def _has_identifier_hint(name: str) -> bool:
    lower_raw = name.strip().lower()
    tokens = _name_tokens(name)
    return (
        _is_id_like_name(name)
        or "uuid" in lower_raw
        or "guid" in lower_raw
        or (bool(tokens) and tokens[-1] in {"key", "code", "number", "no"})
    )


def _has_datetime_hint(name: str) -> bool:
    lowered = _normalized_name(name)
    return any(hint in lowered for hint in ("date", "time", "timestamp", "created", "updated"))


_TARGET_HINTS = (
    "target",
    "label",
    "outcome",
    "churn",
    "revenue",
    "sales",
    "spend",
    "amount",
    "total",
    "score",
    "rate",
    "count",
    "quantity",
    "profit",
    "cost",
    "freight",
    "orders",
    "price",
    "status",
    "class",
    "category",
    "segment",
)
