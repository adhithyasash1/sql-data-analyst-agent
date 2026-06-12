from __future__ import annotations

import re
from dataclasses import dataclass

from analysis.capabilities import ColumnCapability, DatasetCapability


RECIPE_ALIASES = {
    "profile": ("profile", "describe", "summarize", "summary", "stats", "statistics"),
    "correlation": ("correlate", "correlation", "relationships", "relationship"),
    "predict": ("predict", "prediction", "forecast"),
    "explain": ("explain", "drivers", "drives", "importance", "why"),
}


@dataclass(frozen=True)
class FeatureDecision:
    column: str
    reason: str


@dataclass(frozen=True)
class AnalysisPlan:
    recipe: str
    status: str
    request: str
    target_column: str | None
    feature_columns: tuple[str, ...]
    excluded_columns: tuple[FeatureDecision, ...]
    row_scope: str
    row_count: int
    warnings: tuple[str, ...]
    message: str
    confirmation_required: bool


def plan_analysis(request: str, capability: DatasetCapability) -> AnalysisPlan:
    text = request.strip()
    if not text:
        return _plan_message(
            recipe="cannot_run",
            status="needs_clarification",
            request=request,
            capability=capability,
            message="Tell me what analysis you want, such as profile, correlate, predict, or explain.",
        )

    recipe = _infer_recipe(text)
    if recipe is None:
        return _plan_message(
            recipe="cannot_run",
            status="needs_clarification",
            request=request,
            capability=capability,
            message=(
                "I can plan profile, correlation, predict, or explain requests. "
                "Try ':analyze predict <column>' or ':analyze correlate numeric columns'."
            ),
        )

    if recipe == "profile":
        return _profile_plan(text, capability)
    if recipe == "correlation":
        return _correlation_plan(text, capability)
    return _supervised_plan(recipe, text, capability)


def _profile_plan(request: str, capability: DatasetCapability) -> AnalysisPlan:
    if "profile" not in capability.allowed_recipes:
        return _blocked_plan("profile", request, capability, _block_reason("profile", capability))
    return AnalysisPlan(
        recipe="profile",
        status="ready",
        request=request,
        target_column=None,
        feature_columns=(),
        excluded_columns=(),
        row_scope="artifact_rows",
        row_count=capability.row_count,
        warnings=_common_warnings(capability),
        message="Ready to profile this result with deterministic dataset statistics.",
        confirmation_required=False,
    )


def _correlation_plan(request: str, capability: DatasetCapability) -> AnalysisPlan:
    if "correlation" not in capability.allowed_recipes:
        return _blocked_plan(
            "correlation", request, capability, _block_reason("correlation", capability)
        )
    return AnalysisPlan(
        recipe="correlation",
        status="ready",
        request=request,
        target_column=None,
        feature_columns=capability.numeric_columns,
        excluded_columns=_feature_exclusions(capability, target_column=None),
        row_scope="bounded_refetch" if capability.truncated else "artifact_rows",
        row_count=capability.row_count,
        warnings=_common_warnings(capability),
        message="Ready to plan numeric correlation analysis.",
        confirmation_required=True,
    )


def _supervised_plan(
    recipe: str, request: str, capability: DatasetCapability
) -> AnalysisPlan:
    target = _target_from_request(request, capability)
    if target is None:
        if capability.target_candidates:
            return AnalysisPlan(
                recipe=recipe,
                status="needs_clarification",
                request=request,
                target_column=None,
                feature_columns=(),
                excluded_columns=(),
                row_scope="bounded_refetch" if capability.truncated else "artifact_rows",
                row_count=capability.row_count,
                warnings=_common_warnings(capability),
                message=(
                    "Choose a target column: "
                    + ", ".join(capability.target_candidates[:5])
                ),
                confirmation_required=False,
            )
        return _blocked_plan(recipe, request, capability, "no target-like column was detected")

    if recipe not in capability.allowed_recipes:
        return _blocked_plan(recipe, request, capability, _block_reason(recipe, capability), target)

    target_cap = _column_by_name(capability, target)
    if target_cap is None or target_cap.target_score <= 0:
        return _blocked_plan(
            recipe,
            request,
            capability,
            f"target column {target} is not suitable for supervised ML",
            target,
        )

    feature_columns = tuple(
        column for column in capability.usable_feature_columns if column != target
    )
    if len(feature_columns) < 2:
        return _blocked_plan(
            recipe,
            request,
            capability,
            "need at least 2 usable feature columns",
            target,
        )

    return AnalysisPlan(
        recipe=recipe,
        status="ready",
        request=request,
        target_column=target,
        feature_columns=feature_columns,
        excluded_columns=_feature_exclusions(capability, target_column=target),
        row_scope="bounded_refetch" if capability.truncated else "artifact_rows",
        row_count=capability.row_count,
        warnings=_common_warnings(capability),
        message=f"Ready to plan {recipe} analysis for {target}. No model will run yet.",
        confirmation_required=True,
    )


def _blocked_plan(
    recipe: str,
    request: str,
    capability: DatasetCapability,
    reason: str,
    target_column: str | None = None,
) -> AnalysisPlan:
    return AnalysisPlan(
        recipe=recipe,
        status="blocked",
        request=request,
        target_column=target_column,
        feature_columns=(),
        excluded_columns=_feature_exclusions(capability, target_column=target_column),
        row_scope="bounded_refetch" if capability.truncated else "artifact_rows",
        row_count=capability.row_count,
        warnings=_common_warnings(capability),
        message=reason,
        confirmation_required=False,
    )


def _plan_message(
    *,
    recipe: str,
    status: str,
    request: str,
    capability: DatasetCapability,
    message: str,
) -> AnalysisPlan:
    return AnalysisPlan(
        recipe=recipe,
        status=status,
        request=request,
        target_column=None,
        feature_columns=(),
        excluded_columns=(),
        row_scope="bounded_refetch" if capability.truncated else "artifact_rows",
        row_count=capability.row_count,
        warnings=_common_warnings(capability),
        message=message,
        confirmation_required=False,
    )


def _infer_recipe(request: str) -> str | None:
    words = set(_tokens(request))
    for recipe, aliases in RECIPE_ALIASES.items():
        if words.intersection(aliases):
            if recipe == "predict" and "forecast" in words:
                return None
            return recipe
    return None


def _target_from_request(request: str, capability: DatasetCapability) -> str | None:
    target_match = re.search(
        r"\b(?:predict|explain|target|for|drives?|driving)\s+(.+)$",
        request,
        flags=re.IGNORECASE,
    )
    candidates: list[str] = []
    if target_match is not None:
        candidates.append(target_match.group(1).strip(" ?.!,"))
    candidates.extend(_tokens(request))

    for candidate in candidates:
        resolved = _resolve_column(capability, candidate)
        if resolved is not None:
            return resolved
    if len(capability.target_candidates) == 1:
        return capability.target_candidates[0]
    return None


def _resolve_column(capability: DatasetCapability, requested: str) -> str | None:
    cleaned = requested.strip(" ?.!,")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    for column in capability.columns:
        if column.name.lower() == lowered:
            return column.name
    collapsed = _collapse(cleaned)
    for column in capability.columns:
        if _collapse(column.name) == collapsed:
            return column.name
    return None


def _feature_exclusions(
    capability: DatasetCapability, *, target_column: str | None
) -> tuple[FeatureDecision, ...]:
    exclusions: list[FeatureDecision] = []
    for column in capability.columns:
        reason = _exclusion_reason(column, target_column=target_column)
        if reason is not None:
            exclusions.append(FeatureDecision(column.name, reason))
    return tuple(exclusions)


def _exclusion_reason(column: ColumnCapability, *, target_column: str | None) -> str | None:
    if target_column is not None and column.name == target_column:
        return "target column"
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


def _common_warnings(capability: DatasetCapability) -> tuple[str, ...]:
    warnings: list[str] = []
    if capability.truncated:
        warnings.append("Displayed rows are truncated; heavier analysis should use a bounded re-fetch.")
    if capability.row_count < 50:
        warnings.append("Small row counts can make ML-style findings unstable.")
    return tuple(warnings)


def _block_reason(recipe: str, capability: DatasetCapability) -> str:
    for block in capability.blocked_recipes:
        if block.recipe == recipe:
            return block.reason
    return f"{recipe} is not available for this result"


def _column_by_name(capability: DatasetCapability, name: str) -> ColumnCapability | None:
    return next((column for column in capability.columns if column.name == name), None)


def _tokens(text: str) -> tuple[str, ...]:
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", spaced)
    return tuple(token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token)


def _collapse(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
