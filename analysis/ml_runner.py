from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from analysis.capabilities import DatasetCapability
from analysis.supervised import SupervisedPreflight


RANDOM_STATE = 42
TEST_SIZE = 0.2
PERMUTATION_REPEATS = 5


@dataclass(frozen=True)
class SupervisedRunTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class SupervisedRunResult:
    recipe: str
    status: str
    target_column: str | None
    problem_type: str
    row_scope: str
    total_rows: int
    rows_used: int
    rows_dropped_missing_target: int
    train_rows: int
    test_rows: int
    feature_columns: tuple[str, ...]
    baseline_name: str
    model_name: str
    metric_name: str
    baseline_metric: float | None
    model_metric: float | None
    secondary_metric_name: str | None
    secondary_baseline_metric: float | None
    secondary_model_metric: float | None
    warnings: tuple[str, ...]
    tables: tuple[SupervisedRunTable, ...]
    message: str


def run_supervised_model(
    preflight: SupervisedPreflight,
    capability: DatasetCapability,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> SupervisedRunResult:
    if not preflight.can_execute:
        return _blocked_result(preflight, "Run blocked because supervised preflight did not pass.")

    deps = _load_ml_dependencies()
    if deps is None:
        return _blocked_result(
            preflight,
            "Install ML dependencies with `uv sync --extra ml` before using --run.",
            status="missing_dependency",
        )

    pd = deps["pd"]
    np = deps["np"]
    sklearn = deps["sklearn"]

    if preflight.target_column is None:
        return _blocked_result(preflight, "Run blocked because no target column was selected.")

    frame = _dataframe_from_rows(pd, np, columns, rows)
    target_series = _target_series(pd, frame[preflight.target_column], preflight.problem_type)
    non_missing_target = target_series.notna()
    rows_used = int(non_missing_target.sum())
    dropped_target = len(frame) - rows_used
    if rows_used < 2:
        return _blocked_result(
            preflight,
            "Run blocked because too few rows have a non-missing target.",
            total_rows=len(frame),
            rows_used=rows_used,
            dropped_target=dropped_target,
        )

    feature_columns = tuple(preflight.eligible_features)
    x_all = frame.loc[non_missing_target, list(feature_columns)]
    y_all = target_series.loc[non_missing_target]

    split = _split_data(sklearn, x_all, y_all, preflight.problem_type)
    if isinstance(split, str):
        return _blocked_result(
            preflight,
            split,
            total_rows=len(frame),
            rows_used=rows_used,
            dropped_target=dropped_target,
        )
    x_train, x_test, y_train, y_test = split

    pipeline = _build_pipeline(sklearn, capability, feature_columns, preflight.problem_type)
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = _score_predictions(
        np,
        sklearn,
        preflight.problem_type,
        y_train,
        y_test,
        predictions,
    )
    model_beats_baseline = metrics["model_beats_baseline"]
    status = "success" if model_beats_baseline else "weak_signal"
    warnings = list(preflight.warnings)
    if not model_beats_baseline:
        warnings.append("Model did not beat the fixed baseline; treat this as weak signal.")

    tables = [
        _metrics_table(metrics),
        _row_audit_table(len(frame), rows_used, dropped_target, len(x_train), len(x_test)),
    ]
    if preflight.recipe == "explain" and model_beats_baseline:
        tables.append(
            _permutation_importance_table(
                sklearn,
                pipeline,
                x_test,
                y_test,
                feature_columns,
                preflight.problem_type,
            )
        )
    elif preflight.recipe == "explain":
        warnings.append("Feature importance was skipped because the model did not beat baseline.")

    return SupervisedRunResult(
        recipe=preflight.recipe,
        status=status,
        target_column=preflight.target_column,
        problem_type=preflight.problem_type,
        row_scope=preflight.row_scope,
        total_rows=len(frame),
        rows_used=rows_used,
        rows_dropped_missing_target=dropped_target,
        train_rows=len(x_train),
        test_rows=len(x_test),
        feature_columns=feature_columns,
        baseline_name=metrics["baseline_name"],
        model_name=metrics["model_name"],
        metric_name=metrics["metric_name"],
        baseline_metric=metrics["baseline_metric"],
        model_metric=metrics["model_metric"],
        secondary_metric_name=metrics["secondary_metric_name"],
        secondary_baseline_metric=metrics["secondary_baseline_metric"],
        secondary_model_metric=metrics["secondary_model_metric"],
        warnings=tuple(dict.fromkeys(warnings)),
        tables=tuple(tables),
        message=(
            "Supervised ML run completed with a model that beat baseline."
            if model_beats_baseline
            else "Supervised ML run completed, but the model did not beat baseline."
        ),
    )


def _load_ml_dependencies() -> dict[str, Any] | None:
    try:
        import numpy as np
        import pandas as pd
        import sklearn.compose
        import sklearn.inspection
        import sklearn.impute
        import sklearn.linear_model
        import sklearn.metrics
        import sklearn.model_selection
        import sklearn.pipeline
        import sklearn.preprocessing
    except ImportError:
        return None

    return {"np": np, "pd": pd, "sklearn": sklearn}


def _dataframe_from_rows(pd, np, columns: Sequence[str], rows: Sequence[Sequence[Any]]):
    frame = pd.DataFrame(list(rows), columns=list(columns))
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    return frame.where(pd.notna(frame), np.nan)


def _target_series(pd, series, problem_type: str):
    if problem_type == "regression":
        return pd.to_numeric(series, errors="coerce")
    return series.astype("object")


def _split_data(sklearn, x_all, y_all, problem_type: str):
    stratify = y_all if problem_type == "classification" else None
    try:
        return sklearn.model_selection.train_test_split(
            x_all,
            y_all,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError as exc:
        return f"Run blocked because deterministic train/test split failed: {exc}"


def _build_pipeline(sklearn, capability: DatasetCapability, feature_columns, problem_type: str):
    kinds = {column.name: column.kind for column in capability.columns}
    numeric_features = [column for column in feature_columns if kinds.get(column) == "numeric"]
    categorical_features = [
        column for column in feature_columns if kinds.get(column) in {"categorical", "boolean"}
    ]
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                sklearn.pipeline.Pipeline(
                    steps=[
                        (
                            "imputer",
                            sklearn.impute.SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scaler", sklearn.preprocessing.StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                sklearn.pipeline.Pipeline(
                    steps=[
                        (
                            "imputer",
                            sklearn.impute.SimpleImputer(
                                strategy="most_frequent",
                                add_indicator=True,
                            ),
                        ),
                        (
                            "encoder",
                            sklearn.preprocessing.OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = sklearn.compose.ColumnTransformer(transformers=transformers)
    if problem_type == "regression":
        estimator = sklearn.linear_model.Ridge()
    else:
        estimator = sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return sklearn.pipeline.Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def _score_predictions(np, sklearn, problem_type: str, y_train, y_test, predictions) -> dict[str, Any]:
    if problem_type == "regression":
        baseline_value = float(np.median(y_train))
        baseline_predictions = np.full(shape=len(y_test), fill_value=baseline_value)
        baseline_mae = float(sklearn.metrics.mean_absolute_error(y_test, baseline_predictions))
        model_mae = float(sklearn.metrics.mean_absolute_error(y_test, predictions))
        baseline_r2 = float(sklearn.metrics.r2_score(y_test, baseline_predictions))
        model_r2 = float(sklearn.metrics.r2_score(y_test, predictions))
        return {
            "baseline_name": "Median baseline",
            "model_name": "Ridge",
            "metric_name": "MAE",
            "baseline_metric": baseline_mae,
            "model_metric": model_mae,
            "secondary_metric_name": "R2",
            "secondary_baseline_metric": baseline_r2,
            "secondary_model_metric": model_r2,
            "model_beats_baseline": model_mae < baseline_mae,
        }

    majority_class = Counter(y_train).most_common(1)[0][0]
    baseline_predictions = [majority_class] * len(y_test)
    baseline_accuracy = float(sklearn.metrics.accuracy_score(y_test, baseline_predictions))
    model_accuracy = float(sklearn.metrics.accuracy_score(y_test, predictions))
    baseline_balanced = float(
        sklearn.metrics.balanced_accuracy_score(y_test, baseline_predictions)
    )
    model_balanced = float(sklearn.metrics.balanced_accuracy_score(y_test, predictions))
    return {
        "baseline_name": "Majority-class baseline",
        "model_name": "LogisticRegression",
        "metric_name": "accuracy",
        "baseline_metric": baseline_accuracy,
        "model_metric": model_accuracy,
        "secondary_metric_name": "balanced_accuracy",
        "secondary_baseline_metric": baseline_balanced,
        "secondary_model_metric": model_balanced,
        "model_beats_baseline": model_accuracy > baseline_accuracy,
    }


def _metrics_table(metrics: dict[str, Any]) -> SupervisedRunTable:
    return SupervisedRunTable(
        title="Model Metrics",
        columns=("Metric", "Baseline", "Model"),
        rows=(
            (
                metrics["metric_name"],
                _format_metric(metrics["baseline_metric"]),
                _format_metric(metrics["model_metric"]),
            ),
            (
                metrics["secondary_metric_name"],
                _format_metric(metrics["secondary_baseline_metric"]),
                _format_metric(metrics["secondary_model_metric"]),
            ),
        ),
    )


def _row_audit_table(
    total_rows: int,
    rows_used: int,
    dropped_target: int,
    train_rows: int,
    test_rows: int,
) -> SupervisedRunTable:
    return SupervisedRunTable(
        title="Row Audit",
        columns=("Field", "Rows"),
        rows=(
            ("Materialized rows", total_rows),
            ("Rows with non-missing target", rows_used),
            ("Rows dropped due to missing target", dropped_target),
            ("Train rows", train_rows),
            ("Test rows", test_rows),
        ),
    )


def _permutation_importance_table(
    sklearn,
    pipeline,
    x_test,
    y_test,
    feature_columns: Sequence[str],
    problem_type: str,
) -> SupervisedRunTable:
    scoring = "neg_mean_absolute_error" if problem_type == "regression" else "accuracy"
    result = sklearn.inspection.permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring=scoring,
    )
    ranked = sorted(
        zip(feature_columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return SupervisedRunTable(
        title="Permutation Importance",
        columns=("Feature", "Mean importance", "Std dev"),
        rows=tuple(
            (feature, _format_metric(float(mean)), _format_metric(float(std)))
            for feature, mean, std in ranked[:10]
        ),
    )


def _blocked_result(
    preflight: SupervisedPreflight,
    message: str,
    *,
    status: str = "blocked",
    total_rows: int | None = None,
    rows_used: int = 0,
    dropped_target: int = 0,
) -> SupervisedRunResult:
    return SupervisedRunResult(
        recipe=preflight.recipe,
        status=status,
        target_column=preflight.target_column,
        problem_type=preflight.problem_type,
        row_scope=preflight.row_scope,
        total_rows=preflight.row_count if total_rows is None else total_rows,
        rows_used=rows_used,
        rows_dropped_missing_target=dropped_target,
        train_rows=0,
        test_rows=0,
        feature_columns=preflight.eligible_features,
        baseline_name="none",
        model_name="none",
        metric_name="none",
        baseline_metric=None,
        model_metric=None,
        secondary_metric_name=None,
        secondary_baseline_metric=None,
        secondary_model_metric=None,
        warnings=preflight.warnings,
        tables=(),
        message=message,
    )


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3f}"
