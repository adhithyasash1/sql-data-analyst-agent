from __future__ import annotations

from analysis.capabilities import analyze_dataset_capability, suggest_analyses


def _column(capability, name: str):
    return next(column for column in capability.columns if column.name == name)


def test_capability_detects_column_roles_and_suggestions() -> None:
    columns = (
        "CustomerID",
        "TotalRevenue",
        "OrderCount",
        "OrderDate",
        "Country",
        "Constant",
        "Sparse",
    )
    rows = tuple(
        (
            f"C{index:03d}",
            100.0 + index,
            (index % 5) + 1,
            f"2026-01-{(index % 28) + 1:02d}",
            "US" if index % 2 else "IN",
            "same",
            "present" if index >= 56 else None,
        )
        for index in range(60)
    )

    capability = analyze_dataset_capability(columns, rows, truncated=True)

    assert capability.row_count == 60
    assert capability.truncated is True
    assert capability.numeric_columns == ("TotalRevenue", "OrderCount")
    assert capability.categorical_columns == ("Country",)
    assert capability.datetime_columns == ("OrderDate",)
    assert capability.id_like_columns == ("CustomerID",)
    assert capability.constant_columns == ("Constant", "Sparse")
    assert capability.mostly_null_columns == ("Sparse",)
    assert capability.target_candidates[0] == "TotalRevenue"
    assert _column(capability, "CustomerID").usable_as_feature is False
    assert _column(capability, "Country").usable_as_feature is True
    assert set(capability.allowed_recipes) == {"profile", "correlation", "predict", "explain"}

    suggestions = suggest_analyses(capability)

    assert [suggestion.recipe for suggestion in suggestions] == [
        "profile",
        "visualize",
        "correlation",
        "ml_optional",
    ]
    assert "`:plot line x=OrderDate y=TotalRevenue`" in suggestions[1].message
    assert "TotalRevenue" in suggestions[3].message


def test_capability_blocks_supervised_ml_when_rows_are_too_few() -> None:
    columns = ("TotalRevenue", "OrderCount", "Country")
    rows = tuple((100 + index, index % 5, "US" if index % 2 else "IN") for index in range(20))

    capability = analyze_dataset_capability(columns, rows)

    assert "correlation" in capability.allowed_recipes
    assert "predict" not in capability.allowed_recipes
    assert "explain" not in capability.allowed_recipes
    assert any(
        block.recipe == "predict" and "at least 50" in block.reason
        for block in capability.blocked_recipes
    )
    suggestions = suggest_analyses(capability)
    assert [suggestion.recipe for suggestion in suggestions] == [
        "profile",
        "visualize",
        "correlation",
        "ml_not_recommended",
    ]
    assert "Skip supervised ML" in suggestions[-1].message


def test_visualization_suggestion_prefers_bar_for_ranked_dimension() -> None:
    columns = ("GenreName", "TotalRevenue")
    rows = (
        ("Rock", 826.65),
        ("Latin", 382.14),
        ("Metal", 261.36),
    )

    capability = analyze_dataset_capability(columns, rows)

    suggestions = suggest_analyses(capability)
    assert [suggestion.recipe for suggestion in suggestions] == [
        "profile",
        "visualize",
        "ml_not_recommended",
    ]
    assert "`:plot bar x=GenreName y=TotalRevenue`" in suggestions[1].message


def test_visualization_suggestion_prefers_histogram_without_safe_dimension() -> None:
    columns = ("ShipVia", "Freight")
    rows = tuple(((index % 3) + 1, 10.0 + index) for index in range(10))

    capability = analyze_dataset_capability(columns, rows)

    suggestions = suggest_analyses(capability)
    assert suggestions[1].recipe == "visualize"
    assert "`:plot hist column=Freight`" in suggestions[1].message


def test_capability_blocks_classification_when_a_class_is_too_small() -> None:
    columns = ("Status", "FeatureA", "FeatureB", "Country")
    rows = tuple(
        (
            "churned" if index == 0 else "active",
            index % 7,
            50.0 + index,
            "US" if index % 2 else "IN",
        )
        for index in range(60)
    )

    capability = analyze_dataset_capability(columns, rows)

    assert capability.target_candidates[0] == "Status"
    assert "predict" not in capability.allowed_recipes
    assert any("at least 5 rows per class" in block.reason for block in capability.blocked_recipes)


def test_capability_does_not_treat_valid_or_paid_as_ids() -> None:
    columns = ("Valid", "Paid")
    rows = ((True, "yes"), (False, "no"), (True, "yes"))

    capability = analyze_dataset_capability(columns, rows)

    assert capability.id_like_columns == ()
    assert _column(capability, "Valid").kind == "boolean"
    assert _column(capability, "Paid").kind == "boolean"


def test_empty_capability_has_no_suggestions() -> None:
    capability = analyze_dataset_capability(("TotalRevenue",), ())

    assert capability.row_count == 0
    assert capability.allowed_recipes == ()
    assert suggest_analyses(capability) == ()
