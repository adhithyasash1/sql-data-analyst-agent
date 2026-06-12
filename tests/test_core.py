from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import core
from core import (
    AnswerResult,
    AnalysisArtifact,
    AnalysisArtifactTable,
    AppError,
    IndexStateError,
    ResultArtifact,
    SQL_REPAIR_RULES,
    SqlValidationError,
    analyze_execution,
    answer_question,
    artifact_describe_text,
    artifact_preview_rows,
    build_schema_document,
    build_shape_repair_prompt,
    build_summary_payload,
    build_sql_prompt,
    build_validation_repair_prompt,
    check_result_shape,
    dataframe_profile_text,
    deterministic_summary,
    embed_texts,
    execute_readonly_query,
    export_artifact_csv,
    extract_schema,
    extract_sql,
    get_kv,
    infer_shape_expectation,
    index_schema,
    log_query,
    export_artifact_chart,
    _chart_numeric,
    make_result_artifact,
    make_unique_column_names,
    metadata_connection,
    ArtifactTransformResult,
    _compare_values,
    parse_colon_command,
    parse_count,
    parse_key_value_args,
    parse_plot_command_args,
    parse_transform_args,
    resolve_column,
    route_artifact_followup,
    list_saved_workspaces,
    inspect_workspace,
    delete_workspace,
    load_artifact_workspace,
    sanitize_workspace_name,
    save_artifact_workspace,
    transform_artifact,
    transform_artifact_filter,
    transform_artifact_groupby,
    transform_artifact_select,
    transform_artifact_sort,
    profile_result_dataframe,
    quote_guidance,
    quote_identifier,
    read_query_logs,
    result_to_arrow_table,
    retrieve_schema,
    schema_documents_for_index,
    schema_fingerprint,
    set_kv,
    setup_metadata,
    translate_model_error,
    validate_sql,
    verify_sqlite_database,
)
from config import Settings, default_metadata_path_for_source


def create_small_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE Customers (
                CustomerID TEXT PRIMARY KEY,
                CompanyName TEXT NOT NULL
            );
            CREATE TABLE Orders (
                OrderID INTEGER PRIMARY KEY,
                CustomerID TEXT NOT NULL,
                FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
            );
            CREATE TABLE Products (
                ProductID INTEGER PRIMARY KEY,
                SupplierID INTEGER NOT NULL,
                ProductName TEXT NOT NULL
            );
            CREATE TABLE Suppliers (
                SupplierID INTEGER PRIMARY KEY,
                CompanyName TEXT NOT NULL
            );
            INSERT INTO Customers VALUES ('ALFKI', 'Alfreds');
            INSERT INTO Customers VALUES ('SAVEA', 'Save-a-lot Markets');
            INSERT INTO Orders VALUES (1, 'SAVEA');
            INSERT INTO Orders VALUES (2, 'SAVEA');
            INSERT INTO Orders VALUES (3, 'ALFKI');
            INSERT INTO Suppliers VALUES (1, 'Exotic Liquids');
            INSERT INTO Products VALUES (1, 1, 'Chai');
            """
        )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "northwind.db"
    create_small_db(path)
    return path


def test_schema_extraction_returns_tables(db_path: Path) -> None:
    schema = extract_schema(db_path)
    names = {table.name for table in schema}
    assert names == {"Customers", "Orders", "Products", "Suppliers"}
    orders = next(table for table in schema if table.name == "Orders")
    assert orders.foreign_keys[0].referred_table == "Customers"


def test_sql_validator_accepts_valid_select(db_path: Path) -> None:
    schema = extract_schema(db_path)
    validate_sql(
        """
        SELECT c.CompanyName, COUNT(*) AS order_count
        FROM Customers AS c
        JOIN Orders AS o ON c.CustomerID = o.CustomerID
        GROUP BY c.CustomerID, c.CompanyName
        ORDER BY order_count DESC
        """,
        schema,
    )


def test_sql_validator_rejects_drop(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("DROP TABLE Customers", schema)


def test_sql_validator_rejects_multiple_statements(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT * FROM Customers; SELECT * FROM Orders", schema)


def test_sql_validator_rejects_pragma(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("PRAGMA table_info(Customers)", schema)


def test_sql_validator_allows_forbidden_words_inside_literals(db_path: Path) -> None:
    schema = extract_schema(db_path)
    validate_sql("SELECT CompanyName FROM Customers WHERE CompanyName = 'Drop Shop'", schema)


def test_sql_validator_rejects_recursive_cte(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql(
            """
            WITH RECURSIVE nums(n) AS (
                SELECT 1
                UNION ALL
                SELECT n + 1 FROM nums WHERE n < 3
            )
            SELECT n FROM nums
            """,
            schema,
        )


def test_sql_validator_rejects_unknown_columns(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT MissingColumn FROM Customers", schema)


def test_sql_validator_rejects_ambiguous_unqualified_columns(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql(
            """
            SELECT SupplierID, CompanyName, COUNT(ProductID) AS ProductCount
            FROM Suppliers
            JOIN Products ON Suppliers.SupplierID = Products.SupplierID
            GROUP BY SupplierID
            """,
            schema,
        )


def test_sql_validator_rejects_prose_output(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("I will query Customers. SELECT CompanyName FROM Customers", schema)


def test_sql_validator_rejects_columns_without_from(db_path: Path) -> None:
    schema = extract_schema(db_path)
    with pytest.raises(SqlValidationError):
        validate_sql("SELECT CustomerID, COUNT(*) AS OrderCount", schema)


def test_extract_sql_uses_last_fenced_sql_block() -> None:
    raw = """
    I will think first.
    ```sql
    SELECT CustomerID FROM Customers
    ```
    Final:
    ```sql
    SELECT CompanyName FROM Customers
    ```
    """
    assert extract_sql(raw) == "SELECT CompanyName FROM Customers"


def test_extract_sql_uses_final_select_line() -> None:
    raw = """
    Reasoning about the query.
    Final query:
    SELECT e.EmployeeID, e.FirstName, e.LastName, COUNT(*) AS OrderCount FROM Orders o JOIN Employees e ON o.EmployeeID = e.EmployeeID GROUP BY e.EmployeeID, e.FirstName, e.LastName ORDER BY OrderCount DESC LIMIT 50
    """
    assert extract_sql(raw).startswith("SELECT e.EmployeeID")


def test_extract_sql_keeps_multiline_final_sql_block() -> None:
    raw = """
    Reasoning about freight.
    Final query:
    SELECT ShipCountry, SUM(Freight) AS TotalFreight
    FROM Orders
    GROUP BY ShipCountry
    ORDER BY TotalFreight DESC
    LIMIT 50

    That is the final query.
    """
    sql = extract_sql(raw)
    assert "FROM Orders" in sql
    assert sql.startswith("SELECT ShipCountry")


def test_readonly_execution_returns_rows(db_path: Path) -> None:
    result = execute_readonly_query(
        db_path,
        "SELECT CompanyName FROM Customers ORDER BY CompanyName",
        max_rows=10,
        timeout_ms=3000,
    )
    assert result.columns == ("CompanyName",)
    assert result.rows == (("Alfreds",), ("Save-a-lot Markets",))


def test_readonly_execution_enforces_row_limit(db_path: Path) -> None:
    result = execute_readonly_query(
        db_path,
        "SELECT OrderID FROM Orders ORDER BY OrderID",
        max_rows=2,
        timeout_ms=3000,
    )
    assert len(result.rows) == 2
    assert result.truncated is True


def test_schema_fingerprint_changes_on_schema_change(db_path: Path) -> None:
    before = schema_fingerprint(extract_schema(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE Customers ADD COLUMN City TEXT")
    after = schema_fingerprint(extract_schema(db_path))
    assert before != after


def test_shape_check_flags_ranked_question_without_order_by() -> None:
    check = check_result_shape(
        "Which customers placed the most orders?",
        """
        SELECT c.CompanyName, COUNT(*) AS order_count
        FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID
        GROUP BY c.CompanyName
        """,
        ("CompanyName", "order_count"),
        (("Save-a-lot Markets", 2),),
    )
    assert check.ok is False
    assert check.warning is not None
    assert "ORDER BY" in check.warning


def test_shape_check_flags_top_n_result_with_too_many_rows() -> None:
    check = check_result_shape(
        "What are the top 5 products by quantity ordered?",
        """
        SELECT p.ProductName, SUM(od.Quantity) AS TotalQuantity
        FROM Products p JOIN "Order Details" od ON p.ProductID = od.ProductID
        GROUP BY p.ProductName
        ORDER BY TotalQuantity DESC
        LIMIT 50
        """,
        ("ProductName", "TotalQuantity"),
        tuple((f"Product {index}", index) for index in range(6)),
    )
    assert check.ok is False
    assert check.warning is not None
    assert "at most 5 rows" in check.warning


def test_shape_check_flags_employee_id_without_name() -> None:
    check = check_result_shape(
        "Which employees handled the most orders?",
        """
        SELECT e.EmployeeID, COUNT(*) AS OrderCount
        FROM Orders o JOIN Employees e ON o.EmployeeID = e.EmployeeID
        GROUP BY e.EmployeeID
        ORDER BY OrderCount DESC
        """,
        ("EmployeeID", "OrderCount"),
        ((4, 1908), (3, 1846)),
    )
    assert check.ok is False
    assert check.warning is not None
    assert "first, last, name" in check.warning


def test_shape_expectation_uses_question_subject_before_metric_object() -> None:
    supplier = infer_shape_expectation("Which suppliers provide the most products?")
    employee = infer_shape_expectation("Which employees handled the most orders?")
    assert supplier.entity_terms == ("supplier", "company", "contact")
    assert employee.entity_terms == ("first", "last", "name")


def test_shape_expectation_treats_most_recent_as_temporal_order() -> None:
    expectation = infer_shape_expectation("List the 10 most recent orders.")
    assert expectation.requires_order is True
    assert expectation.order_direction == "DESC"
    assert expectation.requires_numeric is False
    assert expectation.aggregate_kind is None
    assert expectation.expected_limit == 10


def test_shape_expectation_does_not_count_country_substring() -> None:
    expectation = infer_shape_expectation(
        "List order id, ship country, order date, and freight for orders."
    )
    assert expectation.requires_numeric is False
    assert expectation.aggregate_kind is None
    assert expectation.requires_order is False


def test_empty_results_are_summarized_without_repair_signal() -> None:
    summary = deterministic_summary("List discontinued products", ("ProductName",), (), False)
    assert summary == "The query returned no rows."


def test_deterministic_summary_skips_constant_entity_label() -> None:
    summary = deterministic_summary(
        "Which customers placed the most orders?",
        ("EntityLabel", "CustomerID", "CompanyName", "OrderCount"),
        (
            ("customer", "BSBEV", "B's Beverages", 210),
            ("customer", "LILAS", "LILA-Supermercado", 203),
        ),
        False,
    )
    assert summary == "Top result: B's Beverages with 210."


def test_deterministic_summary_combines_first_and_last_name() -> None:
    summary = deterministic_summary(
        "Which employees handled the most orders?",
        ("EmployeeID", "FirstName", "LastName", "OrderCount"),
        ((4, "Margaret", "Peacock", 1908),),
        False,
    )
    assert summary == "Top result: Margaret Peacock with 1,908."


def test_deterministic_summary_finds_highest_metric_when_not_ranked() -> None:
    summary = deterministic_summary(
        "What is the total freight by ship country?",
        ("ShipCountry", "TotalFreight"),
        (("Argentina", 140088.5), ("USA", 573251.0), ("France", 442055.75)),
        False,
    )
    assert summary == "Highest TotalFreight is USA with 573,251."


def test_deterministic_summary_does_not_treat_ids_as_metrics() -> None:
    summary = deterministic_summary(
        "List customers in Germany who placed orders in 2023.",
        ("CustomerID", "CompanyName", "OrderID", "OrderDate"),
        (("ALFKI", "Alfreds Futterkiste", 26376, "2023-10-15"),),
        False,
    )
    assert summary == "Returned 1 row."


def test_deterministic_summary_does_not_rank_detail_recency_by_metric() -> None:
    summary = deterministic_summary(
        "List the 10 most recent invoices.",
        ("InvoiceId", "CustomerId", "InvoiceDate", "Total"),
        (
            (412, 58, "2025-12-22 00:00:00", 1.99),
            (411, 44, "2025-11-13 00:00:00", 25.86),
        ),
        False,
    )
    assert summary == "Returned 2 rows."


def test_deterministic_summary_does_not_rank_detail_list_by_freight() -> None:
    summary = deterministic_summary(
        "List order id, employee id, ship via, ship country, ship city, order date, and freight for orders.",
        ("OrderID", "EmployeeID", "ShipVia", "ShipCountry", "ShipCity", "OrderDate", "Freight"),
        (
            (10248, 5, 3, "France", "Reims", "2016-07-04", 16.75),
            (10263, 9, 3, "Austria", "Graz", "2016-07-23", 56),
        ),
        False,
    )
    assert summary == "Returned 2 rows."


class FakeEmbeddingData:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data: list[FakeEmbeddingData]) -> None:
        self.data = data


class FakeEmbeddings:
    def create(self, model: str, input: list[str]) -> FakeEmbeddingResponse:
        del model
        data = []
        for index, text in enumerate(input):
            lower = text.lower()
            if "customer" in lower:
                embedding = [1.0, 0.0, 0.0, 0.0]
            elif "order" in lower:
                embedding = [0.0, 1.0, 0.0, 0.0]
            else:
                embedding = [0.0, 0.0, 1.0, 0.0]
            data.append(FakeEmbeddingData(index, embedding))
        return FakeEmbeddingResponse(data)


class FakeClient:
    embeddings = FakeEmbeddings()


def test_retrieval_returns_schema_object(db_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("sqlite_vec")
    settings = Settings(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        embedding_batch_size=2,
        retrieval_top_k=2,
        enable_schema_profiling=False,
    )
    try:
        index_schema(settings, FakeClient())  # type: ignore[arg-type]
        retrieved = retrieve_schema(settings, FakeClient(), "Which customers placed orders?")  # type: ignore[arg-type]
    except IndexStateError as exc:
        pytest.skip(str(exc))
    assert retrieved
    assert any(item.table_name == "Customers" for item in retrieved)


def test_metadata_connection_commits_on_success(tmp_path: Path) -> None:
    path = tmp_path / "meta.db"
    with metadata_connection(path) as conn:
        setup_metadata(conn)
        set_kv(conn, "probe", "value")
    with metadata_connection(path) as conn:
        assert get_kv(conn, "probe") == "value"


def test_metadata_connection_rolls_back_on_error(tmp_path: Path) -> None:
    path = tmp_path / "meta.db"
    with metadata_connection(path) as conn:
        setup_metadata(conn)
    with pytest.raises(RuntimeError):
        with metadata_connection(path) as conn:
            set_kv(conn, "probe", "value")
            raise RuntimeError("boom")
    with metadata_connection(path) as conn:
        assert get_kv(conn, "probe") is None


def test_metadata_connection_closes_after_block(tmp_path: Path) -> None:
    path = tmp_path / "meta.db"
    with metadata_connection(path) as conn:
        setup_metadata(conn)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_quote_identifier_escapes_embedded_quotes() -> None:
    assert quote_identifier("Order Details") == '"Order Details"'
    assert quote_identifier('a"b') == '"a""b"'


def test_extract_schema_parses_column_attributes(tmp_path: Path) -> None:
    path = tmp_path / "widgets.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE Widgets (
                WidgetID INTEGER PRIMARY KEY,
                Name TEXT NOT NULL,
                Price REAL DEFAULT 0,
                Note TEXT
            );
            """
        )
    table = next(t for t in extract_schema(path) if t.name == "Widgets")
    by_name = {column.name: column for column in table.columns}
    assert table.primary_key == ("WidgetID",)
    assert by_name["WidgetID"].primary_key is True
    assert by_name["Name"].nullable is False
    assert by_name["Note"].nullable is True
    assert by_name["Note"].default is None
    assert by_name["Price"].default == "0"


def test_extract_schema_handles_spaced_table_and_composite_pk(tmp_path: Path) -> None:
    path = tmp_path / "details.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE "Order Details" (
                OrderID INTEGER NOT NULL,
                ProductID INTEGER NOT NULL,
                Quantity INTEGER NOT NULL,
                PRIMARY KEY (OrderID, ProductID)
            );
            """
        )
    table = next(t for t in extract_schema(path) if t.name == "Order Details")
    assert table.primary_key == ("OrderID", "ProductID")
    assert {column.name for column in table.columns} == {"OrderID", "ProductID", "Quantity"}


COMMON_RULE_LINES = (
    "- Return SQL only. Do not explain your reasoning.",
    "- Do not include analysis, prose, bullets, or markdown.",
    "- Use only the provided tables and columns.",
    "- Every selected source column must come from an explicit FROM or JOIN table.",
    "- Use explicit JOIN conditions.",
)


def test_prompt_builders_share_common_rule_lines() -> None:
    expectation = infer_shape_expectation("List customers")
    initial = build_sql_prompt("List customers", "SCHEMA", 50, expectation)
    validation = build_validation_repair_prompt(
        "List customers", "SCHEMA", "BAD", "err", 50, expectation
    )
    shape = build_shape_repair_prompt("List customers", "SCHEMA", "SQL", {"a": 1}, {"b": 2}, 50)
    for line in COMMON_RULE_LINES:
        assert line in initial
        assert line in validation
        assert line in shape


def test_repair_prompts_share_identical_rules_block() -> None:
    rendered = SQL_REPAIR_RULES.format(quote_guidance=quote_guidance(), max_result_rows=50)
    validation = build_validation_repair_prompt(
        "q", "SCHEMA", "BAD", "err", 50, infer_shape_expectation("q")
    )
    shape = build_shape_repair_prompt("q", "SCHEMA", "SQL", {}, {}, 50)
    assert rendered in validation
    assert rendered in shape


def test_initial_prompt_keeps_dynamic_and_unique_lines() -> None:
    expectation = infer_shape_expectation("top 5 products")
    prompt = build_sql_prompt("top 5 products", "SCHEMA_TEXT", 50, expectation)
    assert "add LIMIT 50 for detailed row queries." in prompt
    assert "- Prefer explicit columns over SELECT *." in prompt
    assert "- Return only SQL. Do not use markdown fences." in prompt
    assert "SCHEMA_TEXT" in prompt
    assert "top 5 products" in prompt
    assert "Corrected SQL:" not in prompt
    assert prompt.rstrip().endswith("SQL:")


def test_validation_repair_prompt_keeps_dynamic_sections() -> None:
    prompt = build_validation_repair_prompt(
        "q", "SCHEMA_TEXT", "BAD SQL", "the error", 50, infer_shape_expectation("q")
    )
    assert "Validation error:\nthe error" in prompt
    assert "Invalid SQL:\nBAD SQL" in prompt
    assert "SCHEMA_TEXT" in prompt
    assert prompt.rstrip().endswith("Corrected SQL:")


def test_shape_repair_prompt_keeps_dynamic_sections() -> None:
    prompt = build_shape_repair_prompt(
        "q", "SCHEMA_TEXT", "PREV SQL", {"requires_order": True}, {"row_count": 7}, 50
    )
    assert "Previous SQL:\nPREV SQL" in prompt
    assert "SCHEMA_TEXT" in prompt
    assert "requires_order" in prompt
    assert "row_count" in prompt
    assert prompt.rstrip().endswith("Corrected SQL:")


def test_log_query_is_best_effort_when_metadata_db_unwritable(tmp_path: Path) -> None:
    blocked = tmp_path / "metadata.db"
    blocked.mkdir()  # a directory at the DB path makes sqlite unable to open it
    settings = Settings(
        omlx_api_key="test",
        db_path=tmp_path / "northwind.db",
        metadata_db_path=blocked,
    )
    result = AnswerResult(question="hi", retrieved_tables=["Customers"], expanded_tables=[])
    result.success = True

    log_query(settings, result)  # must not raise


def test_read_query_logs_returns_dict_rows_with_expected_keys(tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        db_path=tmp_path / "northwind.db",
        metadata_db_path=tmp_path / "metadata.db",
    )
    result = AnswerResult(question="hi", retrieved_tables=["Customers"], expanded_tables=[])
    result.sql = "SELECT 1"
    result.success = True
    result.executed = True
    log_query(settings, result)

    rows = read_query_logs(settings, 10)
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "hi"
    assert row["generated_sql"] == "SELECT 1"
    assert row["success"] == 1
    assert set(row.keys()) == {
        "id",
        "question",
        "generated_sql",
        "initial_sql",
        "success",
        "executed",
        "cancelled",
        "error_message",
        "validation_error",
        "repair_reason",
        "repaired",
        "retrieved_objects",
        "summary",
        "summary_mode",
        "summary_error",
        "shape_warning",
        "execution_ms",
        "row_count",
        "created_at",
    }


class FakeChatCompletions:
    def __init__(self, sql_responses: list[str]) -> None:
        self._responses = list(sql_responses)
        self.calls = 0

    def create(self, model: str, messages: list, **kwargs: object) -> SimpleNamespace:
        del model, messages, kwargs
        content = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )


class FakeSqlClient:
    def __init__(self, sql_responses: list[str]) -> None:
        self.embeddings = FakeEmbeddings()
        self.chat = SimpleNamespace(completions=FakeChatCompletions(sql_responses))


def _answer_settings(db_path: Path, tmp_path: Path, **overrides: object) -> Settings:
    base: dict[str, object] = dict(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        embedding_batch_size=2,
        retrieval_top_k=4,
        enable_result_shape_check=False,
        enable_schema_profiling=False,
    )
    base.update(overrides)
    return Settings(**base)


def _require_loadable_sqlite_vec() -> None:
    pytest.importorskip("sqlite_vec")
    import sqlite_vec

    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"sqlite-vec cannot load: {exc}")
    finally:
        conn.close()


def test_answer_question_happy_path(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path)
    client = FakeSqlClient(["SELECT CompanyName FROM Customers ORDER BY CompanyName"])
    result = answer_question(settings, client, "Which customers do we have?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.executed is True
    assert result.repaired is False
    assert result.columns == ("CompanyName",)
    assert ("Alfreds",) in result.rows


def test_answer_question_repairs_invalid_sql(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path)
    client = FakeSqlClient(
        [
            "SELECT MissingColumn FROM Customers",
            "SELECT CompanyName FROM Customers ORDER BY CompanyName",
        ]
    )
    result = answer_question(settings, client, "Which customers do we have?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.repaired is True
    assert result.validation_error is None
    assert result.repair_reason is not None
    assert result.repair_reason.startswith("validation failed:")
    assert result.sql == "SELECT CompanyName FROM Customers ORDER BY CompanyName"


def test_answer_question_repairs_result_shape(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path, enable_result_shape_check=True)
    client = FakeSqlClient(
        [
            "SELECT CustomerID, COUNT(OrderID) AS OrderCount FROM Orders GROUP BY CustomerID",
            "SELECT c.CompanyName, COUNT(o.OrderID) AS OrderCount "
            "FROM Customers c JOIN Orders o ON c.CustomerID = o.CustomerID "
            "GROUP BY c.CompanyName ORDER BY OrderCount DESC",
        ]
    )
    result = answer_question(settings, client, "Which customers placed the most orders?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.repaired is True
    assert result.repair_reason == "result shape mismatch"
    assert "CompanyName" in result.columns


def test_answer_question_cancelled_by_approval(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path, require_sql_approval=True)
    client = FakeSqlClient(["SELECT CompanyName FROM Customers"])
    seen: dict[str, str] = {}

    def deny(sql: str) -> bool:
        seen["sql"] = sql
        return False

    result = answer_question(  # type: ignore[arg-type]
        settings, client, "Which customers placed orders?", approval_callback=deny
    )
    assert result.cancelled is True
    assert result.executed is False
    assert result.success is False
    assert result.rows == ()
    assert seen["sql"] == "SELECT CompanyName FROM Customers"
    logs = read_query_logs(settings, 10)
    assert logs and logs[0]["cancelled"] == 1


def create_books_db(path: Path) -> None:
    """A deliberately non-Northwind schema, to prove database independence."""
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE authors (
                author_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                author_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                published_year INTEGER,
                FOREIGN KEY (author_id) REFERENCES authors(author_id)
            );
            INSERT INTO authors VALUES (1, 'Ursula K. Le Guin');
            INSERT INTO authors VALUES (2, 'Octavia Butler');
            INSERT INTO books VALUES (1, 1, 'A Wizard of Earthsea', 1968);
            INSERT INTO books VALUES (2, 2, 'Kindred', 1979);
            INSERT INTO books VALUES (3, 1, 'The Dispossessed', 1974);
            """
        )


def test_answer_question_works_on_non_northwind_schema(tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    db = tmp_path / "books.db"
    create_books_db(db)
    settings = _answer_settings(db, tmp_path)
    client = FakeSqlClient(
        [
            "SELECT b.title, a.name FROM books AS b "
            "JOIN authors AS a ON b.author_id = a.author_id ORDER BY b.title"
        ]
    )
    result = answer_question(settings, client, "List book titles and author names.")  # type: ignore[arg-type]
    assert result.success is True
    assert result.executed is True
    assert result.columns == ("title", "name")
    assert ("Kindred", "Octavia Butler") in result.rows


def test_default_metadata_path_is_path_based(tmp_path: Path) -> None:
    p1 = default_metadata_path_for_source(tmp_path / "shop.db")
    p2 = default_metadata_path_for_source(tmp_path / "copy" / "shop.db")
    assert p1 != p2  # same name, different dirs -> different metadata files
    assert p1.name.startswith("shop-")
    assert p1.name.endswith(".metadata.db")
    assert default_metadata_path_for_source(tmp_path / "shop.db") == p1  # deterministic


def test_translate_model_error_connection() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8000/v1/embeddings")
    err = translate_model_error(openai.APIConnectionError(request=request), "http://127.0.0.1:8000/v1")
    assert isinstance(err, AppError)
    assert "Could not reach the model server" in str(err)
    assert "127.0.0.1:8000" in str(err)


def test_translate_model_error_fallback() -> None:
    class FakeOpenAIError(openai.OpenAIError):
        pass

    err = translate_model_error(FakeOpenAIError("boom"), "http://127.0.0.1:8000/v1")
    assert isinstance(err, AppError)
    assert "Model request failed" in str(err)
    assert "127.0.0.1:8000" in str(err)


class _BoomEmbeddings:
    def create(self, model: str, input: list[str]) -> object:
        del model, input
        raise openai.APIConnectionError(
            request=httpx.Request("POST", "http://127.0.0.1:8000/v1/embeddings")
        )


class _BoomClient:
    base_url = "http://127.0.0.1:8000/v1"
    embeddings = _BoomEmbeddings()


def test_embed_texts_translates_connection_error() -> None:
    with pytest.raises(AppError) as excinfo:
        embed_texts(_BoomClient(), "model", ["x"], 2, instruction="i")  # type: ignore[arg-type]
    assert "Could not reach the model server" in str(excinfo.value)
    assert "127.0.0.1:8000" in str(excinfo.value)


# --- v2.2: total-count result shape ---------------------------------------------------


def test_total_count_expectation_detection() -> None:
    total = infer_shape_expectation("How many customers are there?")
    assert total.is_total_count is True
    assert total.entity_terms == ()  # the counted entity is aggregated, not a label column
    assert infer_shape_expectation("Count the invoices.").is_total_count is True
    assert infer_shape_expectation("What is the total number of customers?").is_total_count is True
    assert infer_shape_expectation("What is the number of tracks?").is_total_count is True
    # breakdown phrases turn a count into a grouped result, not a single total
    assert infer_shape_expectation("How many customers are there by country?").is_total_count is False
    assert infer_shape_expectation("Count orders per employee.").is_total_count is False
    # ranked questions are not total counts
    assert infer_shape_expectation("Which customers placed the most orders?").is_total_count is False


def test_shape_check_flags_grouped_total_count() -> None:
    check = check_result_shape(
        "How many customers are there?",
        "SELECT CustomerID, CompanyName, COUNT(*) AS CustomerCount "
        "FROM Customers GROUP BY CustomerID",
        ("CustomerID", "CompanyName", "CustomerCount"),
        (("ALFKI", "Alfreds", 1), ("SAVEA", "Save-a-lot Markets", 1)),
    )
    assert check.ok is False
    assert check.warning is not None
    assert "total count" in check.warning


def test_shape_check_allows_grouped_count_with_breakdown() -> None:
    check = check_result_shape(
        "How many customers are there by country?",
        "SELECT Country, COUNT(*) AS CustomerCount FROM Customers GROUP BY Country",
        ("Country", "CustomerCount"),
        (("Germany", 11), ("USA", 13)),
    )
    assert check.ok is True
    assert check.warning is None


def test_answer_question_repairs_total_count_shape(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path, enable_result_shape_check=True)
    client = FakeSqlClient(
        [
            "SELECT CustomerID, CompanyName, COUNT(*) AS CustomerCount "
            "FROM Customers GROUP BY CustomerID",
            "SELECT COUNT(*) AS CustomerCount FROM Customers",
        ]
    )
    result = answer_question(settings, client, "How many customers are there?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.repaired is True
    assert result.repair_reason == "result shape mismatch"
    assert result.columns == ("CustomerCount",)
    assert result.rows == ((2,),)


def test_prompts_include_total_count_rule() -> None:
    expectation = infer_shape_expectation("How many customers are there?")
    rule = "If the user asks for a total count, return a single COUNT(*) aggregate row."
    initial = build_sql_prompt("How many customers are there?", "SCHEMA", 50, expectation)
    validation = build_validation_repair_prompt(
        "How many customers are there?", "SCHEMA", "BAD", "err", 50, expectation
    )
    shape = build_shape_repair_prompt("How many customers are there?", "SCHEMA", "SQL", {}, {}, 50)
    assert rule in initial
    assert rule in validation
    assert rule in shape


# --- v2.3: schema profiling -----------------------------------------------------------


def test_schema_profile_includes_row_count_and_samples(db_path: Path, tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        enable_schema_profiling=True,
    )
    docs = schema_documents_for_index(settings, extract_schema(db_path))
    customers = next(doc for doc in docs if doc.table_name == "Customers")
    assert "Profile" in customers.content
    assert "Row count: 2" in customers.content
    # sample values appear somewhere in the document (no row-order assumption)
    assert "Alfreds" in customers.content


def test_schema_profile_truncates_long_text(tmp_path: Path) -> None:
    db = tmp_path / "long.db"
    long_value = "x" * 200
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE Notes (NoteID INTEGER PRIMARY KEY, Body TEXT)")
        conn.execute("INSERT INTO Notes VALUES (1, ?)", (long_value,))
    settings = Settings(
        omlx_api_key="test",
        db_path=db,
        metadata_db_path=tmp_path / "metadata.db",
        enable_schema_profiling=True,
        max_profile_text_length=20,
    )
    docs = schema_documents_for_index(settings, extract_schema(db))
    notes = next(doc for doc in docs if doc.table_name == "Notes")
    assert long_value not in notes.content
    assert "x" * 20 + "…" in notes.content


def test_schema_profile_skips_blob_columns(tmp_path: Path) -> None:
    db = tmp_path / "blobs.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE Files (FileID INTEGER PRIMARY KEY, Data BLOB, Label TEXT)")
        conn.execute("INSERT INTO Files VALUES (1, ?, 'first')", (b"\x00\x01\x02binary",))
    settings = Settings(
        omlx_api_key="test",
        db_path=db,
        metadata_db_path=tmp_path / "metadata.db",
        enable_schema_profiling=True,
    )
    docs = schema_documents_for_index(settings, extract_schema(db))
    files = next(doc for doc in docs if doc.table_name == "Files")
    assert "Data: binary data (not profiled)" in files.content
    assert "Data: samples" not in files.content  # the BLOB column is never sampled
    assert "first" in files.content  # the text column is still profiled


def test_disabling_schema_profiling_omits_profile(db_path: Path, tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        enable_schema_profiling=False,
    )
    docs = schema_documents_for_index(settings, extract_schema(db_path))
    assert all("Profile" not in doc.content for doc in docs)


# --- v2.4: view support ---------------------------------------------------------------


def create_view_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE authors (
                author_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                author_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                FOREIGN KEY (author_id) REFERENCES authors(author_id)
            );
            CREATE VIEW book_author_view AS
                SELECT b.title, a.name AS author_name
                FROM books b JOIN authors a ON b.author_id = a.author_id;
            INSERT INTO authors VALUES (1, 'Ursula K. Le Guin');
            INSERT INTO books VALUES (1, 1, 'A Wizard of Earthsea');
            """
        )


def test_extract_schema_includes_views(tmp_path: Path) -> None:
    db = tmp_path / "library.db"
    create_view_db(db)
    by_name = {table.name: table for table in extract_schema(db)}
    assert "book_author_view" in by_name
    assert by_name["book_author_view"].kind == "view"
    assert by_name["books"].kind == "table"


def test_view_schema_document_marks_view(tmp_path: Path) -> None:
    db = tmp_path / "library.db"
    create_view_db(db)
    view = next(table for table in extract_schema(db) if table.name == "book_author_view")
    content = build_schema_document(view).content
    assert "Object type: view" in content
    assert "View: book_author_view" in content


def test_index_schema_indexes_views(tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    db = tmp_path / "library.db"
    create_view_db(db)
    settings = Settings(
        omlx_api_key="test",
        db_path=db,
        metadata_db_path=tmp_path / "metadata.db",
        embedding_batch_size=2,
        enable_schema_profiling=False,
    )
    summary = index_schema(settings, FakeClient())  # type: ignore[arg-type]
    assert summary.table_count == 3
    with metadata_connection(settings.metadata_path) as conn:
        setup_metadata(conn)
        kinds = dict(
            conn.execute("SELECT object_name, object_type FROM schema_objects").fetchall()
        )
    assert kinds["book_author_view"] == "view"
    assert kinds["books"] == "table"


def test_validate_sql_allows_view_select(tmp_path: Path) -> None:
    db = tmp_path / "library.db"
    create_view_db(db)
    schema = extract_schema(db)
    validate_sql("SELECT title, author_name FROM book_author_view", schema)


def test_validate_sql_rejects_drop_view(tmp_path: Path) -> None:
    db = tmp_path / "library.db"
    create_view_db(db)
    schema = extract_schema(db)
    with pytest.raises(SqlValidationError):
        validate_sql("DROP VIEW book_author_view", schema)


# --- v3.0: deterministic result analysis ----------------------------------------------


def test_make_unique_column_names() -> None:
    assert make_unique_column_names(("count", "count", "x")) == ("count", "count__2", "x")


def test_result_to_arrow_table_basic() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    table = result_to_arrow_table(("id", "name"), ((1, "a"), (2, "b")))
    assert table.num_rows == 2
    assert table.column_names == ["id", "name"]


def test_result_to_arrow_table_mixed_types_fall_back_to_string() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    # SQLite is dynamically typed: a column may hold int + str; it must fall back to string
    # with None preserved as null (not the literal "None").
    table = result_to_arrow_table(("mixed",), ((1,), ("two",), (None,)))
    assert table.column(0).to_pylist() == ["1", "two", None]


def test_result_to_arrow_table_duplicate_names_do_not_crash() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    table = result_to_arrow_table(("count", "count"), ((1, 2),))
    assert table.num_rows == 1
    assert table.num_columns == 2


def test_profile_result_dataframe_computes_stats() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    profile = profile_result_dataframe(
        ("country", "total"),
        (("USA", 10), ("France", 20), ("USA", None)),
    )
    assert profile.row_count == 3
    assert profile.column_count == 2
    by_name = {stat.name: stat for stat in profile.columns}
    assert by_name["country"].distinct_count == 2
    assert by_name["country"].null_count == 0
    total = by_name["total"]
    assert total.null_count == 1
    assert total.minimum == 10
    assert total.maximum == 20
    assert total.mean == pytest.approx(15.0)


def test_profile_result_dataframe_handles_empty_result() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    profile = profile_result_dataframe(("CustomerID", "CompanyName"), ())
    assert profile.row_count == 0
    assert profile.column_count == 2
    for stat in profile.columns:
        assert stat.null_count == 0
        assert stat.distinct_count == 0


def test_dataframe_profile_text_caps_columns() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    columns = tuple(f"c{index}" for index in range(5))
    profile = profile_result_dataframe(columns, (tuple(range(5)),))
    text = dataframe_profile_text(profile, max_columns=2)
    assert "Rows profiled: 1" in text
    assert "... 3 more columns omitted" in text


def test_build_summary_payload_includes_profile_when_present() -> None:
    payload = build_summary_payload(
        "q", "SELECT 1", ("n",), ((1,),), False, analysis_text="Rows profiled: 1, Columns: 1"
    )
    assert "Deterministic result profile" in payload
    assert "do not invent rows or columns" in payload
    assert "Rows profiled: 1, Columns: 1" in payload


def test_build_summary_payload_omits_profile_when_absent() -> None:
    payload = build_summary_payload("q", "SELECT 1", ("n",), ((1,),), False)
    assert "Deterministic result profile" not in payload
    assert payload.rstrip().endswith("Summary:")


def _analysis_settings(db_path: Path, tmp_path: Path) -> Settings:
    return Settings(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        enable_dataframe_analysis=True,
        max_analysis_rows=10,
    )


def test_analyze_execution_reuses_untruncated_rows(db_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    settings = _analysis_settings(db_path, tmp_path)
    sql = "SELECT OrderID FROM Orders ORDER BY OrderID"
    execution = execute_readonly_query(db_path, sql, max_rows=10, timeout_ms=3000)
    assert execution.truncated is False
    profile, text = analyze_execution(settings, sql, execution)
    assert profile.row_count == 3
    assert profile.profiled_truncated is False
    assert "Rows profiled: 3" in text


def test_analyze_execution_refetches_when_truncated(db_path: Path, tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    settings = _analysis_settings(db_path, tmp_path)
    sql = "SELECT OrderID FROM Orders ORDER BY OrderID"
    execution = execute_readonly_query(db_path, sql, max_rows=2, timeout_ms=3000)
    assert execution.truncated is True and len(execution.rows) == 2
    profile, _ = analyze_execution(settings, sql, execution)
    assert profile.row_count == 3  # re-fetched beyond the display cap


def test_answer_question_populates_analysis_when_enabled(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    settings = _answer_settings(db_path, tmp_path, enable_dataframe_analysis=True)
    client = FakeSqlClient(["SELECT CompanyName FROM Customers ORDER BY CompanyName"])
    result = answer_question(settings, client, "Which customers do we have?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.analysis is not None
    assert result.analysis_error is None
    assert result.analysis_text and "Rows profiled:" in result.analysis_text


def test_answer_question_skips_analysis_when_disabled(db_path: Path, tmp_path: Path) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path)  # enable_dataframe_analysis defaults False
    client = FakeSqlClient(["SELECT CompanyName FROM Customers ORDER BY CompanyName"])
    result = answer_question(settings, client, "Which customers do we have?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.analysis is None
    assert result.analysis_text is None


def test_answer_question_analysis_failure_is_non_fatal(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_loadable_sqlite_vec()
    settings = _answer_settings(db_path, tmp_path, enable_dataframe_analysis=True)

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("core.analyze_execution", boom)
    client = FakeSqlClient(["SELECT CompanyName FROM Customers ORDER BY CompanyName"])
    result = answer_question(settings, client, "Which customers do we have?")  # type: ignore[arg-type]
    assert result.success is True
    assert result.analysis is None
    assert result.analysis_error is not None and "kaboom" in result.analysis_error


# --- v3.1: interactive result artifacts + colon commands -------------------------------


def _make_artifact(rows: tuple[tuple[object, ...], ...], **overrides: object) -> ResultArtifact:
    base: dict[str, object] = dict(
        artifact_id=1,
        question="q",
        sql="SELECT x FROM t",
        columns=("x",),
        rows=rows,
        truncated=False,
        analysis_text=None,
        created_at="2026-06-08T00:00:00+00:00",
    )
    base.update(overrides)
    return ResultArtifact(**base)  # type: ignore[arg-type]


def _make_analysis(**overrides: object) -> AnalysisArtifact:
    base: dict[str, object] = dict(
        analysis_id=1,
        source_artifact_id=1,
        recipe="profile",
        status="success",
        title="Profile Result",
        summary="Profiled 2 rows.",
        tables=(
            AnalysisArtifactTable(
                title="Column Profile",
                columns=("Column", "Type", "Rows"),
                rows=(("x", "numeric", 2),),
            ),
        ),
        metrics={"rows_used": 2, "score": 0.5},
        warnings=("Treat as directional.",),
        created_at="2026-06-08T00:01:00+00:00",
    )
    base.update(overrides)
    return AnalysisArtifact(**base)  # type: ignore[arg-type]


def test_parse_colon_command_variants() -> None:
    assert parse_colon_command(":head 5") == ("head", "5")
    assert parse_colon_command(":export csv") == ("export", "csv")
    assert parse_colon_command(":describe") == ("describe", "")
    assert parse_colon_command(":head    5") == ("head", "5")  # collapses extra spaces
    assert parse_colon_command(":workspace-info genre") == ("workspace-info", "genre")
    assert parse_colon_command(":delete-workspace genre") == ("delete-workspace", "genre")
    assert parse_colon_command("list customers") is None
    assert parse_colon_command(":") is None
    assert parse_colon_command("   ") is None



def test_parse_count_rules() -> None:
    assert parse_count("") == 10
    assert parse_count("5") == 5
    for bad in ("abc", "0", "-3"):
        with pytest.raises(AppError):
            parse_count(bad)


def test_make_result_artifact_copies_fields() -> None:
    result = AnswerResult(question="hello", retrieved_tables=[], expanded_tables=[])
    result.sql = "SELECT a FROM t"
    result.columns = ("a",)
    result.rows = ((1,),)
    result.truncated = True
    result.analysis_text = "Rows profiled: 1, Columns: 1"
    artifact = make_result_artifact(3, result)
    assert artifact.artifact_id == 3
    assert artifact.question == "hello"
    assert artifact.sql == "SELECT a FROM t"
    assert artifact.columns == ("a",)
    assert artifact.rows == ((1,),)
    assert artifact.truncated is True
    assert artifact.analysis_text == "Rows profiled: 1, Columns: 1"
    assert artifact.created_at


def test_make_result_artifact_requires_sql_and_columns() -> None:
    result = AnswerResult(question="q", retrieved_tables=[], expanded_tables=[])
    with pytest.raises(AppError):
        make_result_artifact(1, result)  # no SQL
    result.sql = "SELECT a FROM t"
    with pytest.raises(AppError):
        make_result_artifact(1, result)  # no columns


def test_artifact_preview_rows_head_tail() -> None:
    artifact = _make_artifact(tuple((index,) for index in range(5)))
    assert artifact_preview_rows(artifact, "head", 2) == ((0,), (1,))
    assert artifact_preview_rows(artifact, "tail", 2) == ((3,), (4,))
    assert artifact_preview_rows(artifact, "head", 99) == artifact.rows
    assert artifact_preview_rows(artifact, "tail", 1) == ((4,),)
    with pytest.raises(AppError):
        artifact_preview_rows(artifact, "middle", 2)


def test_export_artifact_csv_writes_sequential_sanitized_files(tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        db_path=tmp_path / "x.db",
        metadata_db_path=tmp_path / "m.db",
        output_dir=tmp_path / "outputs",
    )
    artifact = _make_artifact(
        ((1, "a"), (2, None), (3, b"\x00bin")),
        sql="SELECT id, name FROM t",
        columns=("id", "name"),
    )
    export = export_artifact_csv(settings, artifact, settings.output_path)
    assert export.path.name == "result_001.csv"
    assert export.truncated is False
    assert export.row_count == 3
    with export.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["id", "name"]
    assert rows[1] == ["1", "a"]
    assert rows[2] == ["2", ""]  # None -> empty cell
    assert rows[3] == ["3", "<binary>"]  # bytes -> <binary>
    second = export_artifact_csv(settings, artifact, settings.output_path)
    assert second.path.name == "result_002.csv"  # sequential, no overwrite


def test_export_artifact_csv_refetches_when_truncated(db_path: Path, tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        db_path=db_path,
        metadata_db_path=tmp_path / "m.db",
        output_dir=tmp_path / "outputs",
        max_analysis_rows=10,
    )
    sql = "SELECT OrderID FROM Orders ORDER BY OrderID"
    execution = execute_readonly_query(db_path, sql, max_rows=2, timeout_ms=3000)
    assert execution.truncated is True
    artifact = _make_artifact(
        execution.rows, sql=sql, columns=execution.columns, truncated=True
    )
    export = export_artifact_csv(settings, artifact, settings.output_path)
    assert export.row_count == 3  # re-fetched all 3 orders beyond the display cap
    assert export.truncated is False
    with export.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["OrderID"]
    assert len(rows) == 4  # header + 3 rows


def test_artifact_describe_text_reuses_analysis_text() -> None:
    artifact = _make_artifact(((1,),), analysis_text="Rows profiled: 1, Columns: 1")
    settings = Settings(omlx_api_key="test")
    assert artifact_describe_text(settings, artifact) == "Rows profiled: 1, Columns: 1"


def test_artifact_describe_text_computes_when_missing() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    artifact = _make_artifact(((10,), (20,)), sql="SELECT total FROM t", columns=("total",))
    settings = Settings(omlx_api_key="test")
    text = artifact_describe_text(settings, artifact)
    assert "Rows profiled: 2" in text


def test_verify_sqlite_database_accepts_view_only_database(tmp_path: Path) -> None:
    db = tmp_path / "views_only.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE VIEW answer_view AS SELECT 42 AS answer;")
    assert verify_sqlite_database(db) == 1


def test_verify_sqlite_database_rejects_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA user_version = 1")
    with pytest.raises(AppError, match="No user tables or views"):
        verify_sqlite_database(db)


# --- v3.2: deterministic chart artifacts -----------------------------------------------


def _chart_settings(tmp_path: Path) -> Settings:
    return Settings(
        omlx_api_key="test",
        db_path=tmp_path / "x.db",
        metadata_db_path=tmp_path / "m.db",
        output_dir=tmp_path / "outputs",
    )


def test_parse_key_value_args_parses_pairs() -> None:
    assert parse_key_value_args("x=Name y=Revenue") == {"x": "Name", "y": "Revenue"}
    assert parse_key_value_args("") == {}
    # keys lowercase; values keep their casing
    assert parse_key_value_args("X=Name") == {"x": "Name"}


def test_parse_key_value_args_rejects_token_without_equals() -> None:
    with pytest.raises(AppError):
        parse_key_value_args("x=Name Revenue")


def test_parse_key_value_args_rejects_duplicate_keys() -> None:
    with pytest.raises(AppError):
        parse_key_value_args("x=Name x=Other")


def test_parse_plot_command_args_parses_bar_and_hist() -> None:
    assert parse_plot_command_args("bar x=Name y=Revenue") == (
        "bar",
        {"x": "Name", "y": "Revenue"},
    )
    assert parse_plot_command_args("hist column=Revenue") == ("hist", {"column": "Revenue"})


def test_parse_plot_command_args_rejects_missing_type() -> None:
    with pytest.raises(AppError):
        parse_plot_command_args("")


def test_parse_plot_command_args_rejects_unsupported_type() -> None:
    with pytest.raises(AppError):
        parse_plot_command_args("pie x=Name y=Revenue")


def test_resolve_column_exact_and_case_insensitive() -> None:
    columns = ("Name", "Revenue")
    assert resolve_column(columns, "Name") == "Name"
    assert resolve_column(columns, "revenue") == "Revenue"


def test_resolve_column_unknown_raises() -> None:
    with pytest.raises(AppError):
        resolve_column(("Name", "Revenue"), "Missing")


def test_resolve_column_ambiguous_case_insensitive_raises() -> None:
    with pytest.raises(AppError):
        resolve_column(("Name", "name"), "NAME")


def test_chart_numeric_coercion_rules() -> None:
    assert _chart_numeric(1) == 1.0
    assert _chart_numeric(1.5) == 1.5
    assert _chart_numeric("2.0") == 2.0
    for missing in (True, False, None, "", "abc", float("nan"), float("inf")):
        assert _chart_numeric(missing) is None


def test_export_artifact_chart_rejects_empty_rows(tmp_path: Path) -> None:
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact((), columns=("Name", "Revenue"))
    with pytest.raises(AppError):
        export_artifact_chart(settings, artifact, "bar x=Name y=Revenue")


def test_export_artifact_chart_propagates_missing_viz(tmp_path: Path, monkeypatch) -> None:
    def _raise() -> object:
        raise AppError("Chart export needs the viz extra: uv sync --extra viz")

    monkeypatch.setattr(core, "_require_viz_libs", _raise)
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact(
        (("a", 1), ("b", 2)), columns=("Name", "Revenue"), sql="SELECT Name, Revenue FROM t"
    )
    with pytest.raises(AppError, match="viz extra"):
        export_artifact_chart(settings, artifact, "bar x=Name y=Revenue")


def test_export_artifact_chart_bar_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact(
        (("a", 10), ("b", 20), ("c", 30)),
        columns=("Name", "Revenue"),
        sql="SELECT Name, Revenue FROM t",
    )
    chart = export_artifact_chart(settings, artifact, "bar x=Name y=Revenue")
    assert chart.path.name == "chart_001.png"
    assert chart.path.exists()
    assert chart.chart_type == "bar"
    assert chart.row_count == 3


def test_export_artifact_chart_sequential_names(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact(
        (("a", 10), ("b", 20)),
        columns=("Name", "Revenue"),
        sql="SELECT Name, Revenue FROM t",
    )
    first = export_artifact_chart(settings, artifact, "bar x=Name y=Revenue")
    second = export_artifact_chart(settings, artifact, "bar x=Name y=Revenue")
    assert first.path.name == "chart_001.png"
    assert second.path.name == "chart_002.png"
    assert second.path.exists()


def test_export_artifact_chart_hist_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact(
        ((10,), (20,), (30,), (40,)),
        columns=("Revenue",),
        sql="SELECT Revenue FROM t",
    )
    chart = export_artifact_chart(settings, artifact, "hist column=Revenue")
    assert chart.path.exists()
    assert chart.chart_type == "hist"
    assert chart.x_column == "Revenue"
    assert chart.y_column is None
    assert chart.row_count == 4


def test_export_artifact_chart_rejects_non_numeric(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    settings = _chart_settings(tmp_path)
    artifact = _make_artifact(
        (("a", 10), ("b", 20)),
        columns=("Name", "Revenue"),
        sql="SELECT Name, Revenue FROM t",
    )
    with pytest.raises(AppError):
        export_artifact_chart(settings, artifact, "scatter x=Name y=Revenue")
    with pytest.raises(AppError):
        export_artifact_chart(settings, artifact, "hist column=Name")


# --- v3.3: natural-language artifact command router ------------------------------------

_ROUTER_COLUMNS = ("GenreName", "TotalRevenue", "TrackCount")


def test_route_describe_phrase() -> None:
    route = route_artifact_followup("describe this result", _ROUTER_COLUMNS)
    assert route is not None
    assert route.command == "describe"
    assert route.arg == ""


def test_route_head_with_and_without_count() -> None:
    five = route_artifact_followup("show first 5 rows", _ROUTER_COLUMNS)
    assert (five.command, five.arg) == ("head", "5")
    default = route_artifact_followup("head", _ROUTER_COLUMNS)
    assert (default.command, default.arg) == ("head", "10")


def test_route_tail_with_count() -> None:
    route = route_artifact_followup("show last 3 rows", _ROUTER_COLUMNS)
    assert (route.command, route.arg) == ("tail", "3")


def test_route_export_csv() -> None:
    route = route_artifact_followup("export to csv", _ROUTER_COLUMNS)
    assert (route.command, route.arg) == ("export", "csv")


def test_route_sql_columns_artifacts() -> None:
    assert route_artifact_followup("show sql", _ROUTER_COLUMNS).command == "sql"
    assert route_artifact_followup("show columns", _ROUTER_COLUMNS).command == "columns"
    assert route_artifact_followup("show artifacts", _ROUTER_COLUMNS).command == "artifacts"


def test_route_plot_bar_canonicalizes_columns() -> None:
    route = route_artifact_followup("plot bar x=genrename y=totalrevenue", _ROUTER_COLUMNS)
    assert route.command == "plot"
    assert route.arg == "bar x=GenreName y=TotalRevenue"


def test_route_histogram_of_column() -> None:
    route = route_artifact_followup("histogram of totalrevenue", _ROUTER_COLUMNS)
    assert route.command == "plot"
    assert route.arg == "hist column=TotalRevenue"


def test_route_explicit_chart_bad_column_raises() -> None:
    with pytest.raises(AppError):
        route_artifact_followup("histogram of missing", _ROUTER_COLUMNS)
    with pytest.raises(AppError):
        route_artifact_followup("plot bar x=missing y=TotalRevenue", _ROUTER_COLUMNS)


def test_route_ambiguous_chart_column_raises() -> None:
    # The router lowercases the token, so two case-variants that both differ from the
    # lowercased token by case ("name" matches neither "Name" nor "NAME" exactly) are ambiguous.
    columns = ("Name", "NAME")
    with pytest.raises(AppError):
        route_artifact_followup("histogram of name", columns)


def test_route_returns_none_for_vague_and_db_questions() -> None:
    for text in (
        "which genres generated the most revenue?",
        "how many customers are there?",
        "show sales by country",
        "top 5 genres by revenue",
        "plot revenue by country",
        "show results",
        "summarize this result",
    ):
        assert route_artifact_followup(text, _ROUTER_COLUMNS) is None


# --- v3.5: natural-language routing for artifact transformations -----------------------

_TRANSFORM_COLUMNS = ("GenreName", "TotalRevenue", "TrackCount", "Country", "Revenue")


def _route(text: str, columns: tuple[str, ...] = _TRANSFORM_COLUMNS) -> tuple[str, str]:
    route = route_artifact_followup(text, columns)
    assert route is not None
    return route.command, route.arg


def test_route_sort_followups() -> None:
    assert _route("sort by TotalRevenue descending") == ("sort", "column=TotalRevenue order=desc")
    assert _route("order by TrackCount") == ("sort", "column=TrackCount order=asc")
    assert route_artifact_followup("sort countries by revenue", _TRANSFORM_COLUMNS) is None
    with pytest.raises(AppError):
        route_artifact_followup("sort by Missing", _TRANSFORM_COLUMNS)


def test_route_select_followups() -> None:
    assert _route("select GenreName, TotalRevenue") == ("select", "columns=GenreName,TotalRevenue")
    assert _route("only columns GenreName,TrackCount") == ("select", "columns=GenreName,TrackCount")
    assert _route("only columns GenreName") == ("select", "columns=GenreName")
    with pytest.raises(AppError):
        route_artifact_followup("select Missing, TotalRevenue", _TRANSFORM_COLUMNS)
    # bare-select guard: no comma -> not routed (never raises), so SQL-like phrasing is safe
    assert route_artifact_followup("show only high revenue genres", _TRANSFORM_COLUMNS) is None
    assert route_artifact_followup("select GenreName", _TRANSFORM_COLUMNS) is None
    assert route_artifact_followup("select customers", _TRANSFORM_COLUMNS) is None


def test_route_filter_followups() -> None:
    assert _route("filter TotalRevenue greater than 100") == (
        "filter",
        "column=TotalRevenue op=gt value=100",
    )
    assert _route("where GenreName contains Rock") == (
        "filter",
        "column=GenreName op=contains value=Rock",
    )
    assert _route("keep rows where TrackCount at least 100") == (
        "filter",
        "column=TrackCount op=gte value=100",
    )
    with pytest.raises(AppError):
        route_artifact_followup("filter Missing greater than 100", _TRANSFORM_COLUMNS)
    # value has spaces -> not routed; non-filter prose starting with "where" -> not routed
    assert (
        route_artifact_followup("where GenreName contains Alternative Rock", _TRANSFORM_COLUMNS)
        is None
    )
    assert route_artifact_followup("where are the customers from USA", _TRANSFORM_COLUMNS) is None


def test_route_groupby_followups() -> None:
    assert _route("count by Country") == ("groupby", "by=Country agg=count")
    assert _route("group by Country sum Revenue") == (
        "groupby",
        "by=Country metric=Revenue agg=sum",
    )
    assert _route("group by Country average Revenue") == (
        "groupby",
        "by=Country metric=Revenue agg=mean",
    )
    with pytest.raises(AppError):
        route_artifact_followup("group by Missing sum Revenue", _TRANSFORM_COLUMNS)
    assert route_artifact_followup("summarize revenue by country", _TRANSFORM_COLUMNS) is None
    assert route_artifact_followup("sales by country", _TRANSFORM_COLUMNS) is None


def test_route_transform_ambiguous_column_raises() -> None:
    with pytest.raises(AppError):
        route_artifact_followup("sort by name", ("Name", "NAME"))


# --- v3.6: persistent artifact workspace -----------------------------------------------


def _workspace_settings(tmp_path: Path) -> Settings:
    return Settings(
        omlx_api_key="test",
        db_path=tmp_path / "x.db",
        metadata_db_path=tmp_path / "m.db",
        output_dir=tmp_path / "outputs",
    )


def _genre_rows_artifact(**overrides: object) -> ResultArtifact:
    return _make_artifact(
        (("Rock", 826.65, 835), ("Latin", 382.14, 579)),
        sql="SELECT GenreName, TotalRevenue, TrackCount FROM t",
        columns=("GenreName", "TotalRevenue", "TrackCount"),
        **overrides,
    )


def test_sanitize_workspace_name_rules() -> None:
    assert sanitize_workspace_name("my analysis!") == "my_analysis_"
    assert sanitize_workspace_name("abc-123_DEF") == "abc-123_DEF"
    for bad in ("", "   ", "!!!"):
        with pytest.raises(AppError):
            sanitize_workspace_name(bad)


def test_save_workspace_requires_artifacts(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        save_artifact_workspace(_workspace_settings(tmp_path), [])


def test_save_workspace_writes_files_and_manifest(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    artifact = _genre_rows_artifact(analysis_text="Rows profiled: 2, Columns: 3")
    result = save_artifact_workspace(settings, [artifact], name="genre_revenue")
    assert result.artifact_count == 1
    assert (result.path / "manifest.json").is_file()
    assert (result.path / "artifact_001.csv").is_file()
    assert (result.path / "artifact_001.sql").is_file()
    assert (result.path / "artifact_001_profile.txt").is_file()

    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1
    assert manifest["artifact_count"] == 1
    assert "app_version" in manifest and "created_at" in manifest
    entry = manifest["artifacts"][0]
    assert entry["columns"] == ["GenreName", "TotalRevenue", "TrackCount"]
    assert entry["row_count"] == 2
    assert entry["truncated"] is False
    assert entry["profile_file"] == "artifact_001_profile.txt"


def test_save_workspace_without_analysis_skips_profile(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    result = save_artifact_workspace(settings, [_genre_rows_artifact()])
    assert not (result.path / "artifact_001_profile.txt").exists()
    manifest = json.loads((result.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["profile_file"] is None


def test_save_workspace_creates_distinct_paths(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    first = save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")
    second = save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")
    assert first.path != second.path


def test_list_saved_workspaces_ignores_dirs_without_manifest(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    saved = save_artifact_workspace(settings, [_genre_rows_artifact()], name="alpha")
    (settings.output_path / "workspaces" / "not_a_workspace").mkdir()
    listed = list_saved_workspaces(settings)
    assert saved.path in listed
    assert all(path.name != "not_a_workspace" for path in listed)


def test_load_workspace_round_trips(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    source = _genre_rows_artifact(truncated=True)
    saved = save_artifact_workspace(settings, [source], name="genre_revenue")

    loaded = load_artifact_workspace(settings, saved.path.name)
    assert len(loaded.artifacts) == 1
    restored = loaded.artifacts[0]
    assert restored.columns == ("GenreName", "TotalRevenue", "TrackCount")
    # CSV cannot preserve types: every loaded cell is a string
    assert restored.rows[0] == ("Rock", "826.65", "835")
    assert all(isinstance(value, str) for row in restored.rows for value in row)
    assert restored.sql == source.sql
    assert restored.question == source.question
    assert restored.truncated is True
    assert restored.created_at == source.created_at
    assert restored.artifact_id == source.artifact_id


def test_load_workspace_rejects_csv_row_width_mismatch(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    saved = save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")

    # Corrupt the stored CSV with a row narrower than the header
    with (saved.path / "artifact_001.csv").open("a", newline="", encoding="utf-8") as handle:
        handle.write("Rock,1\n")

    with pytest.raises(AppError, match="row width does not match header"):
        load_artifact_workspace(settings, saved.path.name)


def test_save_workspace_persists_linked_analysis_artifacts(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    artifact = _genre_rows_artifact()
    linked = _make_analysis(source_artifact_id=artifact.artifact_id)
    orphan = _make_analysis(analysis_id=2, source_artifact_id=999)

    saved = save_artifact_workspace(
        settings,
        [artifact],
        name="genre_revenue",
        analyses=[linked, orphan],
    )

    assert saved.analysis_count == 1
    assert (saved.path / "analysis_001.json").is_file()
    assert (saved.path / "analysis_001.md").is_file()
    assert not (saved.path / "analysis_002.json").exists()

    manifest = json.loads((saved.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis_count"] == 1
    analysis_entry = manifest["analyses"][0]
    assert analysis_entry["analysis_id"] == linked.analysis_id
    assert analysis_entry["source_artifact_id"] == artifact.artifact_id
    assert analysis_entry["json_file"] == "analysis_001.json"
    assert analysis_entry["markdown_file"] == "analysis_001.md"

    loaded = load_artifact_workspace(settings, saved.path.name)
    assert len(loaded.analyses) == 1
    restored = loaded.analyses[0]
    assert restored.analysis_id == linked.analysis_id
    assert restored.source_artifact_id == artifact.artifact_id
    assert restored.recipe == "profile"
    assert restored.status == "success"
    assert restored.tables[0].rows == (("x", "numeric", 2),)
    assert restored.metrics["rows_used"] == 2
    assert restored.warnings == ("Treat as directional.",)


def test_load_workspace_unique_prefix(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    save_artifact_workspace(settings, [_genre_rows_artifact()], name="uniquename")
    loaded = load_artifact_workspace(settings, "uniquename")
    assert len(loaded.artifacts) == 1


def test_load_workspace_ambiguous_prefix_raises(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")
    save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")
    with pytest.raises(AppError, match="Multiple workspaces match"):
        load_artifact_workspace(settings, "genre_revenue")


def test_load_workspace_rejects_unknown_traversal_and_absolute(tmp_path: Path) -> None:
    settings = _workspace_settings(tmp_path)
    save_artifact_workspace(settings, [_genre_rows_artifact()], name="genre_revenue")
    with pytest.raises(AppError):
        load_artifact_workspace(settings, "does_not_exist")
    with pytest.raises(AppError):
        load_artifact_workspace(settings, "../outside")
    with pytest.raises(AppError):
        load_artifact_workspace(settings, str(tmp_path))


# --- v3.4: controlled artifact transformations -----------------------------------------


def _genre_artifact(**overrides: object) -> ResultArtifact:
    return _make_artifact(
        (
            ("Rock", 826.65, 835),
            ("Latin", 382.14, 579),
            ("Jazz", 79.2, 130),
            ("Empty", None, 0),
        ),
        sql="SELECT g.Name AS GenreName, SUM(x) AS TotalRevenue, COUNT(*) AS TrackCount FROM t",
        columns=("GenreName", "TotalRevenue", "TrackCount"),
        **overrides,
    )


def _country_artifact(**overrides: object) -> ResultArtifact:
    return _make_artifact(
        (("USA", 10), ("USA", 15), ("India", 7), ("India", None)),
        sql="SELECT Country, Revenue FROM t",
        columns=("Country", "Revenue"),
        **overrides,
    )


def test_parse_transform_args_rules() -> None:
    assert parse_transform_args("column=TotalRevenue order=desc") == {
        "column": "TotalRevenue",
        "order": "desc",
    }
    assert parse_transform_args("") == {}
    with pytest.raises(AppError):
        parse_transform_args("column")
    with pytest.raises(AppError):
        parse_transform_args("column=A column=B")


def test_compare_values_numeric_and_contains() -> None:
    assert _compare_values(100, "eq", "100.0") is True
    assert _compare_values("Rock", "contains", "ROCK") is True
    assert _compare_values("Rock", "gt", "100") is False  # non-numeric cell


def test_transform_sort_desc_and_asc_put_missing_last() -> None:
    artifact = _genre_artifact()
    desc = transform_artifact_sort(artifact, "column=TotalRevenue order=desc", 2).artifact
    assert [row[0] for row in desc.rows] == ["Rock", "Latin", "Jazz", "Empty"]
    asc = transform_artifact_sort(artifact, "column=TotalRevenue order=asc", 2).artifact
    assert [row[0] for row in asc.rows] == ["Jazz", "Latin", "Rock", "Empty"]


def test_transform_sort_rejects_bad_column_and_order() -> None:
    artifact = _genre_artifact()
    with pytest.raises(AppError):
        transform_artifact_sort(artifact, "column=Missing", 2)
    with pytest.raises(AppError):
        transform_artifact_sort(artifact, "column=TotalRevenue order=sideways", 2)


def test_transform_select_keeps_requested_columns_in_order() -> None:
    artifact = _genre_artifact()
    result = transform_artifact_select(artifact, "columns=GenreName,TotalRevenue", 2).artifact
    assert result.columns == ("GenreName", "TotalRevenue")
    assert result.rows[0] == ("Rock", 826.65)
    with pytest.raises(AppError):
        transform_artifact_select(artifact, "columns=GenreName,Missing", 2)


def test_transform_filter_gt_and_contains() -> None:
    artifact = _genre_artifact()
    gt = transform_artifact_filter(artifact, "column=TotalRevenue op=gt value=100", 2).artifact
    assert [row[0] for row in gt.rows] == ["Rock", "Latin"]
    contains = transform_artifact_filter(
        artifact, "column=GenreName op=contains value=rock", 2
    ).artifact
    assert [row[0] for row in contains.rows] == ["Rock"]


def test_transform_filter_rejects_bad_op_missing_and_empty_value() -> None:
    artifact = _genre_artifact()
    with pytest.raises(AppError):
        transform_artifact_filter(artifact, "column=TotalRevenue op=between value=1", 2)
    with pytest.raises(AppError):
        transform_artifact_filter(artifact, "column=TotalRevenue op=gt", 2)
    with pytest.raises(AppError):
        transform_artifact_filter(artifact, "column=GenreName op=contains value=", 2)


def test_transform_groupby_sum_mean_and_count() -> None:
    artifact = _country_artifact()
    summed = transform_artifact_groupby(artifact, "by=Country metric=Revenue agg=sum", 2).artifact
    assert summed.columns == ("Country", "sum_Revenue")
    assert dict(summed.rows) == {"India": 7, "USA": 25}
    meaned = transform_artifact_groupby(artifact, "by=Country metric=Revenue agg=mean", 2).artifact
    assert dict(meaned.rows) == {"India": 7.0, "USA": 12.5}
    counted = transform_artifact_groupby(artifact, "by=Country agg=count", 2).artifact
    assert counted.columns == ("Country", "count")
    assert dict(counted.rows) == {"India": 2, "USA": 2}
    non_null = transform_artifact_groupby(
        artifact, "by=Country metric=Revenue agg=count", 2
    ).artifact
    assert dict(non_null.rows) == {"India": 1, "USA": 2}


def test_transform_groupby_empty_numeric_group_is_none() -> None:
    artifact = _make_artifact(
        (("India", None),), sql="SELECT Country, Revenue FROM t", columns=("Country", "Revenue")
    )
    result = transform_artifact_groupby(artifact, "by=Country metric=Revenue agg=sum", 2).artifact
    assert result.rows == (("India", None),)


def test_transform_groupby_rejects_bad_agg_and_missing_metric() -> None:
    artifact = _country_artifact()
    with pytest.raises(AppError):
        transform_artifact_groupby(artifact, "by=Country metric=Revenue agg=median", 2)
    with pytest.raises(AppError):
        transform_artifact_groupby(artifact, "by=Country agg=sum", 2)


def test_transform_artifact_dispatch_and_unknown_op() -> None:
    artifact = _genre_artifact()
    result = transform_artifact(artifact, "sort", "column=TotalRevenue order=desc", 2)
    assert isinstance(result, ArtifactTransformResult)
    assert result.artifact.rows[0][0] == "Rock"
    with pytest.raises(AppError):
        transform_artifact(artifact, "pivot", "column=TotalRevenue", 2)


def test_transform_metadata_and_truncated_semantics() -> None:
    artifact = _genre_artifact(truncated=True)
    # sort/select preserve truncated
    sorted_result = transform_artifact_sort(artifact, "column=TotalRevenue", 7).artifact
    assert sorted_result.artifact_id == 7
    assert sorted_result.analysis_text is None
    assert sorted_result.sql == artifact.sql
    assert sorted_result.created_at
    assert sorted_result.truncated is True
    selected = transform_artifact_select(artifact, "columns=GenreName", 3).artifact
    assert selected.truncated is True
    # filter/groupby set truncated False but record the truncated source in the question
    filtered = transform_artifact_filter(artifact, "column=TotalRevenue op=gt value=0", 4).artifact
    assert filtered.truncated is False
    assert "truncated artifact" in filtered.question
    grouped = transform_artifact_groupby(artifact, "by=GenreName agg=count", 5).artifact
    assert grouped.truncated is False
    assert "truncated artifact" in grouped.question


# --- v3.7: report export tests --------------------------------------------------------

def test_markdown_report_rendering() -> None:
    # empty artifacts raises AppError
    with pytest.raises(AppError, match=r"No artifacts to report\."):
        core.render_artifact_report_markdown([])

    artifact = _make_artifact(
        rows=((1, "Alice|Bob", None), (2, b"binary_data", "active")),
        columns=("id", "name", "status"),
        question="Who is Alice?",
        sql="SELECT * FROM users",
        analysis_text="Some profile description here.",
    )

    report = core.render_artifact_report_markdown([artifact], title="Custom Report Title")

    # report includes title
    assert "# Custom Report Title" in report
    # report includes question
    assert "Who is Alice?" in report
    # report includes SQL fenced block
    assert "### SQL" in report
    assert "```sql\nSELECT * FROM users\n```" in report
    # report includes columns
    assert "id, name, status" in report
    # report includes preview rows and pipes in cell values are escaped, None is empty, bytes is <binary>
    assert "id | name | status" in report
    assert "1 | Alice\\|Bob | " in report
    assert "2 | <binary> | active" in report
    # analysis text appears when present
    assert "### Analysis" in report
    assert "Some profile description here." in report
    # stored rows warning is present
    assert "Report uses stored artifact rows and analysis digests only. No SQL was re-run." in report


def test_markdown_report_includes_linked_analysis_artifacts() -> None:
    artifact = _make_artifact(
        rows=((1,),),
        columns=("x",),
        artifact_id=1,
        question="Q1",
    )
    linked = _make_analysis(source_artifact_id=1)
    unrelated = _make_analysis(analysis_id=2, source_artifact_id=999, title="Unrelated")

    report = core.render_artifact_report_markdown(
        [artifact],
        analyses=[linked, unrelated],
    )

    assert "- **Analyses**: 1" in report
    assert "### Saved Analyses" in report
    assert "#### Analysis #1: Profile Result" in report
    assert "| Field | Value |" in report
    assert "| rows_used | 2 |" in report
    assert "| x | numeric | 2 |" in report
    assert "Treat as directional." in report
    assert "Unrelated" not in report


def test_markdown_report_rendering_backticks_fence() -> None:
    # SQL containing triple backticks
    artifact = _make_artifact(
        rows=((1,),),
        columns=("col",),
        question="Q?",
        sql="SELECT '```' AS val",
        analysis_text="Profile containing ``` and ````",
    )
    report = core.render_artifact_report_markdown([artifact])
    assert "````sql\nSELECT '```' AS val\n````" in report
    assert "`````\nProfile containing ``` and ````\n`````" in report

def test_markdown_report_preview_cap_note() -> None:
    # Preview row cap note appears when rows exceed cap
    artifact = _make_artifact(
        rows=tuple((i,) for i in range(15)),
        columns=("col",),
    )
    report = core.render_artifact_report_markdown([artifact], max_preview_rows=10)
    assert "Showing first 10 of 15 stored rows." in report

def test_html_report_rendering() -> None:
    # empty artifacts raises AppError
    with pytest.raises(AppError, match=r"No artifacts to report\."):
        core.render_artifact_report_html([])

    artifact = _make_artifact(
        rows=((1, "Alice & Bob <script>", None),),
        columns=("id", "name <script>"),
        question="Is <this> a test?",
        sql="SELECT 1",
        analysis_text="Test profile <info>",
    )

    report = core.render_artifact_report_html([artifact], title="HTML <title>")
    
    # HTML includes escaped question and values
    assert "Is &lt;this&gt; a test?" in report
    assert "id" in report
    assert "name &lt;script&gt;" in report
    assert "Alice &amp; Bob &lt;script&gt;" in report
    # HTML includes escaped title
    assert "HTML &lt;title&gt;" in report
    # SQL appears inside <pre><code>
    assert "<pre><code>SELECT 1</code></pre>" in report
    # analysis text appears when present
    assert "<h3>Analysis</h3>" in report
    assert "<pre><code>Test profile &lt;info&gt;</code></pre>" in report
    # stored rows warning is present
    assert "Report uses stored artifact rows and analysis digests only. No SQL was re-run." in report


def test_html_report_includes_linked_analysis_artifacts() -> None:
    artifact = _make_artifact(
        rows=((1,),),
        columns=("x",),
        artifact_id=1,
        question="Q1",
    )
    analysis = _make_analysis(title="Profile <Result>", source_artifact_id=1)

    report = core.render_artifact_report_html([artifact], analyses=[analysis])

    assert "<strong>Analyses:</strong> 1" in report
    assert "Saved Analyses" in report
    assert "Analysis #1: Profile &lt;Result&gt;" in report
    assert "<td>rows_used</td><td>2</td>" in report
    assert "<td>x</td>" in report


def test_html_report_preview_cap_note() -> None:
    # preview row cap note appears
    artifact = _make_artifact(
        rows=tuple((i,) for i in range(60)),
        columns=("col",),
    )
    report = core.render_artifact_report_html([artifact], max_preview_rows=50)
    assert "Showing first 50 of 60 stored rows." in report

def test_export_artifact_report_behavior(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    
    art1 = _make_artifact(
        rows=((1,),),
        columns=("col",),
        question="Q1",
        artifact_id=1,
    )
    art2 = _make_artifact(
        rows=((2,),),
        columns=("col",),
        question="Q2",
        artifact_id=2,
    )
    artifacts = [art1, art2]

    # report_format="md" writes .md
    res1 = core.export_artifact_report(settings, artifacts, report_format="md")
    assert res1.path.suffix == ".md"
    assert res1.artifact_count == 1
    assert res1.format == "markdown"
    assert res1.path.read_text(encoding="utf-8").count("## Artifact #") == 1

    # report_format="markdown" writes .md
    res2 = core.export_artifact_report(settings, artifacts, report_format="markdown")
    assert res2.path.suffix == ".md"
    assert res2.format == "markdown"

    # report_format="html" writes .html
    res3 = core.export_artifact_report(settings, artifacts, report_format="html")
    assert res3.path.suffix == ".html"
    assert res3.format == "html"
    assert res3.path.read_text(encoding="utf-8").count("<section>") == 1

    # unsupported format raises AppError
    with pytest.raises(AppError, match="Unsupported report format"):
        core.export_artifact_report(settings, artifacts, report_format="pdf")

    # include_all=True exports all artifacts
    res_all = core.export_artifact_report(settings, artifacts, report_format="md", include_all=True)
    assert res_all.artifact_count == 2
    assert res_all.path.read_text(encoding="utf-8").count("## Artifact #") == 2

    analyses = [
        _make_analysis(analysis_id=1, source_artifact_id=1, title="Analysis for Q1"),
        _make_analysis(analysis_id=2, source_artifact_id=2, title="Analysis for Q2"),
    ]
    res_latest_analysis = core.export_artifact_report(
        settings,
        artifacts,
        report_format="md",
        analyses=analyses,
    )
    latest_content = res_latest_analysis.path.read_text(encoding="utf-8")
    assert res_latest_analysis.analysis_count == 1
    assert "Analysis for Q2" in latest_content
    assert "Analysis for Q1" not in latest_content

    res_all_analysis = core.export_artifact_report(
        settings,
        artifacts,
        report_format="md",
        include_all=True,
        analyses=analyses,
    )
    assert res_all_analysis.analysis_count == 2

    # two exports create distinct file paths
    res_a = core.export_artifact_report(settings, artifacts, report_format="md")
    res_b = core.export_artifact_report(settings, artifacts, report_format="md")
    assert res_a.path != res_b.path


def test_export_workspace_report(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)

    art1 = _make_artifact(
        rows=((1,),),
        columns=("col1",),
        question="Q1",
        artifact_id=1,
    )
    art2 = _make_artifact(
        rows=((2,),),
        columns=("col2",),
        question="Q2",
        artifact_id=2,
    )
    artifacts = [art1, art2]

    analysis = _make_analysis(source_artifact_id=2)

    # Save to a workspace first
    save_artifact_workspace(settings, artifacts, name="test_ws", analyses=[analysis])

    # Export workspace report as Markdown
    res = core.export_workspace_report(settings, "test_ws", report_format="md")
    assert res.path.suffix == ".md"
    assert res.artifact_count == 2
    assert res.analysis_count == 1
    assert res.format == "markdown"

    report_content = res.path.read_text(encoding="utf-8")
    assert "## Artifact #1" in report_content
    assert "## Artifact #2" in report_content
    assert "Q1" in report_content
    assert "Q2" in report_content
    assert "#### Analysis #1: Profile Result" in report_content

    # Unknown workspace raises AppError
    with pytest.raises(AppError, match="No workspace found"):
        core.export_workspace_report(settings, "nonexistent", report_format="md")


def test_export_workspace_report_ambiguous_prefix(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    art = _make_artifact(rows=((1,),), columns=("col",))

    # Save two workspaces with same name stem
    save_artifact_workspace(settings, [art], name="genre_revenue_a")
    save_artifact_workspace(settings, [art], name="genre_revenue_b")

    # Call export_workspace_report with prefix -> raises AppError("Multiple workspaces match ...")
    with pytest.raises(AppError, match="Multiple workspaces match"):
        core.export_workspace_report(settings, "genre_revenue", report_format="md")


def test_inspect_workspace(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    art1 = _make_artifact(
        rows=((1,), (2,)),
        columns=("col1",),
        question="Q1",
        artifact_id=1,
    )
    art2 = _make_artifact(
        rows=((3,), (4,), (5,)),
        columns=("col2",),
        question="Q2",
        artifact_id=2,
    )
    
    # Save a workspace
    saved = save_artifact_workspace(settings, [art1, art2], name="test_inspect")
    
    # inspect it
    info = inspect_workspace(settings, "test_inspect")
    assert info.name.startswith("test_inspect_")
    assert info.path == saved.path
    assert info.artifact_count == 2
    assert info.row_count == 5
    assert info.created_at is not None
    assert "manifest.json" in info.files
    assert "artifact_001.csv" in info.files
    assert "artifact_001.sql" in info.files
    assert "artifact_002.csv" in info.files
    assert "artifact_002.sql" in info.files
    
    # works with unique prefix
    prefix_info = inspect_workspace(settings, "test_ins")
    assert prefix_info.name == info.name

    # unknown target raises AppError
    with pytest.raises(AppError, match="No workspace found"):
        inspect_workspace(settings, "nonexistent")

    # Save another one to make prefix ambiguous
    save_artifact_workspace(settings, [art1], name="test_inspect_other")
    with pytest.raises(AppError, match="Multiple workspaces match"):
        inspect_workspace(settings, "test_ins")


def test_delete_workspace(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    art = _make_artifact(rows=((1,),), columns=("col",))
    
    # Save two workspaces
    saved1 = save_artifact_workspace(settings, [art], name="genre_revenue_a")
    saved2 = save_artifact_workspace(settings, [art], name="genre_revenue_b")
    
    # delete_workspace works with unique prefix
    res = delete_workspace(settings, "genre_revenue_a")
    assert res.path == saved1.path
    assert res.name == saved1.path.name
    
    assert not saved1.path.exists()
    assert saved2.path.exists()  # sibling workspace remains untouched
    
    # delete_workspace unknown target raises AppError
    with pytest.raises(AppError, match="No workspace found"):
        delete_workspace(settings, "nonexistent")
        
    # delete_workspace ambiguous prefix raises AppError
    save_artifact_workspace(settings, [art], name="genre_revenue_a_new")
    # now we have genre_revenue_a_new and genre_revenue_b, wait, a unique prefix for "genre_revenue" is ambiguous
    with pytest.raises(AppError, match="Multiple workspaces match"):
        delete_workspace(settings, "genre_revenue")

    # delete_workspace("../outside") raises AppError
    with pytest.raises(AppError, match="Workspace must be a bare name"):
        delete_workspace(settings, "../outside")
        
    # delete_workspace(str(tmp_path)) absolute path raises AppError
    with pytest.raises(AppError, match="Workspace must be a name under the workspaces directory, not an absolute path"):
        delete_workspace(settings, str(tmp_path))


def test_inspect_workspace_malformed_manifest(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    art = _make_artifact(rows=((1,),), columns=("col",))
    saved = save_artifact_workspace(settings, [art], name="malformed_test")
    
    # Write invalid JSON to manifest.json
    (saved.path / "manifest.json").write_text("invalid json {", encoding="utf-8")
    
    with pytest.raises(AppError, match="Could not read workspace manifest"):
        inspect_workspace(settings, "malformed_test")


def test_inspect_workspace_missing_artifacts_list(tmp_path: Path) -> None:
    settings = SimpleNamespace(output_path=tmp_path)
    art = _make_artifact(rows=((1,),), columns=("col",))
    saved = save_artifact_workspace(settings, [art], name="bad_artifacts_test")
    
    # Write dict with artifacts not being list
    (saved.path / "manifest.json").write_text('{"artifacts": "not a list"}', encoding="utf-8")
    
    with pytest.raises(AppError, match="Invalid workspace manifest: artifacts must be a list"):
        inspect_workspace(settings, "bad_artifacts_test")


def test_resolve_workspace_target_validation(tmp_path: Path) -> None:
    workspaces_dir = tmp_path / "workspaces"
    workspaces_dir.mkdir()
    
    # Empty target
    with pytest.raises(AppError, match=r"Usage: :load <workspace>\."):
        core.resolve_workspace_target(workspaces_dir, "")
        
    # Absolute path
    with pytest.raises(AppError, match="Workspace must be a name under the workspaces directory, not an absolute path"):
        core.resolve_workspace_target(workspaces_dir, "/absolute/path")
        
    # Path separators or traversal
    with pytest.raises(AppError, match="Workspace must be a bare name"):
        core.resolve_workspace_target(workspaces_dir, "../traversal")
    with pytest.raises(AppError, match="Workspace must be a bare name"):
        core.resolve_workspace_target(workspaces_dir, "sub/folder")
    with pytest.raises(AppError, match="Workspace must be a bare name"):
        core.resolve_workspace_target(workspaces_dir, "sub\\folder")
        
    # No workspaces exist at all
    with pytest.raises(AppError, match="No workspace found"):
        core.resolve_workspace_target(workspaces_dir, "missing")
        
    # Create two workspaces
    ws1 = workspaces_dir / "genre_revenue_a"
    ws1.mkdir()
    (ws1 / "manifest.json").write_text("{}", encoding="utf-8")
    ws2 = workspaces_dir / "genre_revenue_b"
    ws2.mkdir()
    (ws2 / "manifest.json").write_text("{}", encoding="utf-8")
    
    # Ambiguous prefix
    with pytest.raises(AppError, match="Multiple workspaces match"):
        core.resolve_workspace_target(workspaces_dir, "genre_revenue")
        
    # Exact match works
    assert core.resolve_workspace_target(workspaces_dir, "genre_revenue_a") == ws1
