from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import (
    AnswerResult,
    IndexStateError,
    SQL_REPAIR_RULES,
    SqlValidationError,
    answer_question,
    build_shape_repair_prompt,
    build_sql_prompt,
    build_validation_repair_prompt,
    check_result_shape,
    deterministic_summary,
    execute_readonly_query,
    extract_schema,
    extract_sql,
    get_kv,
    infer_shape_expectation,
    index_schema,
    log_query,
    metadata_connection,
    quote_guidance,
    quote_identifier,
    read_query_logs,
    retrieve_schema,
    schema_fingerprint,
    set_kv,
    setup_metadata,
    validate_sql,
)
from config import Settings


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
        northwind_db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        embedding_batch_size=2,
        retrieval_top_k=2,
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


def test_read_query_logs_returns_dict_rows_with_expected_keys(tmp_path: Path) -> None:
    settings = Settings(
        omlx_api_key="test",
        northwind_db_path=tmp_path / "northwind.db",
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
        northwind_db_path=db_path,
        metadata_db_path=tmp_path / "metadata.db",
        embedding_batch_size=2,
        retrieval_top_k=4,
        enable_result_shape_check=False,
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
    assert result.validation_error is not None
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
