from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

import core
from core import (
    AnswerResult,
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
    RoutedArtifactCommand,
    parse_colon_command,
    parse_count,
    parse_key_value_args,
    parse_plot_command_args,
    resolve_column,
    route_artifact_followup,
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


def test_parse_colon_command_variants() -> None:
    assert parse_colon_command(":head 5") == ("head", "5")
    assert parse_colon_command(":export csv") == ("export", "csv")
    assert parse_colon_command(":describe") == ("describe", "")
    assert parse_colon_command(":head    5") == ("head", "5")  # collapses extra spaces
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
