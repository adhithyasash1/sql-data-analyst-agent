from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import openai
import sqlglot
from openai import OpenAI
from sqlglot import exp

from config import Settings

METADATA_SCHEMA_VERSION = "1"
SCHEMA_DOCUMENT_VERSION = "2"
NORTHWIND_DOWNLOAD_SHA = "4f56e7f5906dfd23b25244c5bfe8fb5da6402efd"
NORTHWIND_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/"
    f"{NORTHWIND_DOWNLOAD_SHA}/dist/northwind.db"
)
NORTHWIND_EXPECTED_TABLES = {"Customers", "Orders", "Order Details", "Products"}

COMMENT_MARKERS = re.compile(r"(--|/\*|\*/|#)")
RISKY_FUNCTIONS = {"load_extension", "randomblob", "sqlite_version"}
FORBIDDEN_AST_NAMES = (
    "Alter",
    "Attach",
    "Create",
    "Delete",
    "Detach",
    "Drop",
    "Insert",
    "Pragma",
    "Update",
)
FORBIDDEN_AST_TYPES = tuple(
    ast_type for name in FORBIDDEN_AST_NAMES if (ast_type := getattr(exp, name, None)) is not None
)


class AppError(Exception):
    pass


class ConfigError(AppError):
    pass


class IndexStateError(AppError):
    pass


class SqlValidationError(AppError):
    pass


class QueryExecutionError(AppError):
    pass


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool
    default: str | None = None
    primary_key: bool = False


@dataclass(frozen=True)
class ForeignKeyInfo:
    constrained_columns: tuple[str, ...]
    referred_table: str
    referred_columns: tuple[str, ...]


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...]
    foreign_keys: tuple[ForeignKeyInfo, ...]
    kind: str = "table"


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    sample_values: tuple[str, ...] = ()
    minimum: str | None = None
    maximum: str | None = None
    is_blob: bool = False


@dataclass(frozen=True)
class TableProfile:
    row_count: int | None = None
    columns: tuple[ColumnProfile, ...] = ()


@dataclass(frozen=True)
class SchemaDocument:
    table_name: str
    content: str


@dataclass(frozen=True)
class IndexSummary:
    table_count: int
    embedding_dimension: int
    metadata_path: Path
    rebuilt_at: str


@dataclass(frozen=True)
class RetrievedSchema:
    table_name: str
    content: str
    distance: float


@dataclass(frozen=True)
class UsageStats:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    sql: str
    initial_sql: str
    usage: UsageStats


@dataclass(frozen=True)
class ExecutionResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    execution_ms: float


@dataclass(frozen=True)
class ShapeExpectation:
    entity_terms: tuple[str, ...] = ()
    requires_numeric: bool = False
    requires_order: bool = False
    order_direction: str | None = None
    aggregate_kind: str | None = None
    expected_limit: int | None = None
    is_total_count: bool = False

    @property
    def has_expectations(self) -> bool:
        return bool(
            self.entity_terms
            or self.requires_numeric
            or self.requires_order
            or self.expected_limit is not None
            or self.is_total_count
        )


@dataclass(frozen=True)
class ShapeCheck:
    ok: bool
    warning: str | None
    expected: dict[str, Any]
    observed: dict[str, Any]


@dataclass
class AnswerResult:
    question: str
    retrieved_tables: list[str]
    expanded_tables: list[str]
    sql: str | None = None
    initial_sql: str | None = None
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    truncated: bool = False
    summary: str | None = None
    summary_mode: str | None = None
    summary_error: str | None = None
    success: bool = False
    executed: bool = False
    cancelled: bool = False
    repaired: bool = False
    repair_reason: str | None = None
    validation_error: str | None = None
    error_message: str | None = None
    shape_warning: str | None = None
    shape_expected: dict[str, Any] = field(default_factory=dict)
    shape_observed: dict[str, Any] = field(default_factory=dict)
    execution_ms: float | None = None
    token_usage: UsageStats = field(default_factory=UsageStats)
    auto_reindexed: bool = False
    freshness_warning: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def create_openai_client(settings: Settings) -> OpenAI:
    try:
        settings.validate_runtime()
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return OpenAI(base_url=settings.omlx_base_url, api_key=settings.omlx_api_key)


def translate_model_error(exc: openai.OpenAIError, base_url: str) -> AppError:
    """Convert an OpenAI/oMLX client error into an actionable AppError.

    Catches the broad ``openai.OpenAIError`` base class so it is robust across SDK
    versions, then maps the common cases to messages that name the server and what to check.
    """
    if isinstance(exc, openai.APIConnectionError):
        return AppError(
            f"Could not reach the model server at {base_url}. "
            "Make sure your local oMLX server is running and OMLX_BASE_URL is correct."
        )
    if isinstance(exc, openai.AuthenticationError):
        return AppError(f"The model server at {base_url} rejected the API key. Check OMLX_API_KEY.")
    if isinstance(exc, openai.NotFoundError):
        return AppError(
            f"The model server at {base_url} could not find the requested model. "
            "Check TEXT_TO_SQL_MODEL, EMBEDDING_MODEL, and SUMMARY_MODEL."
        )
    if isinstance(exc, openai.APIStatusError):
        return AppError(f"Model server error ({exc.status_code}) from {base_url}: {exc}")
    if isinstance(exc, openai.APIError):
        return AppError(f"Model server request failed for {base_url}: {exc}")
    return AppError(f"Model request failed for {base_url}: {exc}")


@contextmanager
def metadata_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Own the metadata transaction: commit on success, rollback on error, always close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def setup_metadata(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata_kv (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS schema_objects (
            id INTEGER PRIMARY KEY,
            object_type TEXT NOT NULL,
            object_name TEXT NOT NULL,
            table_name TEXT,
            content TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            generated_sql TEXT,
            initial_sql TEXT,
            success INTEGER NOT NULL,
            executed INTEGER NOT NULL DEFAULT 0,
            cancelled INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            validation_error TEXT,
            repair_reason TEXT,
            repaired INTEGER NOT NULL DEFAULT 0,
            retrieved_objects TEXT,
            summary TEXT,
            summary_mode TEXT,
            summary_error TEXT,
            shape_warning TEXT,
            execution_ms REAL,
            row_count INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            created_at TEXT NOT NULL
        );
        """
    )
    set_kv(conn, "metadata_schema_version", METADATA_SCHEMA_VERSION)


def set_kv(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO metadata_kv(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, json.dumps(value, sort_keys=True)),
    )


def get_kv(conn: sqlite3.Connection, key: str) -> Any | None:
    row = conn.execute("SELECT value FROM metadata_kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    try:
        import sqlite_vec
    except ImportError as exc:
        raise IndexStateError("sqlite-vec is not installed. Run uv sync.") from exc

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as exc:  # sqlite extension errors vary by platform.
        raise IndexStateError(
            "Could not load sqlite-vec. Ensure your Python sqlite3 supports loadable extensions."
        ) from exc
    finally:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def serialize_embedding(vector: list[float]) -> bytes:
    from sqlite_vec import serialize_float32

    return serialize_float32([float(value) for value in vector])


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier for safe interpolation into PRAGMA statements."""
    return '"' + name.replace('"', '""') + '"'


def sqlite_readonly_uri(path: Path) -> str:
    """Build a read-only SQLite URI from a filesystem path.

    Uses Path.as_uri(), which requires an absolute path and percent-encodes
    spaces and other special characters correctly.
    """
    return f"{path.expanduser().resolve().as_uri()}?mode=ro"


def extract_schema(db_path: Path) -> list[TableSchema]:
    if not db_path.exists():
        raise ConfigError(f"Database not found at {db_path}.")

    try:
        conn = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True)
    except sqlite3.Error as exc:
        raise ConfigError(f"Could not open database read-only: {exc}") from exc
    try:
        objects = [
            (str(name), str(obj_type))
            for (name, obj_type) in conn.execute(
                "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view')"
            )
            if not str(name).lower().startswith("sqlite_")
        ]
        tables = [read_table_schema(conn, name, kind=obj_type) for name, obj_type in objects]
    finally:
        conn.close()

    tables.sort(key=lambda table: table.name.lower())
    if not tables:
        raise ConfigError(f"No user tables or views found in {db_path}.")
    return tables


def read_table_schema(
    conn: sqlite3.Connection, table_name: str, kind: str = "table"
) -> TableSchema:
    quoted = quote_identifier(table_name)
    columns: list[ColumnInfo] = []
    pk_members: list[tuple[int, str]] = []
    for _cid, name, decl_type, notnull, default, pk in conn.execute(f"PRAGMA table_info({quoted})"):
        column_name = str(name)
        if int(pk) > 0:
            pk_members.append((int(pk), column_name))
        columns.append(
            ColumnInfo(
                name=column_name,
                data_type=str(decl_type) or "UNKNOWN",
                nullable=not int(notnull),
                default=None if default is None else str(default),
                primary_key=int(pk) > 0,
            )
        )
    primary_key = tuple(name for _, name in sorted(pk_members))
    foreign_keys = read_foreign_keys(conn, quoted)
    return TableSchema(table_name, tuple(columns), primary_key, foreign_keys, kind=kind)


def read_foreign_keys(conn: sqlite3.Connection, quoted_table: str) -> tuple[ForeignKeyInfo, ...]:
    grouped: dict[int, list[tuple[int, str, Any]]] = {}
    referred_tables: dict[int, str] = {}
    for fk_id, seq, referred_table, from_col, to_col, *_rest in conn.execute(
        f"PRAGMA foreign_key_list({quoted_table})"
    ):
        grouped.setdefault(int(fk_id), []).append((int(seq), str(from_col), to_col))
        referred_tables[int(fk_id)] = str(referred_table)

    foreign_keys: list[ForeignKeyInfo] = []
    for fk_id in sorted(grouped):
        members = sorted(grouped[fk_id], key=lambda item: item[0])
        constrained = tuple(from_col for _, from_col, _ in members)
        referred = tuple(str(to_col) for _, _, to_col in members if to_col is not None)
        referred_table = referred_tables[fk_id]
        if referred_table and constrained and len(referred) == len(constrained):
            foreign_keys.append(ForeignKeyInfo(constrained, referred_table, referred))
    return tuple(foreign_keys)


def build_schema_document(
    table: TableSchema, profile: TableProfile | None = None
) -> SchemaDocument:
    is_view = table.kind == "view"
    kind_label = "view" if is_view else "table"
    object_label = "View" if is_view else "Table"
    lines = [f"Object type: {kind_label}", f"{object_label}: {table.name}", "Columns:"]
    pk_set = set(table.primary_key)
    for column in table.columns:
        bits = [f"- {column.name}", column.data_type]
        if column.name in pk_set or column.primary_key:
            bits.append("PRIMARY KEY")
        if not column.nullable and column.name not in pk_set:
            bits.append("NOT NULL")
        if column.default is not None:
            bits.append(f"DEFAULT {column.default}")
        lines.append(" ".join(bits))

    lines.append("Relationships:")
    if table.foreign_keys:
        for fk in table.foreign_keys:
            left = ", ".join(f"{table.name}.{col}" for col in fk.constrained_columns)
            right = ", ".join(f"{fk.referred_table}.{col}" for col in fk.referred_columns)
            lines.append(f"- {left} references {right}")
    else:
        lines.append("- none")

    if profile is not None:
        lines.extend(profile_document_lines(table, profile))
    return SchemaDocument(table.name, "\n".join(lines))


def profile_document_lines(table: TableSchema, profile: TableProfile) -> list[str]:
    lines = ["Profile (sampled from a read-only connection):"]
    if profile.row_count is not None:
        lines.append(f"- Row count: {profile.row_count}")
    elif table.kind == "view":
        lines.append("- Row count: not profiled (view)")
    for column in profile.columns:
        if column.is_blob:
            lines.append(f"- {column.name}: binary data (not profiled)")
            continue
        bits: list[str] = []
        if column.sample_values:
            bits.append("samples: " + ", ".join(column.sample_values))
        if column.minimum is not None or column.maximum is not None:
            bits.append(f"min: {column.minimum}, max: {column.maximum}")
        if bits:
            lines.append(f"- {column.name}: " + "; ".join(bits))
    return lines


def build_schema_documents(
    tables: list[TableSchema], profiles: dict[str, TableProfile] | None = None
) -> list[SchemaDocument]:
    profiles = profiles or {}
    return [build_schema_document(table, profiles.get(table.name)) for table in tables]


def is_numeric_declared_type(declared_type: str) -> bool:
    upper = (declared_type or "").upper()
    return any(token in upper for token in ("INT", "REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL"))


def is_blob_declared_type(declared_type: str) -> bool:
    return "BLOB" in (declared_type or "").upper()


def format_profile_value(value: Any, max_len: int) -> str | None:
    """Render one sampled value for a schema document, or None to skip it.

    BLOB-like values are skipped so binary data never lands in a prompt; long text is
    truncated to ``max_len`` characters with a trailing ellipsis.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def profile_table(
    conn: sqlite3.Connection, table: TableSchema, settings: Settings
) -> TableProfile:
    """Collect a bounded, read-only profile for one table or view.

    Row counts and MIN/MAX are skipped for views (their queries can be expensive). Every
    sub-query is isolated so a single failure degrades that piece only, never the table.
    """
    is_view = table.kind == "view"
    quoted_table = quote_identifier(table.name)

    row_count: int | None = None
    if not is_view:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()
            row_count = int(row[0]) if row and row[0] is not None else None
        except sqlite3.Error:
            row_count = None

    column_profiles: list[ColumnProfile] = []
    for column in table.columns:
        if is_blob_declared_type(column.data_type):
            column_profiles.append(ColumnProfile(name=column.name, is_blob=True))
            continue
        quoted_col = quote_identifier(column.name)
        sample_values: tuple[str, ...] = ()
        try:
            rows = conn.execute(
                f"SELECT {quoted_col} FROM {quoted_table} "
                f"WHERE {quoted_col} IS NOT NULL LIMIT ?",
                (settings.max_profile_values,),
            ).fetchall()
            formatted = [
                rendered
                for (value,) in rows
                if (rendered := format_profile_value(value, settings.max_profile_text_length))
                is not None
            ]
            sample_values = tuple(formatted)
        except sqlite3.Error:
            sample_values = ()

        minimum: str | None = None
        maximum: str | None = None
        if not is_view and is_numeric_declared_type(column.data_type):
            try:
                bounds = conn.execute(
                    f"SELECT MIN({quoted_col}), MAX({quoted_col}) FROM {quoted_table}"
                ).fetchone()
                if bounds:
                    minimum = None if bounds[0] is None else str(bounds[0])
                    maximum = None if bounds[1] is None else str(bounds[1])
            except sqlite3.Error:
                minimum = maximum = None

        column_profiles.append(
            ColumnProfile(column.name, sample_values, minimum, maximum, is_blob=False)
        )
    return TableProfile(row_count=row_count, columns=tuple(column_profiles))


def profile_tables(
    db_path: Path, tables: list[TableSchema], settings: Settings
) -> dict[str, TableProfile]:
    """Profile each table/view through a single bounded, read-only connection.

    Never loads sqlite-vec. A progress-handler timeout keeps profiling from hanging on a
    huge or expensive object, and per-table isolation prevents one failure from aborting
    the rest.
    """
    profiles: dict[str, TableProfile] = {}
    try:
        conn = sqlite3.connect(sqlite_readonly_uri(db_path), uri=True)
    except sqlite3.Error:
        return profiles
    try:
        try:
            conn.enable_load_extension(False)
        except sqlite3.Error:
            pass

        start = time.monotonic()

        def progress_handler() -> int:
            return 1 if (time.monotonic() - start) * 1000 > settings.query_timeout_ms else 0

        conn.set_progress_handler(progress_handler, 1000)
        for table in tables:
            try:
                profiles[table.name] = profile_table(conn, table, settings)
            except sqlite3.Error:
                continue
    finally:
        try:
            conn.set_progress_handler(None, 0)
        finally:
            conn.close()
    return profiles


def schema_documents_for_index(
    settings: Settings, tables: list[TableSchema]
) -> list[SchemaDocument]:
    """Build schema documents for indexing, profiling once when enabled."""
    profiles = (
        profile_tables(settings.source_db_path, tables, settings)
        if settings.enable_schema_profiling
        else None
    )
    return build_schema_documents(tables, profiles)


def schema_fingerprint(tables: list[TableSchema]) -> str:
    payload = [
        {
            "name": table.name,
            "kind": table.kind,
            "columns": [
                {
                    "name": column.name,
                    "type": column.data_type,
                    "nullable": column.nullable,
                    "default": column.default,
                    "primary_key": column.primary_key,
                }
                for column in table.columns
            ],
            "primary_key": list(table.primary_key),
            "foreign_keys": [
                {
                    "constrained_columns": list(fk.constrained_columns),
                    "referred_table": fk.referred_table,
                    "referred_columns": list(fk.referred_columns),
                }
                for fk in table.foreign_keys
            ],
        }
        for table in tables
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def schema_document_signature(settings: Settings) -> str:
    """Signature of how schema documents are rendered/profiled.

    A change here (doc-format version bump, profiling toggled, or profiling limits changed)
    means stored documents are stale even when the schema fingerprint is unchanged, so
    ``ensure_index_ready`` reindexes once.
    """
    payload = {
        "doc_version": SCHEMA_DOCUMENT_VERSION,
        "profiling": settings.enable_schema_profiling,
        "max_profile_values": settings.max_profile_values,
        "max_profile_text_length": settings.max_profile_text_length,
    }
    return json.dumps(payload, sort_keys=True)


def embed_texts(
    client: OpenAI,
    model: str,
    texts: list[str],
    batch_size: int,
    *,
    instruction: str,
) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = [f"{instruction}\n{text}" for text in texts[start : start + batch_size]]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except openai.OpenAIError as exc:
            base_url = str(getattr(client, "base_url", "configured model server"))
            raise translate_model_error(exc, base_url) from exc
        for item in sorted(response.data, key=lambda data: data.index):
            embeddings.append([float(value) for value in item.embedding])
    if len(embeddings) != len(texts):
        raise IndexStateError("Embedding response count did not match input count.")
    return embeddings


def index_schema(
    settings: Settings, client: OpenAI, tables: list[TableSchema] | None = None
) -> IndexSummary:
    source_path = settings.source_db_path
    if tables is None:
        tables = extract_schema(source_path)
    documents = schema_documents_for_index(settings, tables)
    kind_by_name = {table.name: table.kind for table in tables}
    texts = [document.content for document in documents]
    embeddings = embed_texts(
        client,
        settings.embedding_model,
        texts,
        settings.embedding_batch_size,
        instruction="Represent this database schema object for retrieval:",
    )
    dimension = len(embeddings[0])
    if dimension <= 0:
        raise IndexStateError("Embedding model returned an empty vector.")
    if any(len(vector) != dimension for vector in embeddings):
        raise IndexStateError("Embedding model returned inconsistent vector dimensions.")

    metadata_path = settings.metadata_path
    with metadata_connection(metadata_path) as conn:
        setup_metadata(conn)
        load_sqlite_vec(conn)
        conn.execute("DROP TABLE IF EXISTS schema_vectors")
        conn.execute("DELETE FROM schema_objects")
        conn.execute(
            f"CREATE VIRTUAL TABLE schema_vectors USING vec0(embedding float[{dimension}])"
        )
        created_at = utc_now()
        for document, embedding in zip(documents, embeddings, strict=True):
            cursor = conn.execute(
                """
                INSERT INTO schema_objects(
                    object_type, object_name, table_name, content,
                    embedding_model, embedding_dimension, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind_by_name.get(document.table_name, "table"),
                    document.table_name,
                    document.table_name,
                    document.content,
                    settings.embedding_model,
                    dimension,
                    created_at,
                ),
            )
            conn.execute(
                "INSERT INTO schema_vectors(rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, serialize_embedding(embedding)),
            )
        stat = source_path.stat()
        set_kv(conn, "embedding_model", settings.embedding_model)
        set_kv(conn, "embedding_dimension", dimension)
        set_kv(conn, "source_db_path", str(source_path))
        set_kv(conn, "source_db_size", stat.st_size)
        set_kv(conn, "source_db_mtime_ns", stat.st_mtime_ns)
        set_kv(conn, "schema_fingerprint", schema_fingerprint(tables))
        set_kv(conn, "schema_document_signature", schema_document_signature(settings))
        set_kv(conn, "indexed_at", created_at)
    return IndexSummary(len(documents), dimension, metadata_path, created_at)


def ensure_index_ready(
    settings: Settings, client: OpenAI, tables: list[TableSchema]
) -> tuple[bool, str | None]:
    metadata_path = settings.metadata_path
    if not metadata_path.exists():
        if settings.auto_reindex:
            index_schema(settings, client, tables)
            return True, None
        raise IndexStateError("Schema index is missing. Run uv run python app.py index.")

    current_fingerprint = schema_fingerprint(tables)
    stat = settings.source_db_path.stat()

    with metadata_connection(metadata_path) as conn:
        setup_metadata(conn)
        row = conn.execute("SELECT COUNT(*) FROM schema_objects").fetchone()
        indexed_count = int(row[0]) if row else 0
        vector_row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_vectors'"
        ).fetchone()
        vector_table_exists = vector_row is not None
        stored_model = get_kv(conn, "embedding_model")
        stored_path = get_kv(conn, "source_db_path")
        stored_fingerprint = get_kv(conn, "schema_fingerprint")
        stored_signature = get_kv(conn, "schema_document_signature")
        stored_size = get_kv(conn, "source_db_size")
        stored_mtime = get_kv(conn, "source_db_mtime_ns")

    must_reindex = (
        indexed_count == 0
        or not vector_table_exists
        or stored_model != settings.embedding_model
        or stored_path != str(settings.source_db_path)
        or stored_fingerprint != current_fingerprint
        or stored_signature != schema_document_signature(settings)
    )
    if must_reindex:
        if settings.auto_reindex:
            index_schema(settings, client, tables)
            return True, None
        raise IndexStateError("Schema index is stale. Run uv run python app.py index.")

    data_changed = stored_size != stat.st_size or stored_mtime != stat.st_mtime_ns
    warning = None
    if data_changed:
        warning = "Database file changed since indexing, but table schema is unchanged."
        with metadata_connection(metadata_path) as conn:
            setup_metadata(conn)
            set_kv(conn, "source_db_size", stat.st_size)
            set_kv(conn, "source_db_mtime_ns", stat.st_mtime_ns)
    return False, warning


def retrieve_schema(
    settings: Settings,
    client: OpenAI,
    question: str,
) -> list[RetrievedSchema]:
    query_embedding = embed_texts(
        client,
        settings.embedding_model,
        [question],
        1,
        instruction="Retrieve the database schema objects needed to answer this question:",
    )[0]
    with metadata_connection(settings.metadata_path) as conn:
        setup_metadata(conn)
        load_sqlite_vec(conn)
        stored_dimension = get_kv(conn, "embedding_dimension")
        if stored_dimension != len(query_embedding):
            raise IndexStateError("Embedding dimension changed. Rebuild the schema index.")
        rows = conn.execute(
            """
            SELECT so.table_name, so.content, v.distance
            FROM schema_vectors AS v
            JOIN schema_objects AS so ON so.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (serialize_embedding(query_embedding), settings.retrieval_top_k),
        ).fetchall()

    return [
        RetrievedSchema(str(table_name), str(content), float(distance))
        for table_name, content, distance in rows
    ]


def load_indexed_documents(settings: Settings) -> list[SchemaDocument]:
    """Return the schema documents stored at index time (profiled when profiling is on).

    Reused for prompts so the model sees exactly what was embedded, and so profiling runs
    only during indexing rather than on every question.
    """
    with metadata_connection(settings.metadata_path) as conn:
        setup_metadata(conn)
        rows = conn.execute(
            "SELECT object_name, content FROM schema_objects ORDER BY object_name"
        ).fetchall()
    documents = [SchemaDocument(str(name), str(content)) for name, content in rows]
    if not documents:
        raise IndexStateError(
            "Schema index is empty. Run uv run python app.py index or enable AUTO_REINDEX."
        )
    return documents


def relationship_neighbors(tables: list[TableSchema]) -> dict[str, set[str]]:
    neighbors: dict[str, set[str]] = {table.name: set() for table in tables}
    for table in tables:
        for fk in table.foreign_keys:
            if fk.referred_table in neighbors:
                neighbors[table.name].add(fk.referred_table)
                neighbors[fk.referred_table].add(table.name)
    return neighbors


def expanded_schema_order(retrieved: list[RetrievedSchema], tables: list[TableSchema]) -> list[str]:
    table_names = {table.name for table in tables}
    neighbors = relationship_neighbors(tables)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        if item.table_name not in table_names:
            continue
        if item.table_name not in seen:
            ordered.append(item.table_name)
            seen.add(item.table_name)
        for neighbor in sorted(neighbors.get(item.table_name, set()), key=str.lower):
            if neighbor not in seen:
                ordered.append(neighbor)
                seen.add(neighbor)
    return ordered


def schema_text_for_tables(table_names: list[str], documents: list[SchemaDocument]) -> str:
    docs = {doc.table_name: doc.content for doc in documents}
    return "\n\n".join(docs[name] for name in table_names if name in docs)


def all_schema_text(documents: list[SchemaDocument]) -> str:
    return "\n\n".join(document.content for document in documents)


def quote_guidance() -> str:
    return "Use double quotes for identifiers with spaces or special characters."


BREAKDOWN_RE = re.compile(
    r"\b(per|each|every|across|grouped\s+by|group\s+by|break\s*down|breakdown)\b|\bby\s+\w+"
)


def question_has_count_intent(question: str) -> bool:
    """True when the question asks to count rows (how many / count / number of)."""
    text = question.lower()
    return "how many" in text or "number of" in text or re.search(r"\bcount\b", text) is not None


def is_total_count_question(question: str) -> bool:
    """A count question with no breakdown phrase expects a single total (no GROUP BY)."""
    if not question_has_count_intent(question):
        return False
    return BREAKDOWN_RE.search(question.lower()) is None


def infer_shape_expectation(question: str) -> ShapeExpectation:
    text = question.lower()
    entity_specs = [
        ("customers?", ("customer", "company", "contact")),
        ("products?", ("product",)),
        ("employees?", ("first", "last", "name")),
        ("suppliers?", ("supplier", "company", "contact")),
        ("shippers?", ("shipper", "company")),
        ("categories", ("category",)),
        ("category", ("category",)),
        ("orders?", ("order",)),
    ]
    entity_matches: list[tuple[int, tuple[str, ...]]] = []
    for pattern, terms in entity_specs:
        match = re.search(rf"\b{pattern}\b", text)
        if match:
            entity_matches.append((match.start(), terms))
    entity_terms = list(min(entity_matches, key=lambda item: item[0])[1]) if entity_matches else []
    if question_has_count_intent(question):
        # A counted entity is aggregated into a metric, not rendered as a label column,
        # so do not require an entity label for "how many / count / number of" questions.
        entity_terms = []

    numeric_words = (
        "count",
        "how many",
        "total",
        "sum",
        "average",
        "avg",
        "mean",
        "minimum",
        "maximum",
        "most",
        "least",
        "top",
        "highest",
        "lowest",
        "largest",
        "smallest",
    )
    requires_numeric = any(word in text for word in numeric_words)
    order_words = ("most", "least", "top", "highest", "lowest", "largest", "smallest")
    requires_order = any(word in text for word in order_words)
    order_direction = None
    if requires_order:
        order_direction = "ASC" if any(word in text for word in ("least", "lowest", "smallest")) else "DESC"

    aggregate_kind = None
    if "average" in text or "avg" in text or "mean" in text:
        aggregate_kind = "average"
    elif "total" in text or "sum" in text:
        aggregate_kind = "total"
    elif "count" in text or "how many" in text or "most" in text or "least" in text:
        aggregate_kind = "count"

    expected_limit = None
    limit_patterns = (
        r"\btop\s+(\d{1,3})\b",
        r"\bfirst\s+(\d{1,3})\b",
        r"\blimit\s+(\d{1,3})\b",
    )
    for pattern in limit_patterns:
        match = re.search(pattern, text)
        if match:
            expected_limit = max(1, int(match.group(1)))
            break

    return ShapeExpectation(
        entity_terms=tuple(dict.fromkeys(entity_terms)),
        requires_numeric=requires_numeric,
        requires_order=requires_order,
        order_direction=order_direction,
        aggregate_kind=aggregate_kind,
        expected_limit=expected_limit,
        is_total_count=is_total_count_question(question),
    )


def shape_expectation_text(expectation: ShapeExpectation) -> str:
    if not expectation.has_expectations:
        return "- No special result-shape constraint inferred."
    lines = []
    if expectation.is_total_count:
        lines.append(
            "- Return a single total count using COUNT(*) with no GROUP BY "
            "(one row, one numeric column)."
        )
    if expectation.entity_terms:
        lines.append(
            "- Include a human-readable entity identifier column. "
            "Prefer source columns whose names contain: "
            + ", ".join(expectation.entity_terms)
            + "."
        )
    if expectation.requires_numeric:
        lines.append("- Include a numeric metric column.")
    if expectation.requires_order:
        lines.append(f"- Include ORDER BY for ranked results, preferably {expectation.order_direction}.")
    if expectation.expected_limit is not None:
        lines.append(f"- Return at most {expectation.expected_limit} rows because the user asked for that count.")
    return "\n".join(lines)


def build_sql_prompt(
    question: str,
    schema_text: str,
    max_result_rows: int,
    expectation: ShapeExpectation,
) -> str:
    return f"""You are an expert SQLite query generator.
Generate exactly one read-only SQLite SELECT query for the user's question.
Rules:
- Return SQL only. Do not explain your reasoning.
- Do not include analysis, prose, bullets, or markdown.
- Use only the provided tables and columns.
- Every selected source column must come from an explicit FROM or JOIN table.
- Do not invent tables or columns.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH, or DETACH.
- Do not call load_extension, randomblob, or sqlite_version.
- Do not use comments.
- Use explicit JOIN conditions.
- Use SQLite-compatible syntax.
- {quote_guidance()}
- Prefer explicit columns over SELECT *.
- If the user asks to list entities and a join is only needed for filtering, prefer DISTINCT entity rows over repeated transaction rows.
- If the user asks for top N, first N, or a specific row count, use that LIMIT. Otherwise add LIMIT {max_result_rows} for detailed row queries.
- If the user asks for a total count, return a single COUNT(*) aggregate row. Do not GROUP BY unless the user asks for a breakdown by a dimension.
- Return only SQL. Do not use markdown fences.
Expected result shape:
{shape_expectation_text(expectation)}
Relevant schema:
{schema_text}
User question:
{question}

SQL:"""


SQL_REPAIR_RULES = """Rules:
- Return SQL only. Do not explain your reasoning.
- Do not include analysis, prose, bullets, or markdown.
- Use only the provided tables and columns.
- Every selected source column must come from an explicit FROM or JOIN table.
- {quote_guidance}
- Use explicit JOIN conditions.
- If the user asks to list entities and a join is only needed for filtering, prefer DISTINCT entity rows over repeated transaction rows.
- If the user asks for top N, first N, or a specific row count, use that LIMIT. Otherwise add LIMIT {max_result_rows} for detailed row queries.
- If the user asks for a total count, return a single COUNT(*) aggregate row. Do not GROUP BY unless the user asks for a breakdown by a dimension.
- Do not use comments or prohibited operations."""


def build_validation_repair_prompt(
    question: str,
    schema_text: str,
    invalid_sql: str,
    validation_error: str,
    max_result_rows: int,
    expectation: ShapeExpectation,
) -> str:
    rules = SQL_REPAIR_RULES.format(quote_guidance=quote_guidance(), max_result_rows=max_result_rows)
    return f"""The previous SQL was invalid or unsafe.
Generate exactly one corrected read-only SQLite SELECT query.
Return only SQL.

{rules}

Expected result shape:
{shape_expectation_text(expectation)}

Validation error:
{validation_error}

Invalid SQL:
{invalid_sql}

Relevant schema:
{schema_text}

User question:
{question}

Corrected SQL:"""


def build_shape_repair_prompt(
    question: str,
    schema_text: str,
    sql: str,
    expected: dict[str, Any],
    observed: dict[str, Any],
    max_result_rows: int,
) -> str:
    rules = SQL_REPAIR_RULES.format(quote_guidance=quote_guidance(), max_result_rows=max_result_rows)
    return f"""The SQL was safe and executable, but the returned result shape may not match the question.
Generate exactly one corrected read-only SQLite SELECT query.
Return only SQL.

{rules}

Expected result shape:
{json.dumps(expected, sort_keys=True)}

Observed result shape:
{json.dumps(observed, sort_keys=True)}

Previous SQL:
{sql}

Relevant schema:
{schema_text}

User question:
{question}

Corrected SQL:"""


def extract_usage(response: Any) -> UsageStats:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageStats()
    return UsageStats(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def generate_sql(client: OpenAI, settings: Settings, prompt: str) -> GenerationResult:
    try:
        response = client.chat.completions.create(
            model=settings.text_to_sql_model,
            messages=[
                {"role": "system", "content": "You generate safe, precise SQLite SELECT queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=settings.sql_temperature,
            top_p=settings.sql_top_p,
            max_tokens=settings.max_sql_tokens,
        )
    except openai.OpenAIError as exc:
        base_url = str(getattr(client, "base_url", settings.omlx_base_url))
        raise translate_model_error(exc, base_url) from exc
    content = response.choices[0].message.content or ""
    sql = extract_sql(content)
    return GenerationResult(sql=sql, initial_sql=sql, usage=extract_usage(response))


def extract_sql(raw: str) -> str:
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:sql|sqlite)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if re.match(r"^(SELECT|WITH)\b", text, flags=re.IGNORECASE):
        return text
    fenced_blocks = re.findall(
        r"```(?:sql|sqlite)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_blocks:
        return fenced_blocks[-1].strip()
    lines = text.splitlines()
    select_indexes = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^(SELECT|WITH)\b", line.strip(), flags=re.IGNORECASE)
    ]
    if select_indexes:
        collected: list[str] = []
        for line in lines[select_indexes[-1] :]:
            stripped = line.strip()
            if not stripped and collected:
                break
            if stripped:
                collected.append(stripped)
        return "\n".join(collected).strip()
    return text


def validate_sql(sql: str, tables: list[TableSchema]) -> None:
    normalized = sql.strip()
    if not normalized:
        raise SqlValidationError("SQL is empty.")
    if COMMENT_MARKERS.search(normalized):
        raise SqlValidationError("SQL comments are not allowed.")
    try:
        parsed = sqlglot.parse(normalized, read="sqlite")
    except sqlglot.errors.SqlglotError as exc:
        raise SqlValidationError(f"SQL could not be parsed as SQLite: {exc}") from exc
    if len(parsed) != 1:
        raise SqlValidationError("SQL must contain exactly one statement.")
    expression = parsed[0]
    for forbidden_type in FORBIDDEN_AST_TYPES:
        if expression.find(forbidden_type) is not None:
            raise SqlValidationError("SQL contains a prohibited operation or command.")
    if not isinstance(expression, exp.Select):
        raise SqlValidationError("SQL must be a SELECT query.")
    with_expression = expression.args.get("with_")
    if with_expression is not None and with_expression.args.get("recursive"):
        raise SqlValidationError("Recursive CTEs are not allowed.")

    known_tables = {table.name.lower(): table for table in tables}
    known_columns = {
        table.name.lower(): {column.name.lower() for column in table.columns}
        for table in tables
    }

    cte_names = {
        cte.alias_or_name.lower()
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }
    shadowed = sorted(name for name in cte_names if name in known_tables)
    if shadowed:
        raise SqlValidationError(f"CTE name shadows a real table: {', '.join(shadowed)}.")

    subquery_aliases = {
        subquery.alias_or_name.lower()
        for subquery in expression.find_all(exp.Subquery)
        if subquery.alias_or_name
    }

    alias_to_table: dict[str, str] = {}
    referenced_tables: set[str] = set()
    has_relation = False
    for table_expr in expression.find_all(exp.Table):
        table_name = table_expr.name
        if not table_name:
            continue
        lower_name = table_name.lower()
        if lower_name in cte_names:
            has_relation = True
            continue
        if lower_name.startswith("sqlite_"):
            raise SqlValidationError("SQLite system tables are not allowed.")
        if lower_name not in known_tables:
            raise SqlValidationError(f"Unknown table: {table_name}.")
        has_relation = True
        referenced_tables.add(lower_name)
        alias_to_table[lower_name] = lower_name
        alias = table_expr.alias_or_name
        if alias:
            alias_to_table[alias.lower()] = lower_name

    select_aliases = {
        alias.alias.lower()
        for alias in expression.find_all(exp.Alias)
        if alias.alias
    }
    column_expressions = list(expression.find_all(exp.Column))
    if not has_relation and any(column.name and column.name != "*" for column in column_expressions):
        raise SqlValidationError("SQL references columns without a FROM table.")

    for column in column_expressions:
        column_name = column.name
        if not column_name or column_name == "*":
            continue
        lower_column = column_name.lower()
        qualifier = column.table
        if qualifier:
            lower_qualifier = qualifier.lower()
            if lower_qualifier in cte_names or lower_qualifier in subquery_aliases:
                continue
            real_table = alias_to_table.get(lower_qualifier)
            if real_table is None:
                raise SqlValidationError(f"Unknown table or alias qualifier: {qualifier}.")
            if lower_column not in known_columns[real_table]:
                raise SqlValidationError(f"Unknown column: {qualifier}.{column_name}.")
            continue

        if lower_column in select_aliases:
            continue
        if referenced_tables:
            matching_tables = [
                table_name
                for table_name in referenced_tables
                if lower_column in known_columns[table_name]
            ]
            if len(matching_tables) == 1:
                continue
            if len(matching_tables) > 1:
                raise SqlValidationError(f"Ambiguous unqualified column: {column_name}.")
            raise SqlValidationError(f"Unknown column: {column_name}.")
        if not referenced_tables and any(
            lower_column in columns for columns in known_columns.values()
        ):
            continue
        raise SqlValidationError(f"Unknown column: {column_name}.")

    for func in expression.find_all(exp.Func):
        name = (func.sql_name() or "").lower()
        if name in RISKY_FUNCTIONS:
            raise SqlValidationError(f"Function is not allowed: {name}.")
    if re.search(r"\b(load_extension|randomblob|sqlite_version)\s*\(", normalized, re.IGNORECASE):
        raise SqlValidationError("SQL calls a prohibited function.")


def has_order_by(sql: str) -> bool:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    return parsed.find(exp.Order) is not None


def has_group_by(sql: str) -> bool:
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return False
    return parsed.find(exp.Group) is not None


def execute_readonly_query(
    db_path: Path,
    sql: str,
    *,
    max_rows: int,
    timeout_ms: int,
) -> ExecutionResult:
    uri = sqlite_readonly_uri(db_path)
    start = time.monotonic()
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise QueryExecutionError(f"Could not open database read-only: {exc}") from exc
    try:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass

        def progress_handler() -> int:
            elapsed_ms = (time.monotonic() - start) * 1000
            return 1 if elapsed_ms > timeout_ms else 0

        conn.set_progress_handler(progress_handler, 1000)
        cursor = conn.execute(sql)
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = tuple(tuple(row) for row in fetched[:max_rows])
        columns = tuple(description[0] for description in cursor.description or ())
        execution_ms = (time.monotonic() - start) * 1000
        return ExecutionResult(columns, rows, truncated, execution_ms)
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            raise QueryExecutionError(f"Query exceeded timeout of {timeout_ms} ms.") from exc
        raise QueryExecutionError(f"Query execution failed: {message}") from exc
    except sqlite3.Error as exc:
        raise QueryExecutionError(f"Query execution failed: {exc}") from exc
    finally:
        try:
            conn.set_progress_handler(None, 0)
        finally:
            conn.close()


def infer_column_types(columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> dict[str, str]:
    types: dict[str, str] = {}
    numeric_name_hints = (
        "count",
        "total",
        "sum",
        "avg",
        "average",
        "quantity",
        "price",
        "amount",
        "freight",
        "number",
    )
    for index, column in enumerate(columns):
        values = [row[index] for row in rows if row[index] is not None]
        if any(isinstance(value, bool) for value in values):
            types[column] = "boolean"
        elif any(isinstance(value, (int, float)) for value in values):
            types[column] = "numeric"
        elif any(isinstance(value, str) for value in values):
            types[column] = "text"
        elif any(hint in column.lower() for hint in numeric_name_hints):
            types[column] = "numeric"
        else:
            types[column] = "unknown"
    return types


def check_result_shape(
    question: str,
    sql: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    expectation: ShapeExpectation | None = None,
) -> ShapeCheck:
    if expectation is None:
        expectation = infer_shape_expectation(question)
    column_types = infer_column_types(columns, rows)
    lower_columns = [column.lower() for column in columns]
    observed = {
        "columns": list(columns),
        "column_types": column_types,
        "row_count": len(rows),
        "has_order_by": has_order_by(sql),
        "has_group_by": has_group_by(sql),
    }
    expected = {
        "entity_terms": list(expectation.entity_terms),
        "requires_numeric": expectation.requires_numeric,
        "requires_order": expectation.requires_order,
        "order_direction": expectation.order_direction,
        "aggregate_kind": expectation.aggregate_kind,
        "expected_limit": expectation.expected_limit,
        "is_total_count": expectation.is_total_count,
    }
    warnings: list[str] = []

    if expectation.is_total_count and (
        observed["has_group_by"] or len(rows) > 1 or len(columns) != 1
    ):
        warnings.append(
            "the question asks for a single total count, but the SQL groups rows or returns "
            "multiple rows/columns; use COUNT(*) without GROUP BY unless the user asks for a "
            "breakdown"
        )

    if expectation.entity_terms and columns:
        has_entity = any(
            "id" not in column
            and any(term in column for term in expectation.entity_terms)
            for column in lower_columns
        )
        if not has_entity:
            warnings.append(
                "expected an entity label column matching "
                + ", ".join(expectation.entity_terms)
            )

    if expectation.requires_numeric and columns:
        if not any(column_type == "numeric" for column_type in column_types.values()):
            warnings.append("expected a numeric metric column")

    if expectation.requires_order and not observed["has_order_by"]:
        warnings.append("expected ORDER BY for ranked results")

    if expectation.expected_limit is not None and len(rows) > expectation.expected_limit:
        warnings.append(f"expected at most {expectation.expected_limit} rows")

    if warnings:
        return ShapeCheck(False, "Result shape mismatch: " + "; ".join(warnings) + ".", expected, observed)
    return ShapeCheck(True, None, expected, observed)


def deterministic_summary(
    question: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    truncated: bool,
    expectation: ShapeExpectation | None = None,
) -> str:
    row_count = len(rows)
    if row_count == 0:
        return "The query returned no rows."
    if row_count == 1 and len(columns) == 1:
        return f"The result is {format_value(rows[0][0])}."
    if truncated:
        return f"Returned {row_count} rows; more rows may exist."

    column_types = infer_column_types(columns, rows)
    numeric_indexes = [
        index
        for index, column in enumerate(columns)
        if column_types.get(column) == "numeric" and is_metric_column(column)
    ]
    if rows and len(columns) >= 2 and numeric_indexes:
        if expectation is None:
            expectation = infer_shape_expectation(question)
        metric_index = numeric_indexes[-1]
        direction = expectation.order_direction or "DESC"
        ranked_row = ranked_numeric_row(rows, metric_index, direction)
        label = format_label(columns, ranked_row, rows, metric_index)
        metric_name = columns[metric_index]
        if expectation.requires_order:
            prefix = "Lowest result" if direction == "ASC" else "Top result"
            return f"{prefix}: {label} with {format_value(ranked_row[metric_index])}."
        return (
            f"Highest {metric_name} is {label} "
            f"with {format_value(ranked_row[metric_index])}."
        )
    return f"Returned {row_count} row{'s' if row_count != 1 else ''}."


def is_metric_column(column: str) -> bool:
    lower = column.lower()
    if lower.endswith("id") or lower == "id":
        return False
    metric_hints = (
        "count",
        "total",
        "sum",
        "avg",
        "average",
        "quantity",
        "price",
        "amount",
        "freight",
        "metric",
        "rate",
        "value",
    )
    return any(hint in lower for hint in metric_hints)


def ranked_numeric_row(
    rows: tuple[tuple[Any, ...], ...],
    metric_index: int,
    direction: str,
) -> tuple[Any, ...]:
    numeric_rows = [
        row for row in rows if isinstance(row[metric_index], (int, float)) and row[metric_index] is not None
    ]
    if not numeric_rows:
        return rows[0]
    reverse = direction != "ASC"
    return sorted(numeric_rows, key=lambda row: row[metric_index], reverse=reverse)[0]


def format_label(
    columns: tuple[str, ...],
    row: tuple[Any, ...],
    rows: tuple[tuple[Any, ...], ...],
    metric_index: int,
) -> str:
    lower_columns = [column.lower() for column in columns]
    first_index = next(
        (index for index, column in enumerate(lower_columns) if "first" in column),
        None,
    )
    last_index = next(
        (index for index, column in enumerate(lower_columns) if "last" in column),
        None,
    )
    if (
        first_index is not None
        and last_index is not None
        and first_index != metric_index
        and last_index != metric_index
    ):
        return f"{format_value(row[first_index])} {format_value(row[last_index])}"
    label_index = choose_label_index(columns, rows, metric_index)
    return format_value(row[label_index])


def choose_label_index(
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    metric_index: int,
) -> int:
    preferred_name_hints = ("name", "title", "company", "contact", "customer", "product", "employee", "supplier")
    candidates = [index for index in range(len(columns)) if index != metric_index]
    varied_candidates = [
        index
        for index in candidates
        if len({row[index] for row in rows if row[index] is not None}) > 1
    ]
    search_space = varied_candidates or candidates
    readable_space = [index for index in search_space if "id" not in columns[index].lower()]
    for index in readable_space or search_space:
        if any(hint in columns[index].lower() for hint in preferred_name_hints):
            return index
    return (readable_space or search_space)[0] if search_space else 0


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    if value is None:
        return "NULL"
    return str(value)


def generate_llm_summary(
    client: OpenAI,
    settings: Settings,
    question: str,
    sql: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    truncated: bool,
) -> str:
    payload = {
        "question": question,
        "sql": sql,
        "columns": list(columns),
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
    prompt = f"""Summarize the SQL result for the user's question in 1 to 3 short sentences.
Use only the returned table data. Do not invent causes, missing rows, or outside facts.
If the table is insufficient to answer the question, say that plainly.

Data:
{json.dumps(payload, default=str, ensure_ascii=False)}

Summary:"""
    response = client.chat.completions.create(
        model=settings.summary_model,
        messages=[
            {"role": "system", "content": "You write concise, grounded result summaries."},
            {"role": "user", "content": prompt},
        ],
        temperature=settings.summary_temperature,
        top_p=settings.summary_top_p,
        max_tokens=settings.max_summary_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def summarize_result(
    client: OpenAI,
    settings: Settings,
    question: str,
    sql: str,
    execution: ExecutionResult,
    expectation: ShapeExpectation | None = None,
) -> tuple[str, str, str | None]:
    fallback = deterministic_summary(
        question, execution.columns, execution.rows, execution.truncated, expectation
    )
    if not settings.enable_llm_summary:
        return fallback, "deterministic", None
    try:
        summary = generate_llm_summary(
            client,
            settings,
            question,
            sql,
            execution.columns,
            execution.rows,
            execution.truncated,
        )
        if not summary:
            return fallback, "deterministic", "LLM summary was empty."
        return summary, "llm", None
    except Exception as exc:
        return fallback, "deterministic", str(exc)


def log_query(settings: Settings, result: AnswerResult) -> None:
    if not settings.enable_query_logging:
        return
    with metadata_connection(settings.metadata_path) as conn:
        setup_metadata(conn)
        conn.execute(
            """
            INSERT INTO query_logs(
                question, generated_sql, initial_sql, success, executed, cancelled,
                error_message, validation_error, repair_reason, repaired,
                retrieved_objects, summary, summary_mode, summary_error, shape_warning,
                execution_ms, row_count, prompt_tokens, completion_tokens, total_tokens,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.question,
                result.sql,
                result.initial_sql,
                int(result.success),
                int(result.executed),
                int(result.cancelled),
                result.error_message,
                result.validation_error,
                result.repair_reason,
                int(result.repaired),
                json.dumps(result.retrieved_tables),
                result.summary,
                result.summary_mode,
                result.summary_error,
                result.shape_warning,
                result.execution_ms,
                len(result.rows),
                result.token_usage.prompt_tokens,
                result.token_usage.completion_tokens,
                result.token_usage.total_tokens,
                utc_now(),
            ),
        )


def add_usage(left: UsageStats, right: UsageStats) -> UsageStats:
    return UsageStats(
        prompt_tokens=sum_optional(left.prompt_tokens, right.prompt_tokens),
        completion_tokens=sum_optional(left.completion_tokens, right.completion_tokens),
        total_tokens=sum_optional(left.total_tokens, right.total_tokens),
    )


def sum_optional(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def validate_with_repair(
    client: OpenAI,
    settings: Settings,
    result: AnswerResult,
    tables: list[TableSchema],
    documents: list[SchemaDocument],
    expectation: ShapeExpectation,
    repair_budget: int,
) -> int:
    """Validate result.sql, repairing once if the budget allows. Returns the remaining budget."""
    try:
        validate_sql(result.sql, tables)
        return repair_budget
    except SqlValidationError as exc:
        result.validation_error = str(exc)
        if repair_budget <= 0:
            raise
        result.repair_reason = f"validation failed: {exc}"
        repair_prompt = build_validation_repair_prompt(
            result.question,
            all_schema_text(documents),
            result.sql,
            str(exc),
            settings.max_result_rows,
            expectation,
        )
        repaired = generate_sql(client, settings, repair_prompt)
        result.sql = repaired.sql
        result.repaired = True
        result.token_usage = add_usage(result.token_usage, repaired.usage)
        validate_sql(result.sql, tables)
        return repair_budget - 1


def check_shape_with_repair(
    client: OpenAI,
    settings: Settings,
    result: AnswerResult,
    tables: list[TableSchema],
    documents: list[SchemaDocument],
    expectation: ShapeExpectation,
    execution: ExecutionResult,
    repair_budget: int,
    approval_callback: Callable[[str], bool] | None,
) -> ExecutionResult | None:
    """Check result shape and repair once if the budget allows.

    Returns the execution to continue with, or None when a required approval was
    declined during the repaired re-execution (the caller should log and return).
    """
    shape = check_result_shape(
        result.question, result.sql or "", execution.columns, execution.rows, expectation
    )
    result.shape_expected = shape.expected
    result.shape_observed = shape.observed
    if shape.ok or repair_budget <= 0:
        result.shape_warning = shape.warning
        return execution

    original_warning = shape.warning
    repair_prompt = build_shape_repair_prompt(
        result.question,
        all_schema_text(documents),
        result.sql or "",
        shape.expected,
        shape.observed,
        settings.max_result_rows,
    )
    try:
        repaired = generate_sql(client, settings, repair_prompt)
        validate_sql(repaired.sql, tables)
        previous_sql = result.sql
        result.sql = repaired.sql
        result.repaired = True
        result.repair_reason = "result shape mismatch"
        result.token_usage = add_usage(result.token_usage, repaired.usage)
        repaired_execution = execute_with_optional_approval(settings, result, approval_callback)
        if repaired_execution is None:
            return None
        shape = check_result_shape(
            result.question,
            result.sql or "",
            repaired_execution.columns,
            repaired_execution.rows,
            expectation,
        )
        result.shape_expected = shape.expected
        result.shape_observed = shape.observed
        result.shape_warning = shape.warning
        if not shape.ok and previous_sql:
            result.initial_sql = result.initial_sql or previous_sql
        return repaired_execution
    except AppError as exc:
        result.shape_warning = original_warning
        result.error_message = f"Shape repair failed: {exc}"
        return execution


def answer_question(
    settings: Settings,
    client: OpenAI,
    question: str,
    *,
    approval_callback: Callable[[str], bool] | None = None,
) -> AnswerResult:
    result = AnswerResult(question=question, retrieved_tables=[], expanded_tables=[])
    try:
        tables = extract_schema(settings.source_db_path)
        auto_reindexed, freshness_warning = ensure_index_ready(settings, client, tables)
        result.auto_reindexed = auto_reindexed
        result.freshness_warning = freshness_warning
        documents = load_indexed_documents(settings)
        retrieved = retrieve_schema(settings, client, question)
        result.retrieved_tables = [item.table_name for item in retrieved]
        expanded = expanded_schema_order(retrieved, tables)
        result.expanded_tables = expanded
        expectation = infer_shape_expectation(question)
        prompt_schema = schema_text_for_tables(expanded, documents)
        prompt = build_sql_prompt(question, prompt_schema, settings.max_result_rows, expectation)
        generation = generate_sql(client, settings, prompt)
        result.sql = generation.sql
        result.initial_sql = generation.initial_sql
        result.token_usage = add_usage(result.token_usage, generation.usage)

        repair_budget = settings.max_repair_attempts
        repair_budget = validate_with_repair(
            client, settings, result, tables, documents, expectation, repair_budget
        )

        execution = execute_with_optional_approval(settings, result, approval_callback)
        if execution is None:
            log_query(settings, result)
            return result

        if settings.enable_result_shape_check:
            execution_or_cancel = check_shape_with_repair(
                client,
                settings,
                result,
                tables,
                documents,
                expectation,
                execution,
                repair_budget,
                approval_callback,
            )
            if execution_or_cancel is None:
                log_query(settings, result)
                return result
            execution = execution_or_cancel

        result.columns = execution.columns
        result.rows = execution.rows
        result.truncated = execution.truncated
        result.execution_ms = execution.execution_ms
        result.summary, result.summary_mode, result.summary_error = summarize_result(
            client, settings, question, result.sql or "", execution, expectation
        )
        result.success = True
        result.executed = True
        log_query(settings, result)
        return result
    except AppError as exc:
        result.error_message = str(exc)
        log_query(settings, result)
        return result
    except Exception as exc:
        result.error_message = f"Unexpected error: {exc}"
        log_query(settings, result)
        return result


def execute_with_optional_approval(
    settings: Settings,
    result: AnswerResult,
    approval_callback: Callable[[str], bool] | None,
) -> ExecutionResult | None:
    if settings.require_sql_approval:
        if approval_callback is None:
            raise QueryExecutionError("SQL approval is required but no approval callback was provided.")
        approved = approval_callback(result.sql or "")
        if not approved:
            result.cancelled = True
            result.executed = False
            result.success = False
            result.error_message = "User cancelled execution."
            return None
    return execute_readonly_query(
        settings.source_db_path,
        result.sql or "",
        max_rows=settings.max_result_rows,
        timeout_ms=settings.query_timeout_ms,
    )


def read_query_logs(settings: Settings, limit: int) -> list[dict[str, Any]]:
    if not settings.metadata_path.exists():
        return []
    with metadata_connection(settings.metadata_path) as conn:
        conn.row_factory = sqlite3.Row
        setup_metadata(conn)
        rows = conn.execute(
            """
            SELECT id, question, generated_sql, initial_sql, success, executed, cancelled,
                   error_message, validation_error, repair_reason, repaired,
                   retrieved_objects, summary, summary_mode, summary_error,
                   shape_warning, execution_ms, row_count, created_at
            FROM query_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def download_northwind(settings: Settings, *, force: bool = False) -> tuple[Path, int]:
    target = settings.source_db_path
    if target.exists() and not force:
        raise AppError(f"{target} already exists. Use --force to overwrite it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(NORTHWIND_DOWNLOAD_URL, timeout=60) as response:
            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        table_count = verify_northwind_database(temp_path)
        os.replace(temp_path, target)
        return target, table_count
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def verify_sqlite_database(path: Path) -> int:
    """Validate that ``path`` is a readable SQLite database with tables or views.

    Distinguishes the failure modes so the user gets an actionable message:
    a missing file, a file that is not a readable SQLite database, and a valid
    but empty database are reported differently. Returns the count of user
    tables and views (both are valid read sources since v2.4).
    """
    if not path.exists():
        raise AppError(f"Database not found: {path}")
    try:
        with sqlite3.connect(sqlite_readonly_uri(path), uri=True) as conn:
            objects = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
                    """
                )
            ]
    except sqlite3.Error as exc:
        raise AppError(f"Not a readable SQLite database: {path} ({exc})") from exc
    if not objects:
        raise AppError(f"No user tables or views found in SQLite database: {path}")
    return len(objects)


def verify_northwind_database(path: Path) -> int:
    verify_sqlite_database(path)
    try:
        with sqlite3.connect(sqlite_readonly_uri(path), uri=True) as conn:
            rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise AppError(f"Downloaded file is not a readable SQLite database: {exc}") from exc
    tables = {row[0] for row in rows}
    missing = NORTHWIND_EXPECTED_TABLES - tables
    if missing:
        raise AppError(
            "Downloaded database did not contain expected Northwind tables: "
            + ", ".join(sorted(missing))
        )
    return len(tables)
