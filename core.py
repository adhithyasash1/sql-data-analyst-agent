from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
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
class ColumnStat:
    name: str
    dtype: str
    null_count: int
    distinct_count: int
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None


@dataclass(frozen=True)
class DataFrameProfile:
    row_count: int
    column_count: int
    columns: tuple[ColumnStat, ...]
    profiled_truncated: bool = False


@dataclass(frozen=True)
class ResultArtifact:
    artifact_id: int
    question: str
    sql: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    analysis_text: str | None
    created_at: str


@dataclass(frozen=True)
class AnalysisArtifactTable:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class AnalysisArtifact:
    analysis_id: int
    source_artifact_id: int
    recipe: str
    status: str
    title: str
    summary: str
    tables: tuple[AnalysisArtifactTable, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    created_at: str = ""


@dataclass(frozen=True)
class ExportResult:
    path: Path
    row_count: int
    truncated: bool


@dataclass(frozen=True)
class ChartResult:
    path: Path
    chart_type: str
    x_column: str | None
    y_column: str | None
    row_count: int  # number of points actually plotted (after skipping missing/invalid)


@dataclass(frozen=True)
class RoutedArtifactCommand:
    command: str  # describe, head, tail, export, sql, columns, artifacts, plot
    arg: str  # e.g. "" / "5" / "csv" / "bar x=GenreName y=TotalRevenue"
    reason: str  # short tag for logging/UX, e.g. "head-count", "plot-hist"


@dataclass(frozen=True)
class ArtifactTransformResult:
    artifact: ResultArtifact
    operation: str
    row_count: int


@dataclass(frozen=True)
class WorkspaceSaveResult:
    path: Path
    artifact_count: int
    analysis_count: int = 0


@dataclass(frozen=True)
class WorkspaceLoadResult:
    artifacts: tuple[ResultArtifact, ...]
    path: Path
    analyses: tuple[AnalysisArtifact, ...] = ()


@dataclass(frozen=True)
class WorkspaceInfo:
    path: Path
    name: str
    artifact_count: int
    row_count: int
    created_at: str | None
    files: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceDeleteResult:
    path: Path
    name: str


@dataclass(frozen=True)
class ReportExportResult:
    path: Path
    artifact_count: int
    format: str
    analysis_count: int = 0



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
    analysis: DataFrameProfile | None = None
    analysis_text: str | None = None
    analysis_error: str | None = None


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


def text_has_intent_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


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

    temporal_order_desc = any(
        phrase in text for phrase in ("most recent", "latest", "newest", "recent")
    )
    temporal_order_asc = any(phrase in text for phrase in ("oldest", "earliest"))
    temporal_order = temporal_order_desc or temporal_order_asc

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
    requires_numeric = (
        any(text_has_intent_phrase(text, word) for word in numeric_words) and not temporal_order
    )
    order_words = ("most", "least", "top", "highest", "lowest", "largest", "smallest")
    ranked_order = (
        any(text_has_intent_phrase(text, word) for word in order_words) and not temporal_order
    )
    requires_order = ranked_order or temporal_order
    order_direction = None
    if requires_order:
        order_direction = (
            "ASC"
            if temporal_order_asc
            or any(text_has_intent_phrase(text, word) for word in ("least", "lowest", "smallest"))
            else "DESC"
        )

    aggregate_kind = None
    if any(text_has_intent_phrase(text, word) for word in ("average", "avg", "mean")):
        aggregate_kind = "average"
    elif any(text_has_intent_phrase(text, word) for word in ("total", "sum")):
        aggregate_kind = "total"
    elif question_has_count_intent(question) or ranked_order:
        aggregate_kind = "count"

    expected_limit = None
    limit_patterns = (
        r"\btop\s+(\d{1,3})\b",
        r"\bfirst\s+(\d{1,3})\b",
        r"\blimit\s+(\d{1,3})\b",
        r"\b(?:list|show)\s+(?:the\s+)?(\d{1,3})\b",
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
    if expectation is None:
        expectation = infer_shape_expectation(question)
    if is_detail_listing_summary(question, expectation):
        return f"Returned {row_count} row{'s' if row_count != 1 else ''}."

    numeric_indexes = [
        index
        for index, column in enumerate(columns)
        if column_types.get(column) == "numeric" and is_metric_column(column)
    ]
    if rows and len(columns) >= 2 and numeric_indexes:
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


def is_detail_listing_summary(question: str, expectation: ShapeExpectation) -> bool:
    text = question.lower()
    starts_like_detail = text.startswith(("list", "show", "give me"))
    temporal_order = any(
        phrase in text
        for phrase in ("most recent", "latest", "newest", "recent", "oldest", "earliest")
    )
    return (starts_like_detail or temporal_order) and not expectation.requires_numeric


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


def _require_dataframe_libs() -> tuple[Any, Any]:
    """Lazy-import pyarrow + pandas, raising a friendly error naming the optional extra."""
    try:
        import pandas as pd
        import pyarrow as pa
    except ImportError as exc:
        raise AppError(
            "Result analysis needs the optional analysis extra. "
            "Install it with: uv sync --extra analysis"
        ) from exc
    return pa, pd


def make_unique_column_names(columns: tuple[str, ...]) -> tuple[str, ...]:
    """De-duplicate column names for the internal Arrow/pandas frame.

    Original display names are kept elsewhere (ColumnStat.name); this only protects the
    Arrow table / pandas conversion from duplicate result columns, e.g.
    ("count", "count") -> ("count", "count__2").
    """
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in columns:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}__{seen[name]}")
        else:
            seen[name] = 1
            unique.append(name)
    return tuple(unique)


def result_to_arrow_table(columns: tuple[str, ...], rows: tuple[tuple[Any, ...], ...]) -> Any:
    """Build a pyarrow.Table from a validated result, column by column.

    SQLite is dynamically typed, so a column can hold mixed types; on a type-inference
    failure that column falls back to a string array with NULLs preserved as null.
    """
    pa, _ = _require_dataframe_libs()
    names = make_unique_column_names(columns)
    arrays = []
    for index in range(len(columns)):
        values = [row[index] for row in rows]
        try:
            arrays.append(pa.array(values))
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
            arrays.append(
                pa.array(
                    [None if value is None else str(value) for value in values],
                    type=pa.string(),
                )
            )
    return pa.Table.from_arrays(arrays, names=list(names))


def _profile_number(value: Any) -> float | None:
    """Coerce a numeric stat to a plain float, dropping NaN (NaN != NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def profile_result_dataframe(
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    profiled_truncated: bool = False,
) -> DataFrameProfile:
    """Compute a deterministic per-column profile via pyarrow -> pandas."""
    _, _pd = _require_dataframe_libs()
    from pandas.api.types import is_numeric_dtype

    frame = result_to_arrow_table(columns, rows).to_pandas()
    stats: list[ColumnStat] = []
    for index, name in enumerate(columns):
        series = frame.iloc[:, index]
        null_count = int(series.isna().sum())
        distinct_count = int(series.nunique(dropna=True))
        minimum = maximum = mean = None
        if is_numeric_dtype(series) and null_count < len(series):
            minimum = _profile_number(series.min())
            maximum = _profile_number(series.max())
            mean = _profile_number(series.mean())
        stats.append(
            ColumnStat(
                name=name,
                dtype=str(series.dtype),
                null_count=null_count,
                distinct_count=distinct_count,
                minimum=minimum,
                maximum=maximum,
                mean=mean,
            )
        )
    return DataFrameProfile(
        row_count=len(frame),
        column_count=len(columns),
        columns=tuple(stats),
        profiled_truncated=profiled_truncated,
    )


def _format_stat_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    rounded = round(value, 4)
    return str(int(rounded)) if rounded == int(rounded) else str(rounded)


def dataframe_profile_text(profile: DataFrameProfile, max_columns: int) -> str:
    """Render a compact, deterministic, column-capped profile for panel + LLM grounding."""
    lead = f"Rows profiled: {profile.row_count}, Columns: {profile.column_count}"
    if profile.profiled_truncated:
        lead += " (analysis row cap reached; more rows may exist)"
    lines = [lead]
    shown = profile.columns[:max_columns]
    for stat in shown:
        parts = [
            f"- {stat.name}: {stat.dtype}",
            f"nulls={stat.null_count}",
            f"distinct={stat.distinct_count}",
        ]
        if stat.minimum is not None or stat.maximum is not None or stat.mean is not None:
            parts.append(f"min={_format_stat_number(stat.minimum)}")
            parts.append(f"max={_format_stat_number(stat.maximum)}")
            parts.append(f"mean={_format_stat_number(stat.mean)}")
        lines.append(", ".join(parts))
    omitted = profile.column_count - len(shown)
    if omitted > 0:
        lines.append(f"... {omitted} more columns omitted")
    return "\n".join(lines)


def analyze_execution(
    settings: Settings, sql: str, execution: ExecutionResult
) -> tuple[DataFrameProfile, str]:
    """Profile a successful result. Reuses the displayed rows unless they were truncated,
    in which case it performs one bounded, read-only re-fetch up to max_analysis_rows.
    """
    columns, rows = execution.columns, execution.rows
    profiled_truncated = execution.truncated
    if execution.truncated and settings.max_analysis_rows > len(rows):
        bigger = execute_readonly_query(
            settings.source_db_path,
            sql,
            max_rows=settings.max_analysis_rows,
            timeout_ms=settings.query_timeout_ms,
        )
        columns, rows = bigger.columns, bigger.rows
        profiled_truncated = bigger.truncated
    profile = profile_result_dataframe(columns, rows, profiled_truncated=profiled_truncated)
    return profile, dataframe_profile_text(profile, settings.max_analysis_columns)


def make_result_artifact(artifact_id: int, result: AnswerResult) -> ResultArtifact:
    """Snapshot a successful result for in-session reuse. Defends itself even though the
    interactive loop only calls it for successful results with SQL and columns."""
    if not result.sql:
        raise AppError("Cannot create an artifact without SQL.")
    if not result.columns:
        raise AppError("Cannot create an artifact without result columns.")
    return ResultArtifact(
        artifact_id=artifact_id,
        question=result.question,
        sql=result.sql,
        columns=result.columns,
        rows=result.rows,
        truncated=result.truncated,
        analysis_text=result.analysis_text,
        created_at=utc_now(),
    )


def parse_colon_command(text: str) -> tuple[str, str] | None:
    """Parse a ``:command [arg]`` line. Returns ``(name_lower, arg)`` or None for non-commands."""
    stripped = text.strip()
    if not stripped.startswith(":"):
        return None
    body = stripped[1:].strip()
    if not body:
        return None
    name, _, arg = body.partition(" ")
    return name.lower(), arg.strip()


def parse_count(arg: str, default: int = 10) -> int:
    """Parse a positive row count for :head/:tail; reject bad input rather than silently default."""
    if not arg:
        return default
    try:
        value = int(arg)
    except ValueError as exc:
        raise AppError("Count must be a positive integer.") from exc
    if value < 1:
        raise AppError("Count must be at least 1.")
    return value


def artifact_preview_rows(
    artifact: ResultArtifact, mode: str, n: int
) -> tuple[tuple[Any, ...], ...]:
    if mode not in {"head", "tail"}:
        raise AppError("Preview mode must be 'head' or 'tail'.")
    count = max(1, n)
    return artifact.rows[-count:] if mode == "tail" else artifact.rows[:count]


def artifact_describe_text(settings: Settings, artifact: ResultArtifact) -> str:
    """Describe the artifact snapshot. Reuses v3.0 analysis_text when present (no pandas import);
    otherwise computes a profile from the stored rows (needs the analysis extra). Never re-runs SQL."""
    if artifact.analysis_text:
        return artifact.analysis_text
    profile = profile_result_dataframe(
        artifact.columns, artifact.rows, profiled_truncated=artifact.truncated
    )
    return dataframe_profile_text(profile, settings.max_analysis_columns)


def materialize_artifact_rows(
    settings: Settings, artifact: ResultArtifact
) -> tuple[tuple[str, ...], tuple[tuple[Any, ...], ...], bool]:
    """Rows to export: the displayed rows when complete, else one bounded read-only re-fetch
    (up to max_analysis_rows) returning the re-fetch's own columns + rows."""
    if not artifact.truncated:
        return artifact.columns, artifact.rows, False
    if not artifact.sql:
        raise AppError("Cannot export the full result because the SQL is unavailable.")
    bigger = execute_readonly_query(
        settings.source_db_path,
        artifact.sql,
        max_rows=settings.max_analysis_rows,
        timeout_ms=settings.query_timeout_ms,
    )
    return bigger.columns, bigger.rows, bigger.truncated


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary>"
    return value


def _next_output_path(output_dir: Path, stem: str = "result", suffix: str = ".csv") -> Path:
    for index in range(1, 10000):
        candidate = output_dir / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
    raise AppError("Could not find a free output filename in the output directory.")


def export_artifact_csv(
    settings: Settings, artifact: ResultArtifact, output_dir: Path
) -> ExportResult:
    columns, rows, truncated = materialize_artifact_rows(settings, artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _next_output_path(output_dir)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([_csv_cell(value) for value in row])
    return ExportResult(path=path, row_count=len(rows), truncated=truncated)


# --- v3.6: persistent artifact workspace -----------------------------------------------

WORKSPACE_FORMAT_VERSION = 1


def _app_version() -> str:
    """Best-effort package version for the workspace manifest (informational only)."""
    try:
        from importlib.metadata import version

        return version("sql-data-analyst-agent")
    except Exception:
        return "unknown"


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "<binary>"
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _analysis_artifact_to_dict(analysis: AnalysisArtifact) -> dict[str, Any]:
    metrics = _json_safe_value(analysis.metrics)
    return {
        "analysis_id": analysis.analysis_id,
        "source_artifact_id": analysis.source_artifact_id,
        "recipe": analysis.recipe,
        "status": analysis.status,
        "title": analysis.title,
        "summary": analysis.summary,
        "metrics": metrics if isinstance(metrics, dict) else {},
        "warnings": list(analysis.warnings),
        "created_at": analysis.created_at,
        "tables": [
            {
                "title": table.title,
                "columns": list(table.columns),
                "rows": [
                    [_json_safe_value(value) for value in row]
                    for row in table.rows
                ],
            }
            for table in analysis.tables
        ],
    }


def _analysis_artifact_from_dict(data: dict[str, Any]) -> AnalysisArtifact:
    tables_data = data.get("tables", [])
    if not isinstance(tables_data, list):
        raise AppError("Invalid analysis artifact: tables must be a list.")

    tables: list[AnalysisArtifactTable] = []
    for table_data in tables_data:
        if not isinstance(table_data, dict):
            raise AppError("Invalid analysis artifact: table entry must be an object.")
        columns_data = table_data.get("columns", [])
        rows_data = table_data.get("rows", [])
        if not isinstance(columns_data, list) or not isinstance(rows_data, list):
            raise AppError("Invalid analysis artifact: table columns and rows must be lists.")
        rows = tuple(
            tuple(row if isinstance(row, list) else [row])
            for row in rows_data
        )
        tables.append(
            AnalysisArtifactTable(
                title=str(table_data.get("title", "Result Table")),
                columns=tuple(str(column) for column in columns_data),
                rows=rows,
            )
        )

    metrics = data.get("metrics", {})
    warnings = data.get("warnings", [])
    return AnalysisArtifact(
        analysis_id=int(data.get("analysis_id", 0) or 0),
        source_artifact_id=int(data.get("source_artifact_id", 0) or 0),
        recipe=str(data.get("recipe", "")),
        status=str(data.get("status", "")),
        title=str(data.get("title", "Analysis Result")),
        summary=str(data.get("summary", "")),
        tables=tuple(tables),
        metrics=metrics if isinstance(metrics, dict) else {},
        warnings=tuple(str(warning) for warning in warnings) if isinstance(warnings, list) else (),
        created_at=str(data.get("created_at", "")),
    )


def _escape_md_cell(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, (bytes, bytearray, memoryview)):
        return "<binary>"
    return str(val).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table_lines(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    if not columns:
        return ["No columns."]
    headers = " | ".join(_escape_md_cell(column) for column in columns)
    sep = " | ".join("---" for _ in columns)
    lines = [f"| {headers} |", f"| {sep} |"]
    for row in rows:
        row_str = " | ".join(_escape_md_cell(value) for value in row)
        lines.append(f"| {row_str} |")
    return lines


def _analysis_artifact_markdown_lines(
    analysis: AnalysisArtifact,
    *,
    heading_level: int = 1,
) -> list[str]:
    heading = "#" * heading_level
    subheading = "#" * (heading_level + 1)
    lines = [
        f"{heading} Analysis #{analysis.analysis_id}: {analysis.title}",
        "",
        f"- **Source Artifact**: #{analysis.source_artifact_id}",
        f"- **Created**: {analysis.created_at}",
        f"- **Recipe**: {analysis.recipe}",
        f"- **Status**: {analysis.status}",
        "",
    ]
    if analysis.summary:
        lines.extend([analysis.summary, ""])
    if analysis.metrics:
        lines.extend([f"{subheading} Metrics", ""])
        rows = tuple((key, value) for key, value in analysis.metrics.items())
        lines.extend(_markdown_table_lines(("Field", "Value"), rows))
        lines.append("")
    if analysis.warnings:
        lines.extend([f"{subheading} Warnings", ""])
        lines.extend(f"- {warning}" for warning in analysis.warnings)
        lines.append("")
    for table in analysis.tables:
        lines.extend([f"{subheading} {table.title}", ""])
        lines.extend(_markdown_table_lines(table.columns, table.rows))
        lines.append("")
    return lines


def _analysis_artifact_markdown(analysis: AnalysisArtifact) -> str:
    return "\n".join(_analysis_artifact_markdown_lines(analysis)).rstrip() + "\n"


def sanitize_workspace_name(name: str) -> str:
    """Reduce a user-supplied workspace name to a filesystem-safe stem.

    Letters/digits/dash/underscore are kept; every other character becomes ``_``. A name with no
    alphanumeric character (empty, blank, or all punctuation) is rejected.
    """
    safe = "".join(char if (char.isalnum() or char in {"-", "_"}) else "_" for char in name.strip())
    if not any(char.isalnum() for char in safe):
        raise AppError("Workspace name must contain a letter or number.")
    return safe


def _next_workspace_path(base_dir: Path, name: str | None = None) -> Path:
    """Pick a fresh workspace directory path, adding a numeric suffix on same-second collisions."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    stem = f"{sanitize_workspace_name(name)}_{timestamp}" if name else f"session_{timestamp}"
    candidate = base_dir / stem
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        alternate = base_dir / f"{stem}_{index}"
        if not alternate.exists():
            return alternate
    raise AppError("Could not find a free workspace directory name.")


def save_artifact_workspace(
    settings: Settings,
    artifacts: Sequence[ResultArtifact],
    name: str | None = None,
    analyses: Sequence[AnalysisArtifact] = (),
) -> WorkspaceSaveResult:
    """Persist session artifacts and executed analysis summaries under OUTPUT_DIR/workspaces."""
    if not artifacts:
        raise AppError("No artifacts to save.")
    workspaces_dir = settings.output_path / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)
    workspace_path = _next_workspace_path(workspaces_dir, name)
    workspace_path.mkdir(parents=True, exist_ok=False)

    entries: list[dict[str, Any]] = []
    artifact_ids = {artifact.artifact_id for artifact in artifacts}
    for index, artifact in enumerate(artifacts, start=1):
        csv_file = f"artifact_{index:03d}.csv"
        sql_file = f"artifact_{index:03d}.sql"
        profile_file = f"artifact_{index:03d}_profile.txt" if artifact.analysis_text else None

        with (workspace_path / csv_file).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(artifact.columns))
            for row in artifact.rows:
                writer.writerow([_csv_cell(value) for value in row])
        (workspace_path / sql_file).write_text(artifact.sql, encoding="utf-8")
        if profile_file is not None:
            (workspace_path / profile_file).write_text(artifact.analysis_text or "", encoding="utf-8")

        entries.append(
            {
                "artifact_id": artifact.artifact_id,
                "question": artifact.question,
                "sql_file": sql_file,
                "csv_file": csv_file,
                "profile_file": profile_file,
                "row_count": len(artifact.rows),
                "column_count": len(artifact.columns),
                "columns": list(artifact.columns),
                "truncated": artifact.truncated,
                "created_at": artifact.created_at,
            }
        )

    linked_analyses = tuple(
        analysis for analysis in analyses if analysis.source_artifact_id in artifact_ids
    )
    analysis_entries: list[dict[str, Any]] = []
    for index, analysis in enumerate(linked_analyses, start=1):
        json_file = f"analysis_{index:03d}.json"
        markdown_file = f"analysis_{index:03d}.md"
        (workspace_path / json_file).write_text(
            json.dumps(_analysis_artifact_to_dict(analysis), indent=2),
            encoding="utf-8",
        )
        (workspace_path / markdown_file).write_text(
            _analysis_artifact_markdown(analysis),
            encoding="utf-8",
        )
        analysis_entries.append(
            {
                "analysis_id": analysis.analysis_id,
                "source_artifact_id": analysis.source_artifact_id,
                "recipe": analysis.recipe,
                "status": analysis.status,
                "title": analysis.title,
                "json_file": json_file,
                "markdown_file": markdown_file,
                "created_at": analysis.created_at,
            }
        )

    manifest = {
        "format_version": WORKSPACE_FORMAT_VERSION,
        "app_version": _app_version(),
        "created_at": utc_now(),
        "artifact_count": len(artifacts),
        "analysis_count": len(linked_analyses),
        "artifacts": entries,
        "analyses": analysis_entries,
    }
    (workspace_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return WorkspaceSaveResult(
        path=workspace_path,
        artifact_count=len(artifacts),
        analysis_count=len(linked_analyses),
    )


def list_saved_workspaces(settings: Settings) -> tuple[Path, ...]:
    """Workspace directories (those containing manifest.json) under OUTPUT_DIR/workspaces, newest first."""
    workspaces_dir = settings.output_path / "workspaces"
    if not workspaces_dir.exists():
        return ()
    dirs = [
        path
        for path in workspaces_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    dirs.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return tuple(dirs)


def resolve_workspace_target(workspaces_dir: Path, target: str) -> Path:
    """Resolve a bare workspace name (exact, else unique prefix) safely under workspaces_dir.

    Absolute paths, path separators, and ``..`` are rejected, so resolution can never escape the
    workspaces directory.
    """
    target = target.strip()
    if not target:
        raise AppError("Usage: :load <workspace>.")
    if Path(target).is_absolute():
        raise AppError("Workspace must be a name under the workspaces directory, not an absolute path.")
    if "/" in target or "\\" in target or ".." in target:
        raise AppError("Workspace must be a bare name under the workspaces directory.")
    if not workspaces_dir.exists():
        raise AppError(f"No workspace found: {target}")
    candidates = [
        path
        for path in workspaces_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ]
    exact = workspaces_dir / target
    if exact.is_dir() and (exact / "manifest.json").is_file():
        return exact
    matches = sorted(
        (path for path in candidates if path.name.startswith(target)), key=lambda path: path.name
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise AppError(f"No workspace found: {target}")
    listed = "\n".join(f"- {path.name}" for path in matches)
    raise AppError(f"Multiple workspaces match {target}. Use one of:\n{listed}")


def load_artifact_workspace(settings: Settings, target: str) -> WorkspaceLoadResult:
    """Reconstruct artifacts from a workspace. Loaded CSV cells are strings because CSV is untyped."""
    workspaces_dir = settings.output_path / "workspaces"
    workspace_path = resolve_workspace_target(workspaces_dir, target)

    try:
        manifest = json.loads((workspace_path / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise AppError(f"Could not read workspace manifest: {exc}") from exc
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise AppError("Invalid workspace manifest: missing artifacts list.")

    artifacts: list[ResultArtifact] = []
    for entry in entries:
        csv_file = entry.get("csv_file")
        if not csv_file:
            raise AppError("Invalid workspace manifest: artifact entry missing csv_file.")
        try:
            with (workspace_path / csv_file).open(newline="", encoding="utf-8") as handle:
                table = list(csv.reader(handle))
            sql_text = (workspace_path / entry["sql_file"]).read_text(encoding="utf-8")
            profile_file = entry.get("profile_file")
            analysis_text = (
                (workspace_path / profile_file).read_text(encoding="utf-8")
                if profile_file
                else None
            )
        except (OSError, KeyError) as exc:
            raise AppError(f"Could not read workspace artifact files: {exc}") from exc

        columns = tuple(table[0]) if table else ()
        rows = tuple(tuple(row) for row in table[1:])
        if any(len(row) != len(columns) for row in rows):
            raise AppError(
                f"Invalid workspace artifact CSV {csv_file}: row width does not match header."
            )
        artifacts.append(
            ResultArtifact(
                artifact_id=entry.get("artifact_id", len(artifacts) + 1),
                question=entry.get("question", ""),
                sql=sql_text,
                columns=columns,
                rows=rows,
                truncated=bool(entry.get("truncated", False)),
                analysis_text=analysis_text,
                created_at=entry.get("created_at", ""),
            )
        )

    analysis_entries = manifest.get("analyses", [])
    if not isinstance(analysis_entries, list):
        raise AppError("Invalid workspace manifest: analyses must be a list.")

    analyses: list[AnalysisArtifact] = []
    for entry in analysis_entries:
        if not isinstance(entry, dict):
            raise AppError("Invalid workspace manifest: analysis entry must be an object.")
        json_file = entry.get("json_file")
        if not json_file:
            raise AppError("Invalid workspace manifest: analysis entry missing json_file.")
        try:
            data = json.loads((workspace_path / json_file).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise AppError("Invalid analysis artifact: JSON root must be an object.")
            analyses.append(_analysis_artifact_from_dict(data))
        except AppError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise AppError(f"Could not read workspace analysis files: {exc}") from exc

    return WorkspaceLoadResult(
        artifacts=tuple(artifacts),
        path=workspace_path,
        analyses=tuple(analyses),
    )


def inspect_workspace(settings, target: str) -> WorkspaceInfo:
    try:
        workspaces_dir = settings.output_path / "workspaces"
        workspace_path = resolve_workspace_target(workspaces_dir, target)
        
        manifest_path = workspace_path / "manifest.json"
        try:
            manifest_content = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(manifest_content)
        except Exception as exc:
            raise AppError(f"Could not read workspace manifest: {exc}") from exc
        
        artifacts = manifest.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise AppError("Invalid workspace manifest: artifacts must be a list.")
            
        artifact_count = int(manifest.get("artifact_count") or len(artifacts))
        row_count = sum(int(entry.get("row_count") or 0) for entry in artifacts)
        created_at = manifest.get("created_at")
        
        files = tuple(sorted(p.name for p in workspace_path.iterdir() if p.is_file()))
        
        return WorkspaceInfo(
            path=workspace_path,
            name=workspace_path.name,
            artifact_count=artifact_count,
            row_count=row_count,
            created_at=created_at,
            files=files,
        )
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"Error inspecting workspace: {exc}") from exc


def delete_workspace(settings, target: str) -> WorkspaceDeleteResult:
    try:
        workspaces_dir = settings.output_path / "workspaces"
        workspace_path = resolve_workspace_target(workspaces_dir, target)
        
        name = workspace_path.name
        try:
            shutil.rmtree(workspace_path)
        except OSError as exc:
            raise AppError(f"Could not delete workspace {name}: {exc}") from exc
            
        return WorkspaceDeleteResult(path=workspace_path, name=name)
    except AppError:
        raise
    except Exception as exc:
        raise AppError(f"Error deleting workspace: {exc}") from exc


# --- v3.2: deterministic chart artifacts -----------------------------------------------


_CHART_TYPES = ("bar", "line", "scatter", "hist")


def _require_viz_libs() -> Any:
    """Lazy-import matplotlib with a non-interactive backend, raising a friendly error.

    Forces the Agg backend before importing pyplot so chart rendering never needs a display
    and never opens a window. Mirrors _require_dataframe_libs for the optional analysis extra.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AppError("Chart export needs the viz extra: uv sync --extra viz") from exc
    return plt


def parse_key_value_args(arg: str) -> dict[str, str]:
    """Parse space-separated ``key=value`` pairs into a dict.

    Keys are lowercased; values keep their original casing (they are column names). An empty
    argument returns ``{}`` so the option-specific error fires later. Tokens without ``=`` and
    duplicate keys are rejected.
    """
    options: dict[str, str] = {}
    stripped = arg.strip()
    if not stripped:
        return options
    for token in stripped.split():
        if "=" not in token:
            raise AppError(f"Invalid option '{token}'. Use key=value, e.g. x=Name.")
        raw_key, _, raw_value = token.partition("=")
        key = raw_key.strip().lower()
        value = raw_value.strip()
        if not key or not value:
            raise AppError(f"Invalid option '{token}'. Use key=value, e.g. x=Name.")
        if key in options:
            raise AppError(f"Duplicate option: {key}")
        options[key] = value
    return options


def parse_plot_command_args(arg: str) -> tuple[str, dict[str, str]]:
    """Split a ``:plot`` argument into ``(chart_type, options)``.

    Examples: ``"bar x=Name y=Revenue"`` -> ``("bar", {"x": "Name", "y": "Revenue"})``;
    ``"hist column=Revenue"`` -> ``("hist", {"column": "Revenue"})``.
    """
    chart_type, _, rest = arg.strip().partition(" ")
    chart_type = chart_type.strip().lower()
    if not chart_type:
        raise AppError("Missing chart type. Use :plot bar x=<column> y=<column>.")
    if chart_type not in _CHART_TYPES:
        raise AppError(f"Unsupported chart type: {chart_type}")
    return chart_type, parse_key_value_args(rest)


def resolve_column(columns: tuple[str, ...], requested: str) -> str:
    """Resolve a requested column to an actual artifact column name.

    Exact match wins; otherwise a unique case-insensitive match is used. No match raises
    "Unknown column"; multiple case-insensitive matches raise "Ambiguous column".
    """
    if requested in columns:
        return requested
    lowered = requested.lower()
    matches = [name for name in columns if name.lower() == lowered]
    if not matches:
        raise AppError(f"Unknown column: {requested}")
    if len(matches) > 1:
        raise AppError(f"Ambiguous column: {requested}")
    return matches[0]


def artifact_rows_as_dicts(artifact: ResultArtifact) -> list[dict[str, object]]:
    """Convert an artifact's columns + rows into row dicts keyed by unique column names.

    Reuses make_unique_column_names so duplicate result columns get deterministic keys
    (e.g. "count", "count__2"); resolve_column is given the same names so lookups match.
    """
    names = make_unique_column_names(artifact.columns)
    return [dict(zip(names, row, strict=True)) for row in artifact.rows]


# --- v3.4: controlled artifact transformations -----------------------------------------

_FILTER_OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "contains")
_GROUPBY_AGGS = ("sum", "mean", "count", "min", "max")
_NUMERIC_AGGS = ("sum", "mean", "min", "max")


def parse_transform_args(arg: str) -> dict[str, str]:
    """Parse ``key=value`` transformation options. Thin wrapper over parse_key_value_args."""
    return parse_key_value_args(arg)


def _require_keys(
    options: dict[str, str],
    allowed: set[str],
    *,
    operation: str,
    required: set[str] | None = None,
) -> None:
    """Reject unknown option keys and (optionally) flag missing required ones."""
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise AppError(f"Unknown option(s) for {operation}: {', '.join(unknown)}")
    missing = sorted((required or set()) - set(options))
    if missing:
        raise AppError(f"Missing required option(s) for {operation}: {', '.join(missing)}")


def _rows_from_dicts(
    dict_rows: list[dict[str, object]], columns: Sequence[str]
) -> tuple[tuple[object, ...], ...]:
    """Project dict rows back into ordered tuple rows for the given columns."""
    return tuple(tuple(row[column] for column in columns) for row in dict_rows)


def _compare_values(cell: object, op: str, raw_value: str) -> bool:
    """Deterministic comparison for :filter. No eval/exec; numeric when both sides convert."""
    if op == "contains":
        haystack = ("" if cell is None else str(cell)).lower()
        return raw_value.lower() in haystack
    if op in {"eq", "ne"}:
        cell_number = _chart_numeric(cell)
        value_number = _chart_numeric(raw_value)
        if cell_number is not None and value_number is not None:
            equal = cell_number == value_number
        else:
            equal = ("" if cell is None else str(cell)) == raw_value
        return equal if op == "eq" else not equal
    if op in {"gt", "gte", "lt", "lte"}:
        cell_number = _chart_numeric(cell)
        value_number = _chart_numeric(raw_value)
        if cell_number is None or value_number is None:
            return False
        if op == "gt":
            return cell_number > value_number
        if op == "gte":
            return cell_number >= value_number
        if op == "lt":
            return cell_number < value_number
        return cell_number <= value_number
    raise AppError(f"Unsupported filter op: {op}")


def _transformed_artifact(
    artifact_id: int,
    question: str,
    source: ResultArtifact,
    columns: Sequence[str],
    rows: tuple[tuple[object, ...], ...],
    truncated: bool,
) -> ResultArtifact:
    """Build a new in-session artifact from a transformation (no analysis, fresh timestamp)."""
    return ResultArtifact(
        artifact_id=artifact_id,
        question=question,
        sql=source.sql,
        columns=tuple(columns),
        rows=rows,
        truncated=truncated,
        analysis_text=None,
        created_at=utc_now(),
    )


def transform_artifact_sort(
    artifact: ResultArtifact, arg: str, artifact_id: int
) -> ArtifactTransformResult:
    """Sort the latest artifact by a column; missing values always last (both directions)."""
    options = parse_transform_args(arg)
    _require_keys(options, {"column", "order"}, operation="sort", required={"column"})
    order = options.get("order", "asc").lower()
    if order not in {"asc", "desc"}:
        raise AppError("Sort order must be 'asc' or 'desc'.")
    names = make_unique_column_names(artifact.columns)
    column = resolve_column(names, options["column"])
    dict_rows = artifact_rows_as_dicts(artifact)

    present = [row for row in dict_rows if row[column] is not None]
    missing = [row for row in dict_rows if row[column] is None]
    all_numeric = bool(present) and all(_chart_numeric(row[column]) is not None for row in present)
    if all_numeric:
        key = lambda row: _chart_numeric(row[column])  # noqa: E731
    else:
        key = lambda row: str(row[column])  # noqa: E731
    ordered = sorted(present, key=key, reverse=(order == "desc")) + missing

    rows = _rows_from_dicts(ordered, names)
    new = _transformed_artifact(
        artifact_id,
        f"{artifact.question} [sort {column} {order}]",
        artifact,
        names,
        rows,
        artifact.truncated,
    )
    return ArtifactTransformResult(new, "sort", len(rows))


def transform_artifact_select(
    artifact: ResultArtifact, arg: str, artifact_id: int
) -> ArtifactTransformResult:
    """Project the latest artifact down to a subset of columns, preserving requested order."""
    options = parse_transform_args(arg)
    _require_keys(options, {"columns"}, operation="select", required={"columns"})
    requested = [token.strip() for token in options["columns"].split(",") if token.strip()]
    if not requested:
        raise AppError("Select needs at least one column, e.g. columns=GenreName,TotalRevenue.")
    names = make_unique_column_names(artifact.columns)
    resolved = [resolve_column(names, token) for token in requested]
    dict_rows = artifact_rows_as_dicts(artifact)

    rows = _rows_from_dicts(dict_rows, resolved)
    new = _transformed_artifact(
        artifact_id,
        f"{artifact.question} [select {','.join(resolved)}]",
        artifact,
        resolved,
        rows,
        artifact.truncated,
    )
    return ArtifactTransformResult(new, "select", len(rows))


def transform_artifact_filter(
    artifact: ResultArtifact, arg: str, artifact_id: int
) -> ArtifactTransformResult:
    """Keep rows matching a deterministic comparison over one column."""
    options = parse_transform_args(arg)
    _require_keys(
        options, {"column", "op", "value"}, operation="filter", required={"column", "op", "value"}
    )
    op = options["op"].lower()
    if op not in _FILTER_OPS:
        raise AppError(f"Unsupported filter op: {op}")
    value = options["value"]
    if not value:
        raise AppError("Filter value must not be empty.")
    names = make_unique_column_names(artifact.columns)
    column = resolve_column(names, options["column"])
    dict_rows = artifact_rows_as_dicts(artifact)

    kept = [row for row in dict_rows if _compare_values(row[column], op, value)]
    rows = _rows_from_dicts(kept, names)
    note = " [filter applied to truncated artifact]" if artifact.truncated else ""
    new = _transformed_artifact(
        artifact_id,
        f"{artifact.question} [filter {column} {op} {value}]{note}",
        artifact,
        names,
        rows,
        False,
    )
    return ArtifactTransformResult(new, "filter", len(rows))


def transform_artifact_groupby(
    artifact: ResultArtifact, arg: str, artifact_id: int
) -> ArtifactTransformResult:
    """Group rows by one column and aggregate; deterministic group order by str(key)."""
    options = parse_transform_args(arg)
    _require_keys(options, {"by", "metric", "agg"}, operation="groupby", required={"by", "agg"})
    agg = options["agg"].lower()
    if agg not in _GROUPBY_AGGS:
        raise AppError(f"Unsupported aggregation: {agg}")
    if agg in _NUMERIC_AGGS and "metric" not in options:
        raise AppError(f"Aggregation '{agg}' needs a metric=<column> option.")
    names = make_unique_column_names(artifact.columns)
    by_column = resolve_column(names, options["by"])
    metric_column = resolve_column(names, options["metric"]) if "metric" in options else None
    dict_rows = artifact_rows_as_dicts(artifact)

    groups: dict[object, list[dict[str, object]]] = {}
    for row in dict_rows:
        groups.setdefault(row[by_column], []).append(row)

    if agg == "count":
        agg_column = "count"
    else:
        agg_column = f"{agg}_{metric_column}"

    rows_out: list[tuple[object, object]] = []
    for key in sorted(groups, key=lambda value: str(value)):
        members = groups[key]
        if agg == "count":
            if metric_column is None:
                aggregated: object = len(members)
            else:
                aggregated = sum(1 for row in members if row[metric_column] is not None)
        else:
            numbers = [
                number
                for row in members
                if (number := _chart_numeric(row[metric_column])) is not None
            ]
            if not numbers:
                aggregated = None
            elif agg == "sum":
                aggregated = sum(numbers)
            elif agg == "mean":
                aggregated = sum(numbers) / len(numbers)
            elif agg == "min":
                aggregated = min(numbers)
            else:
                aggregated = max(numbers)
        rows_out.append((key, aggregated))

    columns = (by_column, agg_column)
    rows = tuple(rows_out)
    note = " [groupby applied to truncated artifact]" if artifact.truncated else ""
    metric_text = f" {metric_column}" if metric_column is not None else ""
    new = _transformed_artifact(
        artifact_id,
        f"{artifact.question} [groupby {by_column} {agg}{metric_text}]{note}",
        artifact,
        columns,
        rows,
        False,
    )
    return ArtifactTransformResult(new, "groupby", len(rows))


def transform_artifact(
    artifact: ResultArtifact, operation: str, arg: str, artifact_id: int
) -> ArtifactTransformResult:
    """Dispatch a transformation command to its deterministic handler."""
    if operation == "sort":
        return transform_artifact_sort(artifact, arg, artifact_id)
    if operation == "select":
        return transform_artifact_select(artifact, arg, artifact_id)
    if operation == "filter":
        return transform_artifact_filter(artifact, arg, artifact_id)
    if operation == "groupby":
        return transform_artifact_groupby(artifact, arg, artifact_id)
    raise AppError(f"Unsupported transformation: {operation}")


def _next_chart_path(charts_dir: Path) -> Path:
    """Return the next free chart_NNN.png path; raise if none is available."""
    for index in range(1, 10000):
        candidate = charts_dir / f"chart_{index:03d}.png"
        if not candidate.exists():
            return candidate
    raise AppError("Could not find a free chart filename in the charts directory.")


def _chart_numeric(value: Any) -> float | None:
    """Coerce a value to a finite float for plotting, or None if it is not plottable.

    Bools are rejected (a bool is an int subclass; plotting True as 1.0 is surprising).
    None and empty/blank strings are treated as missing. Non-finite floats (NaN, +/-inf)
    are dropped. Numeric strings that float() can parse are accepted.
    """
    if isinstance(value, bool):
        return None
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    return number if math.isfinite(number) else None


def export_artifact_chart(settings: Settings, artifact: ResultArtifact, arg: str) -> ChartResult:
    """Render the last artifact snapshot to a PNG under OUTPUT_DIR/charts deterministically.

    Uses only artifact.columns and artifact.rows: no model call, no SQL re-fetch, no generated
    code. matplotlib is imported lazily (viz extra) only after validation passes.
    """
    chart_type, options = parse_plot_command_args(arg)
    if not artifact.rows:
        raise AppError("No rows to plot.")

    if chart_type == "hist":
        required = ("column",)
    else:
        required = ("x", "y")
    missing = [key for key in required if key not in options]
    if missing:
        raise AppError(f"Plot type '{chart_type}' requires options: {', '.join(required)}.")
    extra = [key for key in options if key not in required]
    if extra:
        raise AppError(f"Plot type '{chart_type}' does not support options: {', '.join(sorted(extra))}.")

    names = make_unique_column_names(artifact.columns)
    dicts = artifact_rows_as_dicts(artifact)

    x_column: str | None = None
    y_column: str | None = None
    labels: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    values: list[float] = []

    if chart_type in {"bar", "line"}:
        x_column = resolve_column(names, options["x"])
        y_column = resolve_column(names, options["y"])
        for row in dicts:
            y = _chart_numeric(row[y_column])
            if y is None:
                continue
            raw_x = row[x_column]
            labels.append("" if raw_x is None else str(raw_x))
            ys.append(y)
        if not ys:
            raise AppError("No valid numeric data to plot.")
        plotted = len(ys)
    elif chart_type == "scatter":
        x_column = resolve_column(names, options["x"])
        y_column = resolve_column(names, options["y"])
        for row in dicts:
            x = _chart_numeric(row[x_column])
            y = _chart_numeric(row[y_column])
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            raise AppError("No valid numeric data to plot.")
        plotted = len(xs)
    else:  # hist
        x_column = resolve_column(names, options["column"])
        for row in dicts:
            value = _chart_numeric(row[x_column])
            if value is None:
                continue
            values.append(value)
        if not values:
            raise AppError("No valid numeric data to plot.")
        plotted = len(values)

    plt = _require_viz_libs()
    charts_dir = settings.output_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    path = _next_chart_path(charts_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if chart_type == "bar":
            ax.bar(labels, ys)
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_title(f"Bar: {y_column} by {x_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        elif chart_type == "line":
            ax.plot(labels, ys, marker="o")
            ax.tick_params(axis="x", labelrotation=45)
            ax.set_title(f"Line: {y_column} by {x_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        elif chart_type == "scatter":
            ax.scatter(xs, ys)
            ax.set_title(f"Scatter: {y_column} vs {x_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        else:  # hist
            ax.hist(values)
            ax.set_title(f"Histogram: {x_column}")
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        fig.savefig(path)
    finally:
        plt.close(fig)

    return ChartResult(
        path=path,
        chart_type=chart_type,
        x_column=x_column,
        y_column=y_column,
        row_count=plotted,
    )


_SORT_DIRECTIONS = {"asc": "asc", "desc": "desc", "ascending": "asc", "descending": "desc"}
_GROUPBY_AGG_PHRASES = {
    "sum": "sum",
    "mean": "mean",
    "average": "mean",
    "min": "min",
    "max": "max",
}
_FILTER_OP_PHRASES = {  # lowercased op phrase -> canonical op
    "eq": "eq",
    "equals": "eq",
    "equal to": "eq",
    "is": "eq",
    "ne": "ne",
    "not equals": "ne",
    "not equal to": "ne",
    "is not": "ne",
    "gt": "gt",
    "greater than": "gt",
    "above": "gt",
    "more than": "gt",
    "gte": "gte",
    "at least": "gte",
    "greater than or equal to": "gte",
    "lt": "lt",
    "less than": "lt",
    "below": "lt",
    "under": "lt",
    "lte": "lte",
    "at most": "lte",
    "less than or equal to": "lte",
    "contains": "contains",
    "including": "contains",
}


def _canonical_route_column(names: tuple[str, ...], token: str) -> str:
    """Resolve a routed column token to its canonical artifact name (raises if unknown)."""
    return resolve_column(names, token.strip("?.!"))


def _route_sort_followup(text: str, names: tuple[str, ...]) -> RoutedArtifactCommand | None:
    """Route 'sort by <col> [dir]' / 'order by <col> [dir]' / 'sort <col> [dir]'."""
    match = re.match(r"(?:sort by|order by|sort)\s+(.+)", text, re.IGNORECASE)
    if match is None:
        return None
    tokens = match.group(1).strip().split(" ")
    if len(tokens) == 1:
        column_token, order = tokens[0], "asc"
    elif len(tokens) == 2:
        order = _SORT_DIRECTIONS.get(tokens[1].lower())
        if order is None:
            return None
        column_token = tokens[0]
    else:
        return None
    column = _canonical_route_column(names, column_token)
    return RoutedArtifactCommand("sort", f"column={column} order={order}", "sort")


def _route_select_followup(text: str, names: tuple[str, ...]) -> RoutedArtifactCommand | None:
    """Route 'select <c1,c2>' / 'only columns <...>' / 'show only <...>' / 'only show <...>'."""
    match = re.match(r"(select|only columns|only show|show only)\s+(.+)", text, re.IGNORECASE)
    if match is None:
        return None
    prefix = match.group(1).lower()
    remainder = match.group(2).strip()
    # Bare-select guard: a comma is required unless the prefix is explicitly "only columns",
    # so SQL-like phrasing ("select customers") is not hijacked as a projection.
    if prefix != "only columns" and "," not in remainder:
        return None
    tokens = [token.strip() for token in remainder.split(",")]
    tokens = [token for token in tokens if token]
    if not tokens:
        return None
    if any(" " in token for token in tokens):
        return None
    resolved = [_canonical_route_column(names, token) for token in tokens]
    return RoutedArtifactCommand("select", f"columns={','.join(resolved)}", "select")


def _route_filter_followup(text: str, names: tuple[str, ...]) -> RoutedArtifactCommand | None:
    """Route 'filter|where|keep rows where <col> <op-phrase> <single-token-value>'."""
    match = re.match(r"(?:keep rows where|where|filter)\s+(.+)", text, re.IGNORECASE)
    if match is None:
        return None
    tokens = match.group(1).strip().split(" ")
    if len(tokens) < 3:
        return None
    column_token = tokens[0]
    value = tokens[-1]
    op = _FILTER_OP_PHRASES.get(" ".join(tokens[1:-1]).lower())
    if op is None:  # validated before resolving the column, so non-filter prose returns None
        return None
    if not value:
        return None
    column = _canonical_route_column(names, column_token)
    return RoutedArtifactCommand("filter", f"column={column} op={op} value={value}", "filter")


def _route_groupby_followup(text: str, names: tuple[str, ...]) -> RoutedArtifactCommand | None:
    """Route 'count by <col>' and 'group by <col> [count | <agg> <metric>]'."""
    count_match = re.match(r"count by\s+(.+)", text, re.IGNORECASE)
    if count_match is not None:
        tokens = count_match.group(1).strip().split(" ")
        if len(tokens) != 1:
            return None
        by_column = _canonical_route_column(names, tokens[0])
        return RoutedArtifactCommand("groupby", f"by={by_column} agg=count", "groupby")
    match = re.match(r"group by\s+(.+)", text, re.IGNORECASE)
    if match is None:
        return None
    tokens = match.group(1).strip().split(" ")
    if len(tokens) == 2 and tokens[1].lower() == "count":
        by_column = _canonical_route_column(names, tokens[0])
        return RoutedArtifactCommand("groupby", f"by={by_column} agg=count", "groupby")
    if len(tokens) == 3:
        agg = _GROUPBY_AGG_PHRASES.get(tokens[1].lower())
        if agg is None:
            return None
        by_column = _canonical_route_column(names, tokens[0])
        metric_column = _canonical_route_column(names, tokens[2])
        return RoutedArtifactCommand(
            "groupby", f"by={by_column} metric={metric_column} agg={agg}", "groupby"
        )
    return None


def route_artifact_followup(text: str, columns: Sequence[str]) -> RoutedArtifactCommand | None:
    """Map a natural-language follow-up to an existing artifact command, deterministically.

    Pure and conservative: returns a RoutedArtifactCommand only when the text clearly refers to
    the current result; otherwise returns None so the caller treats it as a new DB question. It
    never calls the model, generates code, or runs SQL. Only the chart-resolution path may raise
    AppError (unknown/ambiguous column); every other family returns a command or None.
    """
    collapsed = re.sub(r"\s+", " ", text.strip())
    norm = collapsed.lower()
    if not norm:
        return None
    core_text = norm.rstrip("?.!").strip()
    core_collapsed = collapsed.rstrip("?.!").strip()  # case-preserving (filter values keep case)

    # A. describe (deterministic profile -- intentionally not "summarize this result")
    if core_text in {
        "describe this",
        "describe this result",
        "profile this",
        "show profile",
        "show stats",
        "statistics",
    }:
        return RoutedArtifactCommand("describe", "", "describe-phrase")

    # B. head (anchored; "top 5 genres by revenue" will not match)
    for pattern in (
        r"head(?:\s+(\d+))?",
        r"(?:show\s+)?(?:first|top)\s+(\d+)\s+rows",
        r"(?:show\s+)?(?:first|top)\s+rows",
        r"(?:show\s+)?(?:first|top)\s+(\d+)",
    ):
        match = re.fullmatch(pattern, core_text)
        if match:
            count = match.group(1) if (match.re.groups and match.group(1)) else "10"
            return RoutedArtifactCommand("head", count, "head")

    # C. tail (symmetric with head)
    for pattern in (
        r"tail(?:\s+(\d+))?",
        r"(?:show\s+)?(?:last|bottom)\s+(\d+)\s+rows",
        r"(?:show\s+)?(?:last|bottom)\s+rows",
        r"(?:show\s+)?(?:last|bottom)\s+(\d+)",
    ):
        match = re.fullmatch(pattern, core_text)
        if match:
            count = match.group(1) if (match.re.groups and match.group(1)) else "10"
            return RoutedArtifactCommand("tail", count, "tail")

    # D. export
    if re.fullmatch(r"(?:export|save|download)(?:\s+to)?\s+csv", core_text):
        return RoutedArtifactCommand("export", "csv", "export-csv")

    # E. sql
    if re.fullmatch(
        r"(?:show\s+(?:the\s+)?sql|what\s+sql\s+did\s+you\s+run|show\s+(?:the\s+)?query)",
        core_text,
    ):
        return RoutedArtifactCommand("sql", "", "sql")

    # F. columns
    if re.fullmatch(
        r"(?:(?:show|list)\s+columns|what\s+columns\s+are\s+in\s+this\s+result)", core_text
    ):
        return RoutedArtifactCommand("columns", "", "columns")

    # G. artifacts (intentionally not bare "show results")
    if re.fullmatch(
        r"(?:(?:show|list)\s+artifacts|(?:show|list)\s+session\s+(?:artifacts|results))",
        core_text,
    ):
        return RoutedArtifactCommand("artifacts", "", "artifacts")

    names = make_unique_column_names(tuple(columns))

    # Transformation routes (v3.5): sort / select / filter / groupby over the latest artifact.
    # Conservative, case-preserving; each returns None when the phrase is not clearly its command.
    for transform_router in (
        _route_sort_followup,
        _route_select_followup,
        _route_filter_followup,
        _route_groupby_followup,
    ):
        routed = transform_router(core_collapsed, names)
        if routed is not None:
            return routed

    # H. plot (checked last; requires explicit options, else None)
    if re.search(r"\b(?:plot\s+bar|bar\s+chart)\b", norm):
        chart_type = "bar"
    elif re.search(r"\b(?:plot\s+line|line\s+chart)\b", norm):
        chart_type = "line"
    elif re.search(r"\b(?:plot\s+scatter|scatter\s+plot)\b", norm):
        chart_type = "scatter"
    elif re.search(r"\b(?:plot\s+hist|histogram|hist)\b", norm):
        chart_type = "hist"
    else:
        return None

    if chart_type == "hist":
        column_match = (
            re.search(r"column=(\S+)", norm)
            or re.search(r"histogram\s+of\s+(\S+)", norm)
            or re.search(r"histogram\s+(\S+)", norm)
        )
        if column_match is None:
            return None
        token = column_match.group(1).strip("?.!")
        if not token or token in {"of", "column"}:
            return None
        resolved = resolve_column(names, token)
        return RoutedArtifactCommand("plot", f"hist column={resolved}", "plot-hist")

    x_match = re.search(r"x=(\S+)", norm)
    y_match = re.search(r"y=(\S+)", norm)
    if x_match is None or y_match is None:
        return None
    resolved_x = resolve_column(names, x_match.group(1).strip("?.!"))
    resolved_y = resolve_column(names, y_match.group(1).strip("?.!"))
    return RoutedArtifactCommand(
        "plot", f"{chart_type} x={resolved_x} y={resolved_y}", f"plot-{chart_type}"
    )


def build_summary_payload(
    question: str,
    sql: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    truncated: bool,
    analysis_text: str | None = None,
) -> str:
    """Build the grounded LLM summary prompt (factored out so it is unit-testable)."""
    payload = {
        "question": question,
        "sql": sql,
        "columns": list(columns),
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
        "truncated": truncated,
    }
    prompt = (
        "Summarize the SQL result for the user's question in 1 to 3 short sentences.\n"
        "Use only the returned table data. Do not invent causes, missing rows, or outside facts.\n"
        "If the table is insufficient to answer the question, say that plainly.\n\n"
        f"Data:\n{json.dumps(payload, default=str, ensure_ascii=False)}\n"
    )
    if analysis_text:
        prompt += (
            "\nDeterministic result profile (computed from the returned rows):\n"
            f"{analysis_text}\n"
            "Use this profile only to summarize the returned result; do not invent rows or "
            "columns. The profile may describe a capped subset, so do not claim it covers all "
            "rows unless the result was not truncated.\n"
        )
    return prompt + "\nSummary:"


def generate_llm_summary(
    client: OpenAI,
    settings: Settings,
    question: str,
    sql: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[Any, ...], ...],
    truncated: bool,
    analysis_text: str | None = None,
) -> str:
    prompt = build_summary_payload(question, sql, columns, rows, truncated, analysis_text)
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
    analysis_text: str | None = None,
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
            analysis_text,
        )
        if not summary:
            return fallback, "deterministic", "LLM summary was empty."
        return summary, "llm", None
    except Exception as exc:
        return fallback, "deterministic", str(exc)


def log_query(settings: Settings, result: AnswerResult) -> None:
    if not settings.enable_query_logging:
        return
    try:
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
    except (sqlite3.Error, OSError):
        # Logging is best-effort: a failed metadata write must never break an answer.
        return


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
        result.validation_error = None
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
        if settings.enable_dataframe_analysis:
            try:
                result.analysis, result.analysis_text = analyze_execution(
                    settings, result.sql or "", execution
                )
            except AppError as exc:
                result.analysis_error = str(exc)
            except Exception as exc:  # analysis must never fail a successful answer
                result.analysis_error = f"Analysis failed: {exc}"
        result.summary, result.summary_mode, result.summary_error = summarize_result(
            client,
            settings,
            question,
            result.sql or "",
            execution,
            expectation,
            analysis_text=result.analysis_text,
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
        with closing(sqlite3.connect(sqlite_readonly_uri(path), uri=True)) as conn:
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
        with closing(sqlite3.connect(sqlite_readonly_uri(path), uri=True)) as conn:
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


def _next_report_path(base_dir: Path, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dot_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    base_name = f"report_{timestamp}"
    path = base_dir / f"{base_name}{dot_suffix}"
    if not path.exists():
        return path
    counter = 2
    while True:
        path = base_dir / f"{base_name}_{counter}{dot_suffix}"
        if not path.exists():
            return path
        counter += 1


def _markdown_fence(text: str, language: str = "") -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    suffix = language if language else ""
    return f"{fence}{suffix}\n{text}\n{fence}"


def _linked_analyses_by_artifact(
    artifacts: Sequence[ResultArtifact],
    analyses: Sequence[AnalysisArtifact],
) -> dict[int, tuple[AnalysisArtifact, ...]]:
    artifact_ids = {artifact.artifact_id for artifact in artifacts}
    grouped: dict[int, list[AnalysisArtifact]] = {artifact_id: [] for artifact_id in artifact_ids}
    for analysis in analyses:
        if analysis.source_artifact_id in grouped:
            grouped[analysis.source_artifact_id].append(analysis)
    return {artifact_id: tuple(items) for artifact_id, items in grouped.items()}


def _linked_analyses(
    artifacts: Sequence[ResultArtifact],
    analyses: Sequence[AnalysisArtifact],
) -> tuple[AnalysisArtifact, ...]:
    artifact_ids = {artifact.artifact_id for artifact in artifacts}
    return tuple(analysis for analysis in analyses if analysis.source_artifact_id in artifact_ids)


def _analysis_artifact_html_parts(analysis: AnalysisArtifact) -> list[str]:
    import html

    def cell(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray, memoryview)):
            return "&lt;binary&gt;"
        return html.escape(str(value))

    parts = [
        '    <article class="analysis-artifact">',
        f"      <h4>Analysis #{analysis.analysis_id}: {html.escape(analysis.title)}</h4>",
        f"      <p><strong>Source Artifact:</strong> #{analysis.source_artifact_id}</p>",
        f"      <p><strong>Created:</strong> {html.escape(analysis.created_at)}</p>",
        f"      <p><strong>Recipe:</strong> {html.escape(analysis.recipe)}</p>",
        f"      <p><strong>Status:</strong> {html.escape(analysis.status)}</p>",
    ]
    if analysis.summary:
        parts.append(f"      <p>{html.escape(analysis.summary)}</p>")
    if analysis.metrics:
        parts.append("      <h5>Metrics</h5>")
        parts.append("      <table>")
        parts.append("        <thead><tr><th>Field</th><th>Value</th></tr></thead>")
        parts.append("        <tbody>")
        for key, value in analysis.metrics.items():
            parts.append(
                f"          <tr><td>{html.escape(str(key))}</td><td>{cell(value)}</td></tr>"
            )
        parts.append("        </tbody>")
        parts.append("      </table>")
    if analysis.warnings:
        parts.append("      <h5>Warnings</h5>")
        parts.append("      <ul>")
        for warning in analysis.warnings:
            parts.append(f"        <li>{html.escape(warning)}</li>")
        parts.append("      </ul>")
    for table in analysis.tables:
        parts.append(f"      <h5>{html.escape(table.title)}</h5>")
        if table.columns:
            parts.append("      <table>")
            parts.append("        <thead>")
            parts.append("          <tr>")
            for column in table.columns:
                parts.append(f"            <th>{html.escape(column)}</th>")
            parts.append("          </tr>")
            parts.append("        </thead>")
            parts.append("        <tbody>")
            for row in table.rows:
                parts.append("          <tr>")
                for value in row:
                    parts.append(f"            <td>{cell(value)}</td>")
                parts.append("          </tr>")
            parts.append("        </tbody>")
            parts.append("      </table>")
        else:
            parts.append("      <p>No columns.</p>")
    parts.append("    </article>")
    return parts


def render_artifact_report_markdown(
    artifacts: Sequence[ResultArtifact],
    *,
    analyses: Sequence[AnalysisArtifact] = (),
    title: str = "Artifact Report",
    max_preview_rows: int = 50,
) -> str:
    if not artifacts:
        raise AppError("No artifacts to report.")

    report_time = utc_now()
    analyses_by_source = _linked_analyses_by_artifact(artifacts, analyses)
    linked_analysis_count = sum(len(items) for items in analyses_by_source.values())

    lines = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Created**: {report_time}")
    lines.append(f"- **Artifacts**: {len(artifacts)}")
    lines.append(f"- **Analyses**: {linked_analysis_count}")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Report uses stored artifact rows and analysis digests only. No SQL was re-run.")
    lines.append("")

    for art in artifacts:
        lines.append("---")
        lines.append("")
        lines.append(f"## Artifact #{art.artifact_id}")
        lines.append("")
        lines.append(f"- **Created**: {art.created_at}")
        lines.append(f"- **Question**: {art.question}")
        lines.append(f"- **SQL Truncated**: {'yes' if art.truncated else 'no'}")
        lines.append(f"- **Columns**: {', '.join(art.columns)}")
        lines.append(f"- **Total Rows**: {len(art.rows)}")
        lines.append("")

        lines.append("### SQL")
        lines.append(_markdown_fence(art.sql, "sql"))
        lines.append("")

        lines.append("### Preview")
        if art.columns:
            lines.extend(_markdown_table_lines(art.columns, art.rows[:max_preview_rows]))
        else:
            lines.append("No columns.")
        lines.append("")

        if len(art.rows) > max_preview_rows:
            lines.append(f"Showing first {max_preview_rows} of {len(art.rows)} stored rows.")
            lines.append("")

        if art.analysis_text:
            lines.append("### Analysis")
            lines.append(_markdown_fence(art.analysis_text))
            lines.append("")

        linked_analyses = analyses_by_source.get(art.artifact_id, ())
        if linked_analyses:
            lines.append("### Saved Analyses")
            lines.append("")
            for analysis in linked_analyses:
                lines.extend(_analysis_artifact_markdown_lines(analysis, heading_level=4))

    return "\n".join(lines)


def render_artifact_report_html(
    artifacts: Sequence[ResultArtifact],
    *,
    analyses: Sequence[AnalysisArtifact] = (),
    title: str = "Artifact Report",
    max_preview_rows: int = 50,
) -> str:
    if not artifacts:
        raise AppError("No artifacts to report.")

    import html

    report_time = utc_now()
    analyses_by_source = _linked_analyses_by_artifact(artifacts, analyses)
    linked_analysis_count = sum(len(items) for items in analyses_by_source.values())

    def escape_html_val(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, (bytes, bytearray, memoryview)):
            return "<binary>"
        return html.escape(str(val))

    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html>")
    html_parts.append("<head>")
    html_parts.append('  <meta charset="utf-8">')
    html_parts.append(f"  <title>{html.escape(title)}</title>")
    html_parts.append("  <style>")
    html_parts.append("    body { font-family: sans-serif; margin: 20px; line-height: 1.5; color: #333; }")
    html_parts.append("    h1 { border-bottom: 2px solid #eee; padding-bottom: 10px; }")
    html_parts.append("    .warning-note { background-color: #fff3cd; border: 1px solid #ffeeba; color: #856404; padding: 10px; border-radius: 4px; margin-bottom: 20px; }")
    html_parts.append("    section { border: 1px solid #ccc; padding: 20px; margin-bottom: 20px; border-radius: 5px; }")
    html_parts.append("    table { border-collapse: collapse; width: 100%; margin: 10px 0; }")
    html_parts.append("    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
    html_parts.append("    th { background-color: #f5f5f5; }")
    html_parts.append("    pre { background-color: #f9f9f9; padding: 10px; border: 1px solid #eee; overflow-x: auto; }")
    html_parts.append("    code { font-family: monospace; }")
    html_parts.append("    .analysis-artifact { border-left: 4px solid #d0e3ff; padding-left: 12px; margin: 14px 0; }")
    html_parts.append("  </style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append(f"  <h1>{html.escape(title)}</h1>")
    html_parts.append(f"  <p><strong>Created:</strong> {html.escape(report_time)}</p>")
    html_parts.append(f"  <p><strong>Artifacts:</strong> {len(artifacts)}</p>")
    html_parts.append(f"  <p><strong>Analyses:</strong> {linked_analysis_count}</p>")
    html_parts.append('  <div class="warning-note">Report uses stored artifact rows and analysis digests only. No SQL was re-run.</div>')

    for art in artifacts:
        html_parts.append("  <section>")
        html_parts.append(f"    <h2>Artifact #{art.artifact_id}</h2>")
        html_parts.append(f"    <p><strong>Created:</strong> {html.escape(art.created_at)}</p>")
        html_parts.append(f"    <p><strong>Question:</strong> {html.escape(art.question)}</p>")
        html_parts.append(f"    <p><strong>SQL Truncated:</strong> {html.escape('yes' if art.truncated else 'no')}</p>")
        html_parts.append(f"    <p><strong>Columns:</strong> {html.escape(', '.join(art.columns))}</p>")
        html_parts.append(f"    <p><strong>Total Rows:</strong> {len(art.rows)}</p>")

        html_parts.append("    <h3>SQL</h3>")
        html_parts.append(f"    <pre><code>{html.escape(art.sql)}</code></pre>")

        html_parts.append("    <h3>Preview</h3>")
        if art.columns:
            html_parts.append("    <table>")
            html_parts.append("      <thead>")
            html_parts.append("        <tr>")
            for col in art.columns:
                html_parts.append(f"          <th>{html.escape(col)}</th>")
            html_parts.append("        </tr>")
            html_parts.append("      </thead>")
            html_parts.append("      <tbody>")
            for row in art.rows[:max_preview_rows]:
                html_parts.append("        <tr>")
                for val in row:
                    html_parts.append(f"          <td>{escape_html_val(val)}</td>")
                html_parts.append("        </tr>")
            html_parts.append("      </tbody>")
            html_parts.append("    </table>")
        else:
            html_parts.append("    <p>No columns.</p>")

        if len(art.rows) > max_preview_rows:
            html_parts.append(f"    <p>Showing first {max_preview_rows} of {len(art.rows)} stored rows.</p>")

        if art.analysis_text:
            html_parts.append("    <h3>Analysis</h3>")
            html_parts.append(f"    <pre><code>{html.escape(art.analysis_text)}</code></pre>")

        linked_analyses = analyses_by_source.get(art.artifact_id, ())
        if linked_analyses:
            html_parts.append("    <h3>Saved Analyses</h3>")
            for analysis in linked_analyses:
                html_parts.extend(_analysis_artifact_html_parts(analysis))

        html_parts.append("  </section>")

    html_parts.append("</body>")
    html_parts.append("</html>")
    return "\n".join(html_parts)


def export_artifact_report(
    settings,
    artifacts: Sequence[ResultArtifact],
    *,
    report_format: str,
    include_all: bool = False,
    analyses: Sequence[AnalysisArtifact] = (),
) -> ReportExportResult:
    fmt = report_format.strip().lower()
    if fmt in {"md", "markdown"}:
        normalized_format = "markdown"
        suffix = "md"
    elif fmt == "html":
        normalized_format = "html"
        suffix = "html"
    else:
        raise AppError(f"Unsupported report format: {report_format}")

    if not artifacts:
        raise AppError("No artifacts to report.")

    to_report = artifacts if include_all else [artifacts[-1]]
    to_report_analyses = _linked_analyses(to_report, analyses)

    reports_dir = Path(settings.output_path) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    file_path = _next_report_path(reports_dir, suffix)

    if normalized_format == "markdown":
        content = render_artifact_report_markdown(to_report, analyses=to_report_analyses)
    else:
        content = render_artifact_report_html(to_report, analyses=to_report_analyses)

    file_path.write_text(content, encoding="utf-8")

    return ReportExportResult(
        path=file_path,
        artifact_count=len(to_report),
        format=normalized_format,
        analysis_count=len(to_report_analyses),
    )


def export_workspace_report(
    settings,
    workspace_target: str,
    *,
    report_format: str,
) -> ReportExportResult:
    loaded = load_artifact_workspace(settings, workspace_target)
    return export_artifact_report(
        settings,
        loaded.artifacts,
        report_format=report_format,
        include_all=True,
        analyses=loaded.analyses,
    )
