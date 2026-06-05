# Northwind Text-to-SQL CLI

A local Text-to-SQL command-line application for the Northwind SQLite database. It inspects the database schema, embeds table-level schema documents with a local oMLX OpenAI-compatible embeddings endpoint, retrieves relevant schema through sqlite-vec, generates SQLite `SELECT` queries with a local oMLX-served model, validates them, executes them read-only, and renders results with Rich.

## Architecture

- `app.py`: Typer commands, interactive prompt, Rich output.
- `core.py`: schema inspection, indexing, retrieval, prompting, SQL validation, read-only execution, summaries, logging.
- `config.py`: pydantic-settings configuration loaded from `.env`.
- `metadata.db`: local schema index, sqlite-vec vectors, and query logs.

The runtime path is intentionally bounded:

1. retrieve table-level schema objects
2. expand one hop through foreign-key relationships
3. generate candidate SQL
4. validate structural safety with SQLGlot and schema checks
5. execute on a read-only SQLite connection with timeout and row limits
6. optionally repair once for validation or high-confidence result-shape mismatch
7. show SQL, table results, and a concise summary

Validation proves only that SQL is structurally safe and executable. It does not prove the query perfectly answers the user's intent.

## Prerequisites

- Python 3.11+
- uv
- A local oMLX OpenAI-compatible API at `OMLX_BASE_URL`
- The local API must serve:
  - `MLX-Qwopus3.5-9B-v3-4bit`
  - `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`

The app sends request-level generation settings, so it does not depend on server-wide defaults for SQL determinism.

## Install

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `OMLX_API_KEY` to any non-empty local value accepted by your oMLX server.

## Database

Download the pinned Northwind SQLite database:

```bash
uv run python app.py download-northwind
```

The command downloads `dist/northwind.db` from `jpwhite3/northwind-SQLite3`, pinned to commit `4f56e7f5906dfd23b25244c5bfe8fb5da6402efd`, verifies core Northwind tables, and saves it to `data/northwind.db`. The upstream database is MIT licensed.

You can also place a compatible Northwind SQLite file manually at `data/northwind.db`.

## Usage

Build or rebuild the schema index:

```bash
uv run python app.py index
```

Start the interactive assistant:

```bash
uv run python app.py ask
```

Ask one question and exit:

```bash
uv run python app.py ask "Which customers placed the most orders?"
```

Show recent query logs:

```bash
uv run python app.py logs
uv run python app.py logs --limit 25 --verbose
```

Run tests:

```bash
uv run pytest
```

## Configuration

Important settings in `.env`:

- `NORTHWIND_DB_PATH`: source SQLite database.
- `METADATA_DB_PATH`: schema index and local query log database.
- `AUTO_REINDEX`: when true, `ask` rebuilds the schema index if the schema/model/path changed.
- `ENABLE_RESULT_SHAPE_CHECK`: enables lightweight result-shape heuristics.
- `MAX_REPAIR_ATTEMPTS`: total repair budget, default `1`.
- `REQUIRE_SQL_APPROVAL`: ask before executing validated SQL.
- `ENABLE_LLM_SUMMARY`: optional second model call for grounded summaries, default `false`.
- `ENABLE_QUERY_LOGGING`: stores local query logs, default `true`.

If only row data changes while table schema is unchanged, the app warns but does not reindex. If table schema changes, `ask` auto-reindexes before answering when `AUTO_REINDEX=true`.

## Safety Boundaries

- Only one parseable SQLite statement is allowed.
- Only `SELECT` and non-recursive `WITH ... SELECT` are allowed.
- Prohibited commands include `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, `ATTACH`, and `DETACH`.
- SQL comments are rejected.
- SQLite system tables are rejected.
- Known tables are validated strictly; column checks are pragmatic for aliases and CTEs.
- Queries run through `sqlite3` with `mode=ro`, extension loading disabled, row limits, and a progress-handler timeout.
- sqlite-vec is loaded only for the metadata database, never for the source database.

## Example Questions

- Which customers placed the most orders?
- What are the top 5 products by quantity ordered?
- Which employees handled the most orders?
- What is the total freight by ship country?
- Which suppliers provide the most products?

## Known Limitations

- Runtime validation is not a semantic correctness judge.
- Views are not indexed or allowed in v1.
- No live oMLX integration tests are included.
- No full prompt or result-row logging is implemented.
- Result-shape checks are conservative heuristics and may warn on valid answers.

## Sources

- Northwind SQLite database: https://github.com/jpwhite3/northwind-SQLite3
- Upstream license: https://github.com/jpwhite3/northwind-SQLite3/blob/main/LICENSE
- sqlite-vec: https://alexgarcia.xyz/sqlite-vec/
- SQLGlot: https://sqlglot.com/sqlglot.html
- SQLite URI/read-only mode: https://www.sqlite.org/uri.html
- Python sqlite3: https://docs.python.org/3/library/sqlite3.html
