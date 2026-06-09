# SQLite Text-to-SQL CLI

A local Text-to-SQL command-line application for **any SQLite database**. Provide a path with `--db <path>` (it defaults to a bundled Northwind demo). It inspects the database schema (tables and views), optionally profiles each object with bounded, read-only sample values, embeds table/view-level schema documents with a local oMLX OpenAI-compatible embeddings endpoint, retrieves relevant schema through sqlite-vec, generates SQLite `SELECT` queries with a local oMLX-served model, validates them, executes them read-only, and renders results with Rich.

## Architecture

- `app.py`: Typer commands, interactive prompt, Rich output.
- `core.py`: schema inspection, indexing, retrieval, prompting, SQL validation, read-only execution, summaries, logging.
- `config.py`: pydantic-settings configuration loaded from `.env`.
- `data/metadata/<db-stem>-<path-hash>.metadata.db`: per-source-database schema index, sqlite-vec vectors, index state, and query logs. Each source database gets its own metadata file, derived automatically from the database's absolute path, so different databases never share an index or logs.

The runtime path is intentionally bounded:

1. retrieve table/view-level schema objects
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

The app works with **any SQLite database file** (`.db`, `.sqlite`, `.sqlite3` — the file is validated by opening it, not by its extension). Provide a path with `--db <path>` on the `ask`, `index`, and `logs` commands. When `--db` is omitted, the bundled Northwind demo at `data/northwind.db` is used.

### Use your own database

```bash
uv run python app.py ask --db /path/to/your.db "List the 10 most recent orders"
```

The first run for a new database automatically builds its schema index (when `AUTO_REINDEX=true`). The index and query logs live in a per-database metadata file under `data/metadata/`, keyed by the database's absolute path — so switching between databases never re-embeds or mixes logs. Metadata is tied to the *path*: the same database copied to two locations gets two metadata files. Relative paths like `data/metadata/` resolve against your current working directory.

### Demo database (optional)

Download the pinned Northwind SQLite database to try the app quickly:

```bash
uv run python app.py download-northwind
```

The command downloads `dist/northwind.db` from `jpwhite3/northwind-SQLite3`, pinned to commit `4f56e7f5906dfd23b25244c5bfe8fb5da6402efd`, verifies core Northwind tables, and saves it to the configured `DB_PATH` (default `data/northwind.db`). The upstream database is MIT licensed.

## Usage

Place the `--db` option before the question argument. When omitted, the default demo database is used.

Build or rebuild the schema index:

```bash
uv run python app.py index                       # default demo database
uv run python app.py index --db data/your.db     # your database
```

Start the interactive assistant:

```bash
uv run python app.py ask --db data/your.db
```

#### Interactive commands

Inside the interactive assistant, the last successful result is kept as an in-session
**artifact** you can inspect and export with deterministic colon-commands (no model call, no
generated code):

| Command | Action |
| --- | --- |
| `:sql` | Show the SQL for the last result |
| `:columns` | Show the result column names |
| `:describe` | Show a deterministic profile (shape, dtypes, null/distinct, numeric min/max/mean) |
| `:head [N]` | Show the first N rows (default 10) |
| `:tail [N]` | Show the last N rows (default 10) |
| `:export [csv]` | Export the result to `OUTPUT_DIR` as CSV |
| `:plot bar x=<column> y=<column>` | Save a bar chart PNG to `OUTPUT_DIR/charts/` |
| `:plot line x=<column> y=<column>` | Save a line chart PNG |
| `:plot scatter x=<column> y=<column>` | Save a scatter chart PNG |
| `:plot hist column=<column>` | Save a histogram PNG |
| `:artifacts` | List this session's results |
| `:help` | Show this help |
| `:q` / `exit` / `quit` | Quit |

These work in interactive mode only. **Artifacts are in-memory and last only for the current
session; only exported CSV files persist.** `:describe` recomputes from the stored rows when
`ENABLE_DATAFRAME_ANALYSIS` was off, so it needs the `analysis` extra in that case. `:export csv`
does a bounded read-only re-fetch up to `MAX_ANALYSIS_ROWS` when the displayed result was
truncated (it reuses the validated SQL through the read-only executor; no extra required), and
reports the exact row count and whether the export was capped.

**Charts (`:plot`)** are deterministic artifact commands like the rest: they do **not** call
the model, do **not** run generated code, and do **not** re-run SQL. Unlike `:export csv`,
`:plot` uses **only the rows currently stored in the session artifact** — it does not perform
the fuller re-fetch that `:export csv` may do when a result was truncated. PNGs are written to
`OUTPUT_DIR/charts/` as `chart_001.png`, `chart_002.png`, … For bar/line, `x` may be
text/date/numeric and `y` must be numeric; scatter needs both numeric; hist needs a numeric
column. Non-numeric/empty cells are skipped. Because options are parsed as whitespace-separated
`key=value` tokens, **column names must not contain spaces** — alias them in SQL first, e.g.
`SELECT Total AS RevenueTotal ...`, then `:plot bar x=Genre y=RevenueTotal`.

Charting requires the optional `viz` extra (matplotlib). Install it once and run normally:

```bash
uv sync --extra viz
ENABLE_DATAFRAME_ANALYSIS=true uv run python app.py ask --db data/test_dbs/Chinook.db
```

`uv run --extra viz python app.py ...` works as a one-shot alternative. If `:plot` is used
without the extra installed, the session prints a notice and continues. To enable both result
analysis and charts together, install both extras: `uv sync --extra analysis --extra viz`.

##### Natural-language follow-ups

In the interactive assistant you can phrase follow-ups in plain English instead of the exact
colon syntax. When a result already exists and the text clearly refers to it, it is routed to
the matching artifact command (the session prints `Routed to :<command>`):

- `describe this result` → `:describe`
- `show first 5 rows` → `:head 5`
- `show last 10 rows` → `:tail 10`
- `export to csv` → `:export csv`
- `show sql` → `:sql`
- `show columns` → `:columns`
- `show artifacts` → `:artifacts`
- `plot bar x=GenreName y=TotalRevenue` → `:plot bar x=GenreName y=TotalRevenue`
- `histogram of TotalRevenue` → `:plot hist column=TotalRevenue`

This is a **deterministic router** to the existing artifact commands: it does **not** call the
model, and it does **not** generate or execute any code. Chart column names are matched
case-insensitively against the current result's columns. **Ambiguous or unrecognized requests
are treated as new database questions** rather than guessed — so an ordinary question like
`top 5 genres by revenue` still runs the Text-to-SQL pipeline. The router is interactive-only;
one-shot `ask "<question>"` never routes.

Ask one question and exit:

```bash
# Demo database (default)
uv run python app.py ask "Which customers placed the most orders?"
# Your own SQLite database (option before the question)
uv run python app.py ask --db data/your.db "List the 5 most recent orders"
```

Show recent query logs (scoped to the selected database):

```bash
uv run python app.py logs --db data/your.db
uv run python app.py logs --db data/your.db --limit 25 --verbose
```

Run tests:

```bash
uv run pytest
```

## Configuration

Important settings in `.env`:

- `DB_PATH`: default source SQLite database used when `--db` is not given.
- `METADATA_DB_PATH`: advanced/debug override for the metadata (index + logs) database. Leave **unset** so each source database gets its own auto-derived metadata file under `data/metadata/`. The `--metadata-db` CLI option overrides it per run.
- `AUTO_REINDEX`: when true, `ask` rebuilds the schema index if the schema/model/path changed.
- `ENABLE_RESULT_SHAPE_CHECK`: enables lightweight result-shape heuristics (tuned for typical business questions). If you see spurious result warnings on an unusual schema, set this to `false`.
- `MAX_REPAIR_ATTEMPTS`: total repair budget, default `1`.
- `REQUIRE_SQL_APPROVAL`: ask before executing validated SQL.
- `ENABLE_LLM_SUMMARY`: optional second model call for grounded summaries, default `false`.
- `ENABLE_QUERY_LOGGING`: stores local query logs, default `true`.
- `ENABLE_SCHEMA_PROFILING`: adds bounded row counts and sample values to schema documents, default `true`. Set `false` to skip profiling.
- `MAX_PROFILE_VALUES`: sample values collected per column during profiling, default `3`.
- `MAX_PROFILE_TEXT_LENGTH`: maximum characters kept per sampled text value, default `80`.
- `ENABLE_DATAFRAME_ANALYSIS`: opt-in deterministic result analysis (needs the `analysis` extra), default `false`.
- `MAX_ANALYSIS_ROWS`: row cap for the analysis fetch when displayed rows were truncated, default `5000`.
- `MAX_ANALYSIS_COLUMNS`: maximum columns rendered in the analysis panel / summary grounding, default `30`.
- `OUTPUT_DIR`: directory for interactive `:export` CSV files and `:plot` charts (under `charts/`), default `data/outputs` (gitignored).

If only row data changes while table schema is unchanged, the app warns but does not reindex. If table schema changes, `ask` auto-reindexes before answering when `AUTO_REINDEX=true`.

### Schema profiling

Schema profiling adds bounded row counts and sample values to table/view-level schema documents to improve SQL generation on unfamiliar databases. It can be disabled with `ENABLE_SCHEMA_PROFILING=false`. Profiling reads only through the read-only SQLite connection and never modifies the source database; profiles are built at index time and refreshed whenever the schema or profiling settings change. Views are profiled lightly (sample values only; no `COUNT(*)` or `MIN`/`MAX`).

### Result analysis

When `ENABLE_DATAFRAME_ANALYSIS=true`, a successful result is turned into a `pyarrow.Table` and a pandas DataFrame, and a deterministic profile is computed: shape, per-column dtype, null and distinct counts, and numeric min/max/mean. The profile is shown in an **Analysis** panel and, when `ENABLE_LLM_SUMMARY=true`, grounds the narrative summary. No model-generated code runs — only deterministic profiling over already-validated, read-only result rows. The Arrow table is an ADBC-ready substrate for a future multi-database backend.

Install the optional dependencies once with `uv sync --extra analysis` (pandas + pyarrow), then run normally (`ENABLE_DATAFRAME_ANALYSIS=true uv run python app.py ask ...`); `uv run --extra analysis python app.py ...` works as a one-shot alternative. The feature is off by default, and if it is enabled without the extra installed the answer still succeeds with a notice. **Note:** if displayed rows were truncated at `MAX_RESULT_ROWS`, analysis performs one additional bounded read-only fetch up to `MAX_ANALYSIS_ROWS`; if the result still exceeds that cap, the profile (and grounded summary) describe that capped subset and say so.

## Safety Boundaries

- Only one parseable SQLite statement is allowed.
- Only `SELECT` and non-recursive `WITH ... SELECT` are allowed.
- Prohibited commands include `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, `ATTACH`, and `DETACH`.
- SQL comments are rejected.
- SQLite system tables are rejected.
- Tables and views are allowed as read-only SELECT sources and validated strictly; column checks are pragmatic for aliases and CTEs.
- Queries run through `sqlite3` with `mode=ro`, extension loading disabled, row limits, and a progress-handler timeout.
- sqlite-vec is loaded only for the metadata database, never for the source database.

## Example Questions

- Which customers placed the most orders?
- What are the top 5 products by quantity ordered?
- Which employees handled the most orders?
- What is the total freight by ship country?
- Which suppliers provide the most products?

## Evals

`evals/northwind_questions.jsonl` and `evals/chinook_questions.jsonl` are small JSON-per-line regression/eval prompts (each line carries a `question` and an `expectation`) used for manual or future automated checks of result-shape behavior such as total counts vs. grouped breakdowns. They are documentation-oriented; there is no eval runner yet.

## Known Limitations

- Runtime validation is not a semantic correctness judge.
- Tables and views are indexed. Queries remain read-only and SELECT-only.
- No live oMLX integration tests are included.
- No full prompt or result-row logging is implemented.
- Result-shape checks are conservative heuristics and may warn on valid answers.
- Result analysis is descriptive (profiling), not inferential, and runs no model-generated code.

## Sources

- Northwind SQLite database: https://github.com/jpwhite3/northwind-SQLite3
- Upstream license: https://github.com/jpwhite3/northwind-SQLite3/blob/main/LICENSE
- sqlite-vec: https://alexgarcia.xyz/sqlite-vec/
- SQLGlot: https://sqlglot.com/sqlglot.html
- SQLite URI/read-only mode: https://www.sqlite.org/uri.html
- Python sqlite3: https://docs.python.org/3/library/sqlite3.html
