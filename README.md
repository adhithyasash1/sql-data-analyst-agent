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

#### Command Reference

The assistant supports top-level command-line entry points and interactive in-session colon commands:

| Command | Purpose | Model? | SQL? | Scope |
| --- | --- | --- | --- | --- |
| `index` | Inspect database and build schema index | Yes | Yes (Read-only) | DB |
| `ask "<question>"` | Ask a question once, or start interactive prompt | Yes | Yes (Read-only) | DB |
| `download-northwind` | Download the demo Northwind SQLite database | No | No | DB |
| `:help` | Show command helper text | No | No | Stored artifact |
| `:q` / `quit` / `exit` | Quit the assistant | No | No | Stored artifact |
| `:sql` | Show the SQL query for the last result | No | No | Stored artifact |
| `:columns` | Show the column names of the last result | No | No | Stored artifact |
| `:describe` | Show a deterministic profile of the last result | No | No | Stored artifact |
| `:head [N]` | Show the first N rows of the last result | No | No | Stored artifact |
| `:tail [N]` | Show the last N rows of the last result | No | No | Stored artifact |
| `:export csv` | Export the last result as CSV | No | Yes (Read-only)* | Stored artifact |
| `:plot ...` | Save a chart PNG to `OUTPUT_DIR/charts/` | No | No | Stored artifact |
| `:sort ...` | Create new artifact sorted by a column | No | No | Stored artifact |
| `:select ...` | Create new artifact with a subset of columns | No | No | Stored artifact |
| `:filter ...` | Create new filtered artifact keeping matching rows | No | No | Stored artifact |
| `:groupby ...` | Create new grouped/aggregated artifact | No | No | Stored artifact |
| `:save [name=<name>]` | Save session's artifacts to `OUTPUT_DIR/workspaces/` | No | No | Workspace |
| `:saved` | List saved workspaces | No | No | Workspace |
| `:workspace-info <workspace>` | Show details of a saved workspace | No | No | Workspace |
| `:delete-workspace <workspace>`| Delete a saved workspace directory | No | No | Workspace |
| `:load <workspace>` | Load saved workspace artifacts | No | No | Workspace |
| `:report <md\|html> [all]` | Export current session report | No | No | Report |
| `:report <md\|html> workspace=<workspace>` | Export report from a saved workspace | No | No | Report |

\* Note: `:export csv` performs a read-only fetch against the database *only if* the displayed rows were truncated.


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

Transformation follow-ups route to the v3.4 commands (each creates a new latest artifact):

- `sort by TotalRevenue descending` → `:sort column=TotalRevenue order=desc`
- `filter TotalRevenue greater than 100` → `:filter column=TotalRevenue op=gt value=100`
- `where GenreName contains Rock` → `:filter column=GenreName op=contains value=Rock`
- `select GenreName, TotalRevenue` → `:select columns=GenreName,TotalRevenue`
- `group by Country sum Revenue` → `:groupby by=Country metric=Revenue agg=sum`
- `count by Country` → `:groupby by=Country agg=count`

This is a **deterministic router** to the existing artifact commands: it does **not** call the
model, and it does **not** generate or execute any code. Column names are matched
case-insensitively against the **latest in-session artifact's** columns — not the original
database schema — so after `select GenreName, TotalRevenue` a later `sort by TrackCount` no
longer resolves (that column was dropped). There is **no fuzzy matching**: values containing
spaces are not routed yet (use single-token values), and **ambiguous or unrecognized requests
are treated as new database questions** rather than guessed — so an ordinary question like
`top 5 genres by revenue` still runs the Text-to-SQL pipeline. The router is interactive-only;
one-shot `ask "<question>"` never routes.

##### Transformations

These commands reshape the latest artifact and create a **new in-session artifact** that
becomes the latest, so `:describe`, `:plot`, `:export csv`, `:head`, and `:tail` then operate on
the transformed result:

| Command | Action |
| --- | --- |
| `:sort column=<col> order=<asc\|desc>` | New artifact sorted by a column (default `asc`; missing values last) |
| `:select columns=<col1,col2,...>` | New artifact with only those columns, in that order |
| `:filter column=<col> op=<eq\|ne\|gt\|gte\|lt\|lte\|contains> value=<v>` | New artifact keeping matching rows |
| `:groupby by=<col> metric=<col> agg=<sum\|mean\|count\|min\|max>` | New aggregated artifact (`metric` optional for `count`) |

Transformations are **deterministic**: they do **not** call the model, do **not** run SQL, and
do **not** execute any generated code — they operate only on the stored artifact snapshot. New
artifacts are in-memory only (not written to disk). If the source artifact was truncated at
`MAX_RESULT_ROWS`, the transform runs over that stored subset and the new artifact's question
records a `[... applied to truncated artifact]` note.

Transformation arguments are whitespace-separated `key=value` tokens. Values containing spaces
are not supported yet (use exact single-token values), and `:select` columns must be
comma-separated with no spaces, e.g. `columns=GenreName,TotalRevenue`.

Example flow — filter, sort, then chart and export the reshaped result:

```text
Which genres generated the most revenue?
:filter column=TotalRevenue op=gt value=100
:sort column=TotalRevenue order=desc
:plot bar x=GenreName y=TotalRevenue
:export csv
```

##### Persistent workspaces

Session artifacts are normally in-memory only. You can explicitly persist them to disk and
reload them in a later session:

| Command | Action |
| --- | --- |
| `:save` | Save the session's artifacts to a timestamped workspace |
| `:save name=my_analysis` | Save under a named workspace (`my_analysis_<timestamp>/`) |
| `:saved` | List saved workspaces |
| `:workspace-info <workspace>` | Show saved workspace details (exact name or unique prefix) |
| `:delete-workspace <workspace>` | Delete a saved workspace (exact name or unique prefix) |
| `:load <workspace>` | Load a saved workspace (exact directory name **or** a unique prefix) |

`:load my_analysis` resolves by unique prefix to `my_analysis_<timestamp>/`; if a prefix matches
more than one workspace, the candidates are listed instead of guessing. Notes:

- Workspaces are saved under `OUTPUT_DIR/workspaces/` (each holds `manifest.json` plus per-artifact
  CSV, SQL, and optional profile text). **Saving is explicit — there is no autosave.**
- **No model call and no SQL execution** happen during save, load, info inspection, or deletion.
- **Loaded CSV values are strings**, because CSV does not preserve original Python/SQLite types.
  Deterministic commands still work on a loaded workspace (e.g. `:filter ... op=gt value=100` and
  `:plot` parse numeric strings), but exact dtypes from the original query are not restored.
- For safety, `:load`, `:workspace-info`, and `:delete-workspace` only accept a bare workspace name under `OUTPUT_DIR/workspaces/` — absolute paths, path separators, and `..` traversal are rejected.
- `:delete-workspace` recursively deletes the target workspace directory under `OUTPUT_DIR/workspaces/` only. It does not delete or touch reports, charts, CSV exports, metadata DBs, or source DBs, and sibling workspaces remain untouched. No confirmation prompt is shown.


##### Report export

You can export a report of the latest result artifact or all session artifacts in Markdown or HTML:

| Command | Action |
| --- | --- |
| `:report md` | Export the latest result artifact as a Markdown report |
| `:report html` | Export the latest result artifact as an HTML report |
| `:report md all` | Export all session artifacts in order as a Markdown report |
| `:report html all` | Export all session artifacts in order as an HTML report |
| `:report md workspace=<workspace>` | Export all artifacts in a saved workspace as a Markdown report |
| `:report html workspace=<workspace>` | Export all artifacts in a saved workspace as an HTML report |

Aliases:
- `:report markdown` is an alias for `:report md`
- `:report markdown all` is an alias for `:report md all`
- `:report markdown workspace=<workspace>` is an alias for `:report md workspace=<workspace>`

Notes:
- Reports are saved under `OUTPUT_DIR/reports/` as `report_<YYYYmmdd_HHMMSS>.<md|html>`. If a file exists, a numeric suffix like `_2`, `_3`, etc. is appended automatically to prevent overwrites.
- Reports use **stored artifact rows only**.
- **No SQL is re-run** and **no model call is made** during export.
- Preview rows are capped at 50 rows.
- Markdown reports are best for editing and sharing; HTML reports are standalone and viewable in any web browser.
##### Example workflows

Here are three common lifecycle patterns in the interactive CLI:

1. **Transform and Chart**: Filter a database query, sort the remaining records, and plot a chart of the results:
   ```text
   Which genres generated the most revenue?
   filter TotalRevenue greater than 100
   sort by TotalRevenue descending
   :plot bar x=GenreName y=TotalRevenue
   ```

2. **Save and Resume**: Save the current interactive session, list the saved workspaces, and load it back later:
   ```text
   :save name=genre_revenue
   :saved
   :load genre_revenue
   :artifacts
   ```

3. **Report a Saved Workspace**: Generate HTML or Markdown reports from a saved workspace directory:
   ```text
   :report md workspace=genre_revenue
   :report html workspace=genre_revenue
   ```
   *Note: Workspace arguments accept either exact folder names or any unique prefix. Ambiguous prefixes are rejected with a list of matches.*


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

## Safety Guarantees

The application enforces the following safety boundaries and guarantees:
- **Read-Only SQLite Connection**: All database queries run through `sqlite3` in read-only mode (`mode=ro`), with extension loading disabled, row limits, and a progress-handler timeout.
- **Write SQL Rejection**: SQL validation checks block prohibited statements (such as `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, `ATTACH`, `DETACH`) and reject comments or system tables.
- **No Python/Generated Code Execution**: No model-generated code or dynamic Python blocks are ever executed by the assistant.
- **Deterministic Artifact Scope**: Interactive artifact commands (like `:sort`, `:select`, `:filter`, `:groupby`, `:plot`) process already-validated in-memory rows only. They do not run SQL or call the model.
- **Confined Workspace Operations**: Workspace reads, writes, and deletions are strictly confined to the `OUTPUT_DIR/workspaces` directory. The resolver rejects absolute paths, path separators, and parent directory traversal (`..`). Sibling workspace folders are left completely untouched.
- **Offline Reports**: Exported reports use stored artifact rows only. No SQL query is re-run and no model calls are made during Markdown or HTML generation.
- **Friendly Model/Connection Errors**: Local model server issues (such as connection failures or API errors) are captured and surfaced as readable `AppError` notices instead of causing system crashes.


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
