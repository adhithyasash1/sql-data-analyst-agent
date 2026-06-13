# SQL Data Analyst Agent

A local, privacy-first **Text-to-SQL CLI for any SQLite database** — with deterministic result
analysis, charts, saved workspaces, and gated machine-learning runs layered on top.

Ask a question in plain English. The agent retrieves the relevant schema with vector search,
generates a `SELECT` query with a locally served model, validates it with SQLGlot, executes it on
a **read-only** connection, and renders the result with Rich. Everything after that — profiling,
transformations, correlation, charts, reports, even supervised ML — is **deterministic Python over
the rows you already fetched**: no model-generated code is ever executed.

On top of that sits an **opt-in agentic layer**: query-log memory, schema exploration tools, a
result critic, a natural-language router fallback, multi-step question decomposition, goal-driven
analysis, and a full bounded agent loop (`ask --agent`). In every one of them the model only
*chooses among* the same validated, read-only, budget-capped actions — it never gains new powers.

```text
> Which genres generated the most revenue?
  ┌────────────┬──────────────┬────────────┐
  │ GenreName  │ TotalRevenue │ TrackCount │
  ├────────────┼──────────────┼────────────┤
  │ Rock       │ 826.65       │ 835        │
  │ ...        │ ...          │ ...        │
> sort by TotalRevenue descending        ← routed to :sort, no model call
> :plot bar x=GenreName y=TotalRevenue   ← chart PNG, no model call
> :save name=genre_revenue               ← workspace on disk, reload anytime
```

## Highlights

- **Works with any SQLite file** — pass `--db <path>`; a bundled Northwind demo is the default.
- **Read-only by construction** — `mode=ro` connections, write-SQL rejection, row limits, timeouts.
- **Schema-aware generation** — table/view documents with bounded sample profiles, embedded via a
  local OpenAI-compatible endpoint and retrieved with sqlite-vec, expanded one hop through foreign keys.
- **Self-checking** — structural validation, schema checks, result-shape heuristics, and a bounded
  repair loop where each retry sees the history of failed SQL and errors.
- **Opt-in agentic layer** — few-shot memory from query logs, read-only schema exploration, an LLM
  result critic, `:multi` decomposition, `:goal` analysis loops, and `ask --agent` — all default-off,
  budget-bounded, and confined to the same gated actions as typed commands.
- **Deterministic analysis layer** — profile, sort/select/filter/groupby, Pearson correlation,
  charts, and fixed-recipe supervised ML (preflight-gated, opt-in `--run`).
- **Persistent workspaces and reports** — save session artifacts to disk, reload them later, and
  export Markdown/HTML reports fully offline.
- **Per-database isolation** — each source database gets its own metadata file (index, vectors,
  query logs) keyed by absolute path, so databases never share state.

## How it works

The runtime path is intentionally bounded:

1. Retrieve table/view-level schema objects via vector search.
2. Expand one hop through foreign-key relationships.
3. Generate candidate SQL with the local model.
4. Validate structural safety with SQLGlot plus schema checks.
5. Execute on a read-only SQLite connection with timeout and row limits.
6. Optionally repair within the `MAX_REPAIR_ATTEMPTS` budget on validation failure or a
   high-confidence result-shape mismatch — each retry prompt carries the accumulated history of
   failed SQL and errors.
7. Show the SQL, the result table, and a concise summary.

Validation proves the SQL is structurally safe and executable — not that it perfectly answers the
question. Result-shape heuristics warn when the answer's shape looks wrong for the question.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- A local oMLX OpenAI-compatible API at `OMLX_BASE_URL` serving:
  - `MLX-Qwopus3.5-9B-v3-4bit` (SQL generation and summaries)
  - `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` (schema embeddings)

Generation settings are sent per request, so SQL determinism does not depend on server-wide defaults.

## Quick start

```bash
uv sync
cp .env.example .env          # set OMLX_API_KEY to any non-empty value your server accepts

# optional: fetch the pinned Northwind demo database
uv run python app.py download-northwind

# build the schema index, then ask
uv run python app.py index
uv run python app.py ask "Which customers placed the most orders?"

# or start an interactive session
uv run python app.py ask
```

### Use your own database

The app accepts **any SQLite database file** — it is validated by opening it, not by extension.
Place `--db` before the question argument:

```bash
uv run python app.py ask --db /path/to/your.db "List the 10 most recent orders"
```

The first run for a new database builds its schema index automatically (when `AUTO_REINDEX=true`).
The index and query logs live in a per-database metadata file under `data/metadata/`, keyed by the
database's absolute path — switching databases never re-embeds or mixes logs. If only row data
changes, the app warns but does not reindex; if the table schema changes, `ask` auto-reindexes.

### Optional extras

| Extra | Installs | Enables |
| --- | --- | --- |
| `analysis` | pandas, pyarrow | result profiling panel (`ENABLE_DATAFRAME_ANALYSIS=true`), `:describe` fallback |
| `viz` | matplotlib | `:plot` charts |
| `ml` | numpy, pandas, pyarrow, scikit-learn | `:analyze predict/explain ... --run` supervised execution |

```bash
uv sync --extra analysis --extra viz   # combine as needed
```

Every extra is optional: if a command needs a missing extra, the session prints a notice and
continues instead of failing.

## CLI commands

| Command | Purpose |
| --- | --- |
| `index [--db <path>]` | Inspect the database and build/rebuild its schema index |
| `ask [--db <path>] ["<question>"]` | Answer one question, or start the interactive session when no question is given |
| `ask --agent "<question>"` | Answer with the bounded multi-step agent loop (explore, query, analyze) |
| `logs [--db <path>] [--limit N] [--verbose]` | Show recent query logs for the selected database |
| `download-northwind` | Download the pinned MIT-licensed Northwind demo database |

## Interactive session

Inside `ask`, each successful query stores an in-memory **result artifact**. Colon commands operate
on artifacts deterministically — apart from the explicitly agentic `:multi` and `:goal`, they never
call the model, and none of them ever execute generated code.
**Artifacts and analysis digests stay in memory until you explicitly export or `:save`.**

The session is interrupt-safe: **Ctrl+C** at the prompt re-prompts, **Ctrl+C** during a running
question cancels just that question, and **Ctrl+D** (or `:q`) exits — none of them crash the
session or lose your in-memory artifacts.

### Inspect the latest result

| Command | Action |
| --- | --- |
| `:sql` | Show the SQL behind the last result |
| `:columns` | Show the column names |
| `:describe` | Deterministic profile (shape, dtypes, nulls, distinct, min/max/mean) |
| `:head [N]` / `:tail [N]` | First / last N rows |
| `:artifacts` | List all session artifacts |
| `:export csv` | Export the last result as CSV to `OUTPUT_DIR/` |
| `:help` | Show command help · `:q` quits |

`:export csv` performs one bounded read-only re-fetch (up to `MAX_ANALYSIS_ROWS`, reusing the
already-validated SQL) when the displayed result was truncated, and reports the exact row count
and whether the export was capped.

### Transform

Each transformation creates a **new in-session artifact** that becomes the latest, so `:describe`,
`:plot`, `:export csv`, `:head`, and `:tail` then operate on the transformed result:

| Command | Action |
| --- | --- |
| `:sort column=<col> order=<asc\|desc>` | Sort by a column (default `asc`; missing values last) |
| `:select columns=<col1,col2,...>` | Keep only those columns, in that order |
| `:filter column=<col> op=<eq\|ne\|gt\|gte\|lt\|lte\|contains> value=<v>` | Keep matching rows |
| `:groupby by=<col> metric=<col> agg=<sum\|mean\|count\|min\|max>` | Aggregate (`metric` optional for `count`) |

Arguments are whitespace-separated `key=value` tokens, so values with spaces are not supported yet,
and `:select` columns must be comma-separated without spaces. Transforms operate on the stored
artifact snapshot only; if the source was truncated at `MAX_RESULT_ROWS`, the new artifact's
question records an `[... applied to truncated artifact]` note.

### Chart

```text
:plot bar x=GenreName y=TotalRevenue
:plot line x=OrderDate y=Freight
:plot scatter x=UnitPrice y=Quantity
:plot hist column=TotalRevenue
```

PNGs are written to `OUTPUT_DIR/charts/` as `chart_001.png`, `chart_002.png`, … Charts use **only
the rows stored in the artifact** (no re-fetch, no SQL, no model). For bar/line, `x` may be
text/date/numeric and `y` must be numeric; scatter needs both numeric; hist needs one numeric
column. Because options are `key=value` tokens, column names must not contain spaces — alias them
in SQL first (`SELECT Total AS RevenueTotal ...`). Requires the `viz` extra.

### Natural-language follow-ups

When a result exists and your text clearly refers to it, the session routes it to the matching
artifact command and prints `Routed to :<command>`:

```text
describe this result            → :describe
show first 5 rows               → :head 5
export to csv                   → :export csv
histogram of TotalRevenue       → :plot hist column=TotalRevenue
sort by TotalRevenue descending → :sort column=TotalRevenue order=desc
filter TotalRevenue greater than 100 → :filter column=TotalRevenue op=gt value=100
where GenreName contains Rock   → :filter column=GenreName op=contains value=Rock
select GenreName, TotalRevenue  → :select columns=GenreName,TotalRevenue
count by Country                → :groupby by=Country agg=count
```

This is a **deterministic router**, not a model call. Column names are matched case-insensitively
against the **latest artifact's** columns (not the database schema), there is no fuzzy matching,
and anything ambiguous or unrecognized is treated as a new database question — `top 5 genres by
revenue` still runs the Text-to-SQL pipeline. The router is interactive-only; one-shot
`ask "<question>"` never routes.

With `ENABLE_LLM_ROUTER_FALLBACK=true`, a follow-up the deterministic router cannot parse (e.g.
`keep only rows where the freight is above 400000`) is offered to the model, which may map it to
one allowlisted artifact command via a strict JSON reply — the session then prints
`Routed to :<command> (model-routed)` and runs the **same deterministic handler** as a typed
command. Only read-only/transform commands are routable (never workspace save/load/delete, never
`:analyze`); anything the model cannot map cleanly falls through to a new database question.

### Analyze

`:analyze <request>` plans or runs a deeper deterministic analysis of the latest artifact:

```text
:analyze profile this result
:analyze correlate numeric columns
:analyze predict TotalRevenue          # preflight only
:analyze predict TotalRevenue --run    # fit the fixed baseline recipe
:analyze explain TotalRevenue --run    # adds permutation importance
:analyses                              # list executed digests in this session
```

- **profile** runs immediately over the stored rows.
- **correlate** computes a pure-Python Pearson correlation table. If the artifact is truncated, it
  performs one bounded read-only re-fetch up to `MAX_ANALYSIS_ROWS`, then warns if still capped.
- **predict / explain** run a supervised **preflight only**: problem-type inference (regression vs.
  classification), hard blocker gates, high-confidence leakage checks, eligible features, and a
  fixed baseline recipe. No model is fitted without `--run`.
- **`--run`** (requires the `ml` extra) fits the fixed recipe after preflight passes. Regression:
  median baseline + Ridge, reporting MAE/R². Classification: majority-class baseline +
  LogisticRegression, reporting accuracy/balanced accuracy. Missing features are imputed with
  missingness indicators; rows with missing targets are dropped and counted. `explain --run` adds
  test-set permutation importance only when the model beats the baseline. The digest is recorded as
  an analysis artifact; trained models, predictions, and materialized rows are ephemeral.

After each successful answer the app also computes a lightweight `DatasetCapability` (pure Python,
no extras) and renders a conservative **Recommended next steps** panel: profile first, then a
direct chart when the shape supports one, then correlation when two numeric columns exist.
Supervised ML is always an explicit, last-rung action — it is never auto-run.

### Save, load, and report

| Command | Action |
| --- | --- |
| `:save [name=<name>]` | Save session artifacts and executed analysis digests to a timestamped workspace |
| `:saved` | List saved workspaces |
| `:workspace-info <ws>` | Show details of a saved workspace |
| `:load <ws>` | Load a workspace (exact name or unique prefix) |
| `:delete-workspace <ws>` | Delete a workspace directory (no confirmation prompt) |
| `:report <md\|html> [all]` | Report the latest artifact, or all session artifacts |
| `:report <md\|html> workspace=<ws>` | Report a saved workspace |

Workspaces live under `OUTPUT_DIR/workspaces/` (each holds `manifest.json`, per-artifact
CSV/SQL/profile files, and per-analysis JSON/Markdown digests). Reports land in
`OUTPUT_DIR/reports/` as `report_<timestamp>.<md|html>`, with numeric suffixes to avoid overwrites
and previews capped at 50 rows. Notes:

- **Saving is explicit — there is no autosave.** Save, load, info, report, and delete never call
  the model and never run SQL.
- Workspace names resolve by exact name or unique prefix; ambiguous prefixes list candidates
  instead of guessing. Absolute paths, separators, and `..` traversal are rejected, and deletion is
  confined to the target directory under `OUTPUT_DIR/workspaces/`.
- **Loaded CSV values are strings** (CSV does not preserve types). Deterministic commands still
  work on loaded artifacts — `:filter ... op=gt value=100` and `:plot` parse numeric strings — but
  original dtypes are not restored. Malformed workspace CSVs (row width ≠ header) are rejected at
  load with a clear error.
- `:report markdown ...` is an alias for `:report md ...`.

### Example workflows

```text
# 1. Transform and chart
Which genres generated the most revenue?
filter TotalRevenue greater than 100
sort by TotalRevenue descending
:plot bar x=GenreName y=TotalRevenue

# 2. Save and resume
:save name=genre_revenue
:saved
:load genre_revenue
:artifacts

# 3. Report a saved workspace
:report md workspace=genre_revenue
:report html workspace=genre_revenue
```

## Agentic features (all opt-in)

The agentic layer lets the model *choose* what to do next — but every choice is validated against
an allowlist, executed by the same gated code paths as typed commands, and bounded by a step
budget. Default behavior with every flag off is identical to the non-agentic releases. All JSON
replies from the model are parsed best-effort: a model error, an unparseable reply, or an
out-of-allowlist action simply degrades to the plain pipeline instead of failing.

### Pipeline upgrades (env flags)

| Flag | What it adds |
| --- | --- |
| `ENABLE_FEWSHOT_MEMORY` | Injects up to `MAX_FEWSHOT_EXAMPLES` past **successful** question→SQL pairs from this database's query log into the SQL prompt, ranked by deterministic token overlap — no embeddings, no extra model call. The agent gets better on *your* database as the log grows. |
| `ENABLE_SCHEMA_EXPLORATION` | Before SQL is written, the model may call up to `MAX_EXPLORATION_CALLS` read-only tools — `list_tables`, `describe_table`, `sample_rows` (≤5 rows), `distinct_values` (≤10 values) — to ground its query in real data (e.g. discovering that `Discontinued` is 0/1). Table and column names resolve against the inspected schema; the only SQL shapes are fixed `SELECT ... LIMIT` templates through the read-only executor. |
| `ENABLE_LLM_RESULT_CRITIC` | After execution, a reflection call judges whether the result plausibly answers the question, grounded in the SQL and a bounded result digest. A mismatch verdict is **advisory**: it surfaces through the existing shape-warning channel and never re-runs anything. |
| `ENABLE_LLM_ROUTER_FALLBACK` | Model-assisted routing of fuzzy follow-ups to allowlisted artifact commands (see Natural-language follow-ups above). |

### Multi-step commands

```text
:multi <question>     # interactive: decompose → answer each sub-question → synthesize
:goal <objective>     # interactive: model-chosen deterministic steps over the latest result
ask --agent "<q>"     # one-shot CLI: full explore/query/analyze agent loop
```

- **`:multi`** asks the model to split a complex question into at most 3 plain-English
  sub-questions, runs each through the full validated Text-to-SQL pipeline (each result becomes a
  normal session artifact), then renders a grounded cross-result synthesis that quotes numbers
  exactly as returned.
- **`:goal`** runs a bounded loop (`MAX_GOAL_STEPS`) in which the model picks among `describe`,
  `head`, `sort`, `select`, `filter`, and `groupby` on the latest artifact, observes each result
  digest, and finishes with findings. Transforms land in the session artifact list like any typed
  command; failed actions feed their error text back as observations so the model can adjust.
- **`ask --agent`** is the umbrella loop (`MAX_AGENT_STEPS`): schema exploration tools, `run_sql`
  (full SQLGlot validation, read-only executor, `REQUIRE_SQL_APPROVAL` honored per query), and the
  artifact transforms — ending in a grounded answer. It can answer multi-hop questions a single
  SELECT cannot (e.g. *"Which employee handled the most orders, and what were that employee's
  three biggest orders by freight?"*). Each step is printed; a repeat-guard stops the loop if the
  model re-runs an identical action.

## Configuration

Settings load from `.env` (see `.env.example`). The most useful ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMLX_BASE_URL` | `http://127.0.0.1:8000/v1` | Local OpenAI-compatible endpoint |
| `OMLX_API_KEY` | – | Any non-empty value your server accepts |
| `DB_PATH` | `data/northwind.db` | Default database when `--db` is omitted |
| `METADATA_DB_PATH` | auto per database | Debug override for the metadata DB; leave unset normally |
| `AUTO_REINDEX` | `true` | Rebuild the index when schema/model/path changes |
| `MAX_RESULT_ROWS` | `50` | Display row cap per result |
| `QUERY_TIMEOUT_MS` | `3000` | Read-only execution timeout |
| `MAX_REPAIR_ATTEMPTS` | `1` | Total SQL repair budget (0–3); each retry sees prior failed attempts |
| `REQUIRE_SQL_APPROVAL` | `false` | Ask before executing validated SQL |
| `ENABLE_RESULT_SHAPE_CHECK` | `true` | Result-shape heuristics (disable if noisy on unusual schemas) |
| `ENABLE_LLM_SUMMARY` | `false` | Optional second model call for a grounded narrative summary |
| `ENABLE_QUERY_LOGGING` | `true` | Store local query logs in the metadata DB (best-effort: a failed log write never breaks an answer) |
| `ENABLE_SCHEMA_PROFILING` | `true` | Add bounded row counts and sample values to schema docs |
| `MAX_PROFILE_VALUES` | `3` | Sample values per column during profiling |
| `MAX_PROFILE_TEXT_LENGTH` | `80` | Max characters per sampled text value |
| `ENABLE_DATAFRAME_ANALYSIS` | `false` | Post-answer profiling panel (needs the `analysis` extra) |
| `MAX_ANALYSIS_ROWS` | `5000` | Cap for bounded analysis re-fetches |
| `MAX_ANALYSIS_COLUMNS` | `30` | Max columns in the analysis panel / summary grounding |
| `ENABLE_FEWSHOT_MEMORY` | `false` | Inject past successful question→SQL pairs into the SQL prompt |
| `MAX_FEWSHOT_EXAMPLES` | `3` | Max few-shot examples per prompt |
| `ENABLE_LLM_ROUTER_FALLBACK` | `false` | Model-assisted routing of fuzzy follow-ups to artifact commands |
| `ENABLE_LLM_RESULT_CRITIC` | `false` | Advisory post-execution verdict on whether the result answers the question |
| `ENABLE_SCHEMA_EXPLORATION` | `false` | Pre-generation read-only schema tool loop |
| `MAX_EXPLORATION_CALLS` | `4` | Exploration tool-call budget (1–8) |
| `MAX_GOAL_STEPS` | `6` | `:goal` step budget (1–10) |
| `MAX_AGENT_STEPS` | `8` | `ask --agent` step budget (1–12) |
| `OUTPUT_DIR` | `data/outputs` | CSV exports, `charts/`, `reports/`, `workspaces/` (gitignored) |

Schema profiling reads only through the read-only connection and never modifies the source
database; views are profiled lightly (sample values only, no `COUNT(*)`/`MIN`/`MAX`). When
`ENABLE_DATAFRAME_ANALYSIS=true`, results become a `pyarrow.Table` + pandas DataFrame and a
deterministic profile is shown in an **Analysis** panel (and grounds the summary when
`ENABLE_LLM_SUMMARY=true`). The Arrow table is an ADBC-ready substrate for a future
multi-database backend.

## Safety guarantees

- **Read-only SQLite connection** — `mode=ro`, extension loading disabled, row limits, and a
  progress-handler timeout on every query.
- **Write SQL rejected** — validation blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`,
  `CREATE`, `PRAGMA`, `ATTACH`, `DETACH`, comments, and system tables.
- **No generated code execution** — model output is only ever treated as SQL to validate, never as
  code to run.
- **Deterministic artifact scope** — `:sort`, `:select`, `:filter`, `:groupby`, `:plot`, and the
  natural-language router operate on already-validated in-memory rows only.
- **Bounded analysis scope** — `:analyze` profile/correlation/preflight/`--run` use deterministic
  Python; only explicit `correlate`/`predict`/`explain` (and `:export csv`) may re-run the
  validated SQL through the read-only executor, capped at `MAX_ANALYSIS_ROWS`, and only when the
  stored artifact was truncated.
- **Confined workspace operations** — reads, writes, and deletions are restricted to
  `OUTPUT_DIR/workspaces/`; absolute paths, separators, and `..` traversal are rejected.
- **Offline reports** — exports use stored rows and saved digests only; no SQL re-runs, no model calls.
- **Gated agentic actions** — every opt-in agentic feature (router fallback, exploration, critic,
  `:multi`, `:goal`, `ask --agent`) only lets the model *choose among* the existing validated,
  read-only operations: identifiers resolve against the inspected schema, every SQL passes the
  same validator and read-only executor, `REQUIRE_SQL_APPROVAL` is honored per query, and every
  loop has a hard step budget.
- **Friendly failures** — model/connection problems surface as readable `AppError` notices, not
  stack traces; keyboard interrupts cancel cleanly without losing the session; query logging is
  best-effort and never fails an answer.

## Architecture

| File | Responsibility |
| --- | --- |
| `app.py` | Typer commands, interactive prompt, Rich rendering |
| `core.py` | Schema inspection, indexing, retrieval, prompting, SQL validation, read-only execution, artifacts, workspaces, reports, agentic loops (exploration, goal, agent) |
| `analysis/` | Deterministic analysis: capability detection, planning, profile/correlation execution, supervised preflight and fixed-recipe ML runs |
| `config.py` | pydantic-settings configuration loaded from `.env` |
| `data/metadata/` | Per-database schema index, sqlite-vec vectors, index state, query logs |
| `evals/` | JSONL question/expectation sets for manual result-shape regression checks (no runner yet) |

## Testing

```bash
uv run pytest
```

The suite covers SQL validation, routing, transformations, workspaces, reports, and the analysis
package, and runs fully offline — no model server required.

## Known limitations

- Runtime validation is structural, not a semantic correctness judge; shape checks are
  conservative heuristics and may warn on valid answers.
- Queries are read-only and `SELECT`-only; tables and views are indexed.
- No live oMLX integration tests; no full prompt or result-row logging.
- Supervised `:analyze` fits models only with `--run`, and persists digests only — never trained
  models or row-level predictions.
- `:analyze correlate` may describe a capped subset when a bounded re-fetch still hits
  `MAX_ANALYSIS_ROWS`.
- Result analysis is descriptive profiling, not statistical inference.
- Agentic features depend on the local model following a JSON action protocol. Replies are parsed
  best-effort (reasoning preamble and multiple JSON objects are tolerated); when a reply cannot be
  used, the feature quietly degrades — no route, no critique, no decomposition — rather than
  erroring. Quality varies with the model served.
- `ask --agent` prints its steps but does not record them in the query log; only the standard
  pipeline logs queries.

## Sources

- Northwind SQLite database: https://github.com/jpwhite3/northwind-SQLite3 (MIT, pinned to commit `4f56e7f`)
- sqlite-vec: https://alexgarcia.xyz/sqlite-vec/
- SQLGlot: https://sqlglot.com/sqlglot.html
- SQLite URI / read-only mode: https://www.sqlite.org/uri.html
- Python sqlite3: https://docs.python.org/3/library/sqlite3.html
