from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from config import Settings
from core import (
    AppError,
    NORTHWIND_DOWNLOAD_SHA,
    NORTHWIND_DOWNLOAD_URL,
    ResultArtifact,
    answer_question,
    artifact_describe_text,
    artifact_preview_rows,
    create_openai_client,
    download_northwind,
    export_artifact_chart,
    export_artifact_csv,
    export_artifact_report,
    export_workspace_report,
    index_schema,
    list_saved_workspaces,
    load_artifact_workspace,
    make_result_artifact,
    parse_colon_command,
    parse_count,
    parse_key_value_args,
    read_query_logs,
    route_artifact_followup,
    save_artifact_workspace,
    transform_artifact,
    verify_sqlite_database,
)

app = typer.Typer(add_completion=False, help="Local Text-to-SQL for any SQLite database.")
console = Console()

ARTIFACT_HELP = """Artifact commands (operate on the last result):
  :sql           Show the SQL for the last result
  :columns       Show the result column names
  :describe      Show a deterministic profile of the last result
  :head [N]      Show the first N rows (default 10)
  :tail [N]      Show the last N rows (default 10)
  :export [csv]  Export the result to OUTPUT_DIR as CSV
  :report <md|html> [all|workspace=<target>]   Export artifact report
  :plot ...      Save a chart to OUTPUT_DIR/charts (bar/line/scatter x=.. y=.. | hist column=..)
  :sort ...      New artifact sorted by a column (column=<c> order=<asc|desc>)
  :select ...    New artifact with a subset of columns (columns=<c1,c2,...>)
  :filter ...    New filtered artifact (column=<c> op=<eq|ne|gt|gte|lt|lte|contains> value=<v>)
  :groupby ...   New aggregated artifact (by=<c> metric=<c> agg=<sum|mean|count|min|max>)
  :save [name=<name>]  Save this session's artifacts to OUTPUT_DIR/workspaces
  :saved         List saved workspaces
  :load <workspace>    Load a saved workspace (exact name or unique prefix)
  :artifacts     List this session's results
  :help          Show this help
  :q             Quit"""


def load_settings(db: Path | None = None, metadata_db: Path | None = None) -> Settings:
    overrides: dict[str, Path] = {}
    if db is not None:
        overrides["db_path"] = db
    if metadata_db is not None:
        overrides["metadata_db_path"] = metadata_db
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("download-northwind")
def download_northwind_cmd(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing DB.")] = False,
) -> None:
    """Download the pinned Northwind SQLite database."""
    settings = load_settings()
    console.print(
        Panel.fit(
            f"Destination: {settings.source_db_path}\n"
            f"Source: {NORTHWIND_DOWNLOAD_URL}\n"
            f"Pinned commit: {NORTHWIND_DOWNLOAD_SHA}\nLicense: MIT upstream",
            title="Northwind Download",
        )
    )
    try:
        with console.status("Downloading and verifying Northwind database..."):
            path, table_count = download_northwind(settings, force=force)
    except AppError as exc:
        console.print(Panel(str(exc), title="Download failed", border_style="red"))
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(Panel(f"Download failed: {exc}", title="Download failed", border_style="red"))
        raise typer.Exit(1) from exc
    console.print(f"[green]Saved[/green] {path} with {table_count} tables.")


@app.command()
def index(
    db: Annotated[Path | None, typer.Option("--db", help="Path to a SQLite database file.")] = None,
    metadata_db: Annotated[
        Path | None,
        typer.Option("--metadata-db", hidden=True, help="Advanced: override the per-database metadata DB."),
    ] = None,
) -> None:
    """Inspect the database and build the sqlite-vec schema index."""
    settings = load_settings(db, metadata_db)
    try:
        verify_sqlite_database(settings.source_db_path)
        client = create_openai_client(settings)
        with console.status("Indexing schema with local embeddings..."):
            summary = index_schema(settings, client)
    except AppError as exc:
        console.print(Panel(str(exc), title="Index failed", border_style="red"))
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(Panel(f"Index failed: {exc}", title="Index failed", border_style="red"))
        raise typer.Exit(1) from exc
    console.print(
        Panel.fit(
            f"Indexed {summary.table_count} tables\n"
            f"Embedding dimension: {summary.embedding_dimension}\n"
            f"Metadata DB: {summary.metadata_path}",
            title="Index ready",
            border_style="green",
        )
    )


@app.command()
def ask(
    question: Annotated[str | None, typer.Argument(help="Optional one-shot question.")] = None,
    db: Annotated[Path | None, typer.Option("--db", help="Path to a SQLite database file.")] = None,
    metadata_db: Annotated[
        Path | None,
        typer.Option("--metadata-db", hidden=True, help="Advanced: override the per-database metadata DB."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show diagnostics.")] = False,
) -> None:
    """Ask a question once, or start the interactive prompt when no question is supplied."""
    settings = load_settings(db, metadata_db)
    try:
        verify_sqlite_database(settings.source_db_path)
        client = create_openai_client(settings)
    except AppError as exc:
        console.print(Panel(str(exc), title="Configuration error", border_style="red"))
        raise typer.Exit(1) from exc

    if question:
        result = run_question(settings, client, question, verbose)
        raise typer.Exit(0 if result.success else 1)

    console.print(
        Panel.fit(
            f"Database: {settings.source_db_path.name}\n"
            "Ask a question, or use :help for result commands. Type 'exit' to quit.",
            title="Text-to-SQL Assistant",
        )
    )
    artifacts: list[ResultArtifact] = []
    while True:
        user_input = Prompt.ask("[bold]You[/bold]").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        command = parse_colon_command(user_input)
        if command is not None:
            name, arg = command
            if name in {"q", "quit", "exit"}:
                break
            handle_artifact_command(name, arg, artifacts, settings, verbose)
            continue
        if artifacts:
            try:
                route = route_artifact_followup(user_input, artifacts[-1].columns)
            except AppError as exc:
                console.print(Panel(str(exc), title="Could not route", border_style="yellow"))
                continue
            if route is not None:
                target = f":{route.command}" if not route.arg else f":{route.command} {route.arg}"
                console.print(f"[dim]Routed to {target}[/dim]")
                handle_artifact_command(route.command, route.arg, artifacts, settings, verbose)
                continue
        result = run_question(settings, client, user_input, verbose)
        if result.success and result.sql and result.columns:
            artifacts.append(make_result_artifact(len(artifacts) + 1, result))


@app.command()
def logs(
    db: Annotated[Path | None, typer.Option("--db", help="Path to a SQLite database file.")] = None,
    metadata_db: Annotated[
        Path | None,
        typer.Option("--metadata-db", hidden=True, help="Advanced: override the per-database metadata DB."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=100)] = 10,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show recent query logs for the selected database."""
    settings = load_settings(db, metadata_db)
    rows = read_query_logs(settings, limit)
    if not rows:
        console.print(f"No query logs found for database: {settings.source_db_path.name}")
        return

    table = Table(title=f"Last {len(rows)} query logs")
    table.add_column("ID", justify="right")
    table.add_column("Time")
    table.add_column("Status")
    table.add_column("Question")
    table.add_column("SQL")
    table.add_column("Summary Mode")
    for row in rows:
        status = "success" if row["success"] else "failed"
        if row["cancelled"]:
            status = "cancelled"
        sql = row["generated_sql"] or ""
        if not verbose and len(sql) > 80:
            sql = sql[:77] + "..."
        table.add_row(
            str(row["id"]),
            str(row["created_at"]),
            status,
            preview(row["question"], 70 if not verbose else 200),
            sql,
            str(row["summary_mode"] or ""),
        )
    console.print(table)

    if verbose:
        for row in rows:
            console.print(
                Panel(
                    f"Question: {row['question']}\n\n"
                    f"SQL:\n{row['generated_sql'] or ''}\n\n"
                    f"Summary: {row['summary'] or ''}\n"
                    f"Error: {row['error_message'] or row['validation_error'] or ''}\n"
                    f"Shape warning: {row['shape_warning'] or ''}",
                    title=f"Log {row['id']}",
                )
            )


def run_question(settings: Settings, client, question: str, verbose: bool):
    def approve(sql: str) -> bool:
        console.print(Panel(Syntax(sql, "sql", theme="monokai", word_wrap=True), title="Validated SQL"))
        return Confirm.ask("Execute this query?", default=False)

    with console.status("Thinking through schema, SQL, and results..."):
        result = answer_question(
            settings,
            client,
            question,
            approval_callback=approve if settings.require_sql_approval else None,
        )
    render_answer(result, verbose)
    return result


def handle_artifact_command(
    name: str,
    arg: str,
    artifacts: list[ResultArtifact],
    settings: Settings,
    verbose: bool,
) -> None:
    if name == "help":
        console.print(Panel(ARTIFACT_HELP, title="Help"))
        return
    if name == "artifacts":
        render_artifacts(artifacts)
        return
    if name == "save":
        try:
            options = parse_key_value_args(arg)
            unknown = set(options) - {"name"}
            if unknown:
                raise AppError(f"Unknown option(s) for save: {', '.join(sorted(unknown))}")
            saved = save_artifact_workspace(settings, artifacts, name=options.get("name"))
        except AppError as exc:
            console.print(Panel(str(exc), title="Save failed", border_style="yellow"))
            return
        console.print(f"[green]Saved[/green] {saved.artifact_count} artifacts to {saved.path}")
        return
    if name == "saved":
        workspaces = list_saved_workspaces(settings)
        if not workspaces:
            console.print("No saved workspaces.")
            return
        table = Table(title=f"{len(workspaces)} saved workspace(s)")
        table.add_column("Name")
        table.add_column("Path")
        for path in workspaces:
            table.add_row(path.name, str(path))
        console.print(table)
        return
    if name == "load":
        target = arg.strip()
        if not target:
            console.print(Panel("Usage: :load <workspace>", title="Load", border_style="yellow"))
            return
        try:
            loaded = load_artifact_workspace(settings, target)
        except AppError as exc:
            console.print(Panel(str(exc), title="Load failed", border_style="yellow"))
            return
        artifacts[:] = list(loaded.artifacts)
        console.print(
            f"[green]Loaded[/green] {len(loaded.artifacts)} artifacts from {loaded.path}"
        )
        return
    if name == "report":
        try:
            tokens = arg.split()
            if not tokens:
                raise AppError("Missing report format. Use :report <md|html> [all] [workspace=<workspace>]")
            if len(tokens) > 2:
                raise AppError("Too many arguments for report. Use :report <md|html> [all] [workspace=<workspace>]")

            format_str = tokens[0]
            fmt_lower = format_str.strip().lower()
            if fmt_lower not in {"md", "markdown", "html"}:
                raise AppError(f"Unsupported report format: {format_str}")

            include_all = False
            workspace_target = None

            if len(tokens) == 2:
                opt = tokens[1]
                if opt == "all":
                    include_all = True
                elif opt.startswith("workspace="):
                    _, _, target = opt.partition("=")
                    target = target.strip()
                    if not target:
                        raise AppError("Empty workspace target. Use workspace=<workspace-name-or-prefix>")
                    workspace_target = target
                else:
                    raise AppError(f"Unknown option for report: {opt}")

            if workspace_target is not None:
                result = export_workspace_report(
                    settings,
                    workspace_target,
                    report_format=format_str,
                )
                console.print(
                    f"Saved {result.format} report for {result.artifact_count} workspace artifact(s) to {result.path}"
                )
                return

            if not artifacts:
                console.print("No result yet — ask a question first.")
                return

            result = export_artifact_report(
                settings,
                artifacts,
                report_format=format_str,
                include_all=include_all,
            )
            suffix_s = "s" if result.artifact_count != 1 else ""
            console.print(
                f"Saved {result.format} report for {result.artifact_count} artifact{suffix_s} to {result.path}"
            )
        except AppError as exc:
            console.print(Panel(str(exc), title="Report failed", border_style="yellow"))
        return
    if not artifacts:
        console.print("No result yet — ask a question first.")
        return
    last = artifacts[-1]
    if name == "sql":
        console.print(Panel(Syntax(last.sql, "sql", theme="monokai", word_wrap=True), title="SQL"))
    elif name == "columns":
        console.print(Panel("\n".join(last.columns) or "none", title="Columns"))
    elif name == "describe":
        try:
            console.print(Panel(artifact_describe_text(settings, last), title="Analysis"))
        except AppError as exc:
            console.print(Panel(str(exc), title="Analysis unavailable", border_style="yellow"))
    elif name in {"head", "tail"}:
        try:
            count = parse_count(arg)
        except AppError as exc:
            console.print(Panel(str(exc), title="Invalid count", border_style="yellow"))
            return
        render_rows(last.columns, artifact_preview_rows(last, name, count), truncated=False)
    elif name == "export":
        if arg not in {"", "csv"}:
            console.print(
                Panel(
                    "Only CSV export is supported currently. Use :export csv.",
                    title="Unsupported export",
                    border_style="yellow",
                )
            )
            return
        try:
            export = export_artifact_csv(settings, last, settings.output_path)
        except AppError as exc:
            console.print(Panel(str(exc), title="Export failed", border_style="yellow"))
            return
        console.print(f"[green]Exported[/green] {export.row_count} rows to {export.path}")
        if export.truncated:
            console.print(
                f"[yellow]Export was capped at MAX_ANALYSIS_ROWS={settings.max_analysis_rows}; "
                "more rows may exist.[/yellow]"
            )
    elif name == "plot":
        try:
            chart = export_artifact_chart(settings, last, arg)
        except AppError as exc:
            console.print(Panel(str(exc), title="Plot failed", border_style="yellow"))
            return
        console.print(f"[green]Saved chart[/green] {chart.path}")
        console.print(f"Chart type: {chart.chart_type}; rows plotted: {chart.row_count}")
    elif name in {"sort", "select", "filter", "groupby"}:
        try:
            transformed = transform_artifact(last, name, arg, len(artifacts) + 1)
        except AppError as exc:
            console.print(Panel(str(exc), title="Transform failed", border_style="yellow"))
            return
        new = transformed.artifact
        artifacts.append(new)
        console.print(
            f"[green]Created artifact #{new.artifact_id}[/green] with {transformed.row_count} rows."
        )
        render_rows(new.columns, new.rows[:10], truncated=False)
    else:
        console.print(Panel(ARTIFACT_HELP, title=f"Unknown command :{name}"))


def render_artifacts(artifacts: list[ResultArtifact]) -> None:
    if not artifacts:
        console.print("No artifacts yet.")
        return
    table = Table(title=f"{len(artifacts)} session artifact(s)")
    table.add_column("ID", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Truncated")
    table.add_column("Question")
    table.add_column("SQL")
    for artifact in artifacts:
        sql = artifact.sql if len(artifact.sql) <= 60 else artifact.sql[:57] + "..."
        table.add_row(
            str(artifact.artifact_id),
            str(len(artifact.rows)),
            "yes" if artifact.truncated else "no",
            preview(artifact.question, 50),
            sql,
        )
    console.print(table)


def render_answer(result, verbose: bool) -> None:
    if result.error_message and not result.success:
        console.print(Panel(result.error_message, title="Error", border_style="red"))
        if result.sql:
            console.print(Panel(Syntax(result.sql, "sql", theme="monokai", word_wrap=True), title="SQL"))
        return

    if result.freshness_warning:
        console.print(Panel(result.freshness_warning, title="Notice", border_style="yellow"))
    if result.shape_warning:
        console.print(Panel(result.shape_warning, title="Result warning", border_style="yellow"))

    console.print(Panel(", ".join(result.retrieved_tables) or "none", title="Retrieved schema"))
    if result.sql:
        console.print(Panel(Syntax(result.sql, "sql", theme="monokai", word_wrap=True), title="Generated SQL"))

    render_rows(result.columns, result.rows, result.truncated)
    if result.analysis_text:
        console.print(Panel(result.analysis_text, title="Analysis", border_style="cyan"))
    if result.analysis_error:
        console.print(
            Panel(result.analysis_error, title="Analysis unavailable", border_style="yellow")
        )
    if result.summary:
        console.print(Panel(result.summary, title="Summary", border_style="green"))

    if verbose:
        diagnostics = [
            f"Expanded tables: {', '.join(result.expanded_tables) or 'none'}",
            f"Auto reindexed: {result.auto_reindexed}",
            f"Repaired: {result.repaired}",
            f"Repair reason: {result.repair_reason or ''}",
            f"Execution ms: {result.execution_ms or 0:.2f}",
            f"Token usage: prompt={result.token_usage.prompt_tokens}, "
            f"completion={result.token_usage.completion_tokens}, total={result.token_usage.total_tokens}",
            f"Shape expected: {result.shape_expected}",
            f"Shape observed: {result.shape_observed}",
            f"Analysis: {result.analysis_text or result.analysis_error or 'disabled'}",
        ]
        console.print(Panel("\n".join(diagnostics), title="Diagnostics"))


def render_rows(columns: tuple[str, ...], rows: tuple[tuple[object, ...], ...], truncated: bool) -> None:
    if not columns:
        console.print("No result columns.")
        return
    table = Table(title="Results")
    for column in columns:
        table.add_column(str(column))
    for row in rows:
        table.add_row(*(format_cell(value) for value in row))
    console.print(table)
    if truncated:
        console.print("[yellow]Rows were truncated by MAX_RESULT_ROWS.[/yellow]")


def format_cell(value: object) -> str:
    if value is None:
        return "NULL"
    return str(value)


def preview(value: str, length: int) -> str:
    return value if len(value) <= length else value[: length - 3] + "..."


if __name__ == "__main__":
    app()
