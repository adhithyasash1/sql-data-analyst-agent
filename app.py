from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from analysis.capabilities import analyze_dataset_capability, suggest_analyses
from analysis.executor import AnalysisExecutionResult, execute_analysis_plan
from analysis.ml_runner import SupervisedRunResult, run_supervised_model
from analysis.planner import AnalysisPlan, plan_analysis
from analysis.supervised import SupervisedPreflight, build_supervised_preflight
from config import Settings
from core import (
    AnalysisArtifact,
    AnalysisArtifactTable,
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
    inspect_workspace,
    delete_workspace,
    list_saved_workspaces,
    load_artifact_workspace,
    make_result_artifact,
    materialize_artifact_rows,
    parse_colon_command,
    parse_count,
    parse_key_value_args,
    read_query_logs,
    route_artifact_followup,
    save_artifact_workspace,
    transform_artifact,
    utc_now,
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
  :export csv    Export the result to OUTPUT_DIR as CSV
  :plot ...      Save a chart to OUTPUT_DIR/charts (bar/line/scatter x=.. y=.. | hist column=..)
  :sort ...      New sorted artifact (column=<col> order=<asc|desc>)
  :select ...    New artifact with subset of columns (columns=<col1,col2,...>)
  :filter ...    New filtered artifact (column=<col> op=<op> value=<val>)
  :groupby ...   New aggregated artifact (by=<col> metric=<col> agg=<agg>)
  :analyze ...   Plan or run deterministic analysis for the latest result (--run for ML)
  :save [name=<name>]  Save session artifacts and executed analysis digests
  :saved         List saved workspaces
  :workspace-info <workspace>    Show saved workspace details
  :delete-workspace <workspace>  Delete a saved workspace
  :load <workspace>              Load saved artifacts and analysis digests
  :report <md|html> [all|workspace=<workspace>]  Export artifact report
  :artifacts     List this session's results
  :analyses      List executed analyses recorded in this session
  :help          Show this help
  :q / quit / exit  Quit the assistant"""


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
    analyses: list[AnalysisArtifact] = []
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
            handle_artifact_command(name, arg, artifacts, analyses, settings, verbose)
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
                handle_artifact_command(
                    route.command,
                    route.arg,
                    artifacts,
                    analyses,
                    settings,
                    verbose,
                )
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
    analyses: list[AnalysisArtifact],
    settings: Settings,
    verbose: bool,
) -> None:
    if name == "help":
        console.print(Panel(ARTIFACT_HELP, title="Help"))
        return
    if name == "artifacts":
        render_artifacts(artifacts)
        return
    if name == "analyses":
        render_analyses(analyses)
        return
    if name == "save":
        try:
            options = parse_key_value_args(arg)
            unknown = set(options) - {"name"}
            if unknown:
                raise AppError(f"Unknown option(s) for save: {', '.join(sorted(unknown))}")
            saved = save_artifact_workspace(
                settings,
                artifacts,
                name=options.get("name"),
                analyses=analyses,
            )
        except AppError as exc:
            console.print(Panel(str(exc), title="Save failed", border_style="yellow"))
            return
        console.print(
            f"[green]Saved[/green] {saved.artifact_count} artifacts and "
            f"{saved.analysis_count} analyses to {saved.path}"
        )
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
    if name == "workspace-info":
        target = arg.strip()
        if not target:
            console.print(Panel("Usage: :workspace-info <workspace>", title="Workspace Info", border_style="yellow"))
            return
        try:
            info = inspect_workspace(settings, target)
        except AppError as exc:
            console.print(Panel(str(exc), title="Inspection failed", border_style="yellow"))
            return
        table = Table(title=f"Workspace: {info.name}")
        table.add_column("Field", style="bold cyan")
        table.add_column("Value")
        table.add_row("Workspace", info.name)
        table.add_row("Path", str(info.path))
        table.add_row("Created", info.created_at or "n/a")
        table.add_row("Artifacts", str(info.artifact_count))
        table.add_row("Rows", str(info.row_count))
        table.add_row("Files", ", ".join(info.files))
        console.print(table)
        return
    if name == "delete-workspace":
        target = arg.strip()
        if not target:
            console.print(Panel("Usage: :delete-workspace <workspace>", title="Delete Workspace", border_style="yellow"))
            return
        try:
            res = delete_workspace(settings, target)
            console.print(f"Deleted workspace {res.name} at {res.path}")
        except AppError as exc:
            console.print(Panel(str(exc), title="Delete failed", border_style="yellow"))
            return
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
        analyses[:] = list(loaded.analyses)
        console.print(
            f"[green]Loaded[/green] {len(loaded.artifacts)} artifacts and "
            f"{len(loaded.analyses)} analyses from {loaded.path}"
        )
        return
    if name == "report":
        try:
            tokens = arg.split()
            if not tokens:
                raise AppError("Usage: :report <md|html> [all|workspace=<workspace>]")
            if len(tokens) > 2:
                raise AppError("Usage: :report <md|html> [all|workspace=<workspace>]")

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
                    raise AppError(f"Unknown option(s) for report: {opt}")


            if workspace_target is not None:
                result = export_workspace_report(
                    settings,
                    workspace_target,
                    report_format=format_str,
                )
                analysis_word = "analysis" if result.analysis_count == 1 else "analyses"
                console.print(
                    f"Saved {result.format} report for {result.artifact_count} workspace artifact(s) "
                    f"and {result.analysis_count} {analysis_word} to {result.path}"
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
                analyses=analyses,
            )
            suffix_s = "s" if result.artifact_count != 1 else ""
            analysis_word = "analysis" if result.analysis_count == 1 else "analyses"
            console.print(
                f"Saved {result.format} report for {result.artifact_count} artifact{suffix_s} "
                f"and {result.analysis_count} {analysis_word} to {result.path}"
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
    elif name == "analyze":
        if not arg.strip():
            console.print(
                Panel(
                    "Usage: :analyze <profile|correlate|predict <column>|explain <column>> [--run]",
                    title="Analyze",
                    border_style="yellow",
                )
            )
            return
        analysis_request, run_requested = parse_analysis_run_request(arg)
        capability = analyze_dataset_capability(last.columns, last.rows, truncated=last.truncated)
        plan = plan_analysis(analysis_request, capability)
        if plan.recipe == "profile" and plan.status == "ready":
            result = execute_analysis_plan(plan, capability, last.columns, last.rows)
            render_analysis_execution(result)
            record_analysis_execution_result(analyses, last, result)
        elif plan.recipe == "correlation" and plan.status == "ready":
            analysis_columns = last.columns
            analysis_rows = last.rows
            analysis_capability = capability
            analysis_plan = plan
            allow_bounded_rows = False
            extra_warnings: tuple[str, ...] = ()

            if plan.row_scope == "bounded_refetch":
                try:
                    analysis_columns, analysis_rows, analysis_truncated = (
                        materialize_artifact_rows(settings, last)
                    )
                except AppError as exc:
                    console.print(Panel(str(exc), title="Analyze failed", border_style="yellow"))
                    return
                analysis_capability = analyze_dataset_capability(
                    analysis_columns,
                    analysis_rows,
                    truncated=analysis_truncated,
                )
                analysis_plan = plan_analysis(analysis_request, analysis_capability)
                allow_bounded_rows = True
                extra_warnings = (
                    f"Analysis used up to MAX_ANALYSIS_ROWS={settings.max_analysis_rows} "
                    "rows from the validated SQL.",
                )
                if analysis_truncated:
                    extra_warnings += (
                        f"Result was capped at MAX_ANALYSIS_ROWS={settings.max_analysis_rows}; "
                        "more rows may exist.",
                    )

            result = execute_analysis_plan(
                analysis_plan,
                analysis_capability,
                analysis_columns,
                analysis_rows,
                allow_bounded_rows=allow_bounded_rows,
            )
            if extra_warnings:
                result = replace(result, warnings=result.warnings + extra_warnings)
            render_analysis_execution(result)
            record_analysis_execution_result(analyses, last, result)
        elif plan.recipe in {"predict", "explain"}:
            analysis_columns = last.columns
            analysis_rows = last.rows
            analysis_capability = capability
            analysis_plan = plan
            analysis_row_scope = plan.row_scope
            extra_warnings: tuple[str, ...] = ()

            if plan.target_column is not None and plan.row_scope == "bounded_refetch":
                try:
                    analysis_columns, analysis_rows, analysis_truncated = (
                        materialize_artifact_rows(settings, last)
                    )
                except AppError as exc:
                    console.print(Panel(str(exc), title="Analyze failed", border_style="yellow"))
                    return
                analysis_capability = analyze_dataset_capability(
                    analysis_columns,
                    analysis_rows,
                    truncated=analysis_truncated,
                )
                analysis_plan = plan_analysis(analysis_request, analysis_capability)
                analysis_row_scope = "bounded_refetch"
                extra_warnings = (
                    f"Preflight used up to MAX_ANALYSIS_ROWS={settings.max_analysis_rows} "
                    "rows from the validated SQL.",
                )
                if analysis_truncated:
                    extra_warnings += (
                        f"Result was capped at MAX_ANALYSIS_ROWS={settings.max_analysis_rows}; "
                        "more rows may exist.",
                    )

            if analysis_plan.target_column is None:
                render_analysis_plan(analysis_plan)
                return

            preflight = build_supervised_preflight(
                analysis_plan,
                analysis_capability,
                analysis_columns,
                analysis_rows,
                row_scope=analysis_row_scope,
                extra_warnings=extra_warnings,
            )
            if run_requested:
                if preflight.can_execute:
                    run_result = run_supervised_model(
                        preflight,
                        analysis_capability,
                        analysis_columns,
                        analysis_rows,
                    )
                    render_supervised_run_result(run_result)
                    record_supervised_run_result(analyses, last, run_result)
                else:
                    render_supervised_preflight(preflight)
                    console.print("[yellow]Run blocked by supervised preflight.[/yellow]")
            else:
                render_supervised_preflight(preflight)
        else:
            render_analysis_plan(plan)
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


def render_analyses(analyses: list[AnalysisArtifact]) -> None:
    if not analyses:
        console.print("No executed analyses yet.")
        return
    table = Table(title=f"{len(analyses)} session analysis artifact(s)")
    table.add_column("ID", justify="right")
    table.add_column("Source", justify="right")
    table.add_column("Recipe")
    table.add_column("Status")
    table.add_column("Title")
    table.add_column("Created")
    for analysis in analyses:
        table.add_row(
            str(analysis.analysis_id),
            f"#{analysis.source_artifact_id}",
            analysis.recipe,
            analysis.status,
            preview(analysis.title, 48),
            analysis.created_at,
        )
    console.print(table)


def record_analysis_execution_result(
    analyses: list[AnalysisArtifact],
    source: ResultArtifact,
    result: AnalysisExecutionResult,
) -> None:
    if result.status != "success":
        return
    analysis = analysis_artifact_from_execution(len(analyses) + 1, source, result)
    analyses.append(analysis)
    console.print(
        f"[green]Recorded analysis #{analysis.analysis_id}[/green] for artifact "
        f"#{analysis.source_artifact_id}."
    )


def record_supervised_run_result(
    analyses: list[AnalysisArtifact],
    source: ResultArtifact,
    result: SupervisedRunResult,
) -> None:
    if result.status not in {"success", "weak_signal"}:
        return
    analysis = analysis_artifact_from_supervised_run(len(analyses) + 1, source, result)
    analyses.append(analysis)
    console.print(
        f"[green]Recorded analysis #{analysis.analysis_id}[/green] for artifact "
        f"#{analysis.source_artifact_id}."
    )


def analysis_artifact_from_execution(
    analysis_id: int,
    source: ResultArtifact,
    result: AnalysisExecutionResult,
) -> AnalysisArtifact:
    return AnalysisArtifact(
        analysis_id=analysis_id,
        source_artifact_id=source.artifact_id,
        recipe=result.recipe,
        status=result.status,
        title=result.title,
        summary=result.summary,
        tables=_analysis_tables_from_result_tables(result.tables),
        warnings=result.warnings,
        created_at=utc_now(),
    )


def analysis_artifact_from_supervised_run(
    analysis_id: int,
    source: ResultArtifact,
    result: SupervisedRunResult,
) -> AnalysisArtifact:
    target = result.target_column or "unknown target"
    metrics = {
        "target_column": target,
        "problem_type": result.problem_type,
        "row_scope": result.row_scope,
        "features": ", ".join(result.feature_columns),
        "baseline": result.baseline_name,
        "model": result.model_name,
        "metric": result.metric_name,
        "baseline_metric": result.baseline_metric,
        "model_metric": result.model_metric,
        "secondary_metric": result.secondary_metric_name,
        "secondary_baseline_metric": result.secondary_baseline_metric,
        "secondary_model_metric": result.secondary_model_metric,
        "rows_used": result.rows_used,
        "rows_dropped_missing_target": result.rows_dropped_missing_target,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
    }
    return AnalysisArtifact(
        analysis_id=analysis_id,
        source_artifact_id=source.artifact_id,
        recipe=result.recipe,
        status=result.status,
        title=f"{result.recipe.title()} Run: {target}",
        summary=result.message,
        tables=_analysis_tables_from_result_tables(result.tables),
        metrics=metrics,
        warnings=result.warnings,
        created_at=utc_now(),
    )


def _analysis_tables_from_result_tables(tables) -> tuple[AnalysisArtifactTable, ...]:
    return tuple(
        AnalysisArtifactTable(
            title=table.title,
            columns=tuple(table.columns),
            rows=tuple(tuple(row) for row in table.rows),
        )
        for table in tables
    )


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
    render_analysis_suggestions(result.columns, result.rows, result.truncated)

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


def render_analysis_suggestions(
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
    truncated: bool,
) -> None:
    capability = analyze_dataset_capability(columns, rows, truncated=truncated)
    suggestions = suggest_analyses(capability)
    if not suggestions:
        return
    lines = [f"- {suggestion.message}" for suggestion in suggestions]
    if any(suggestion.recipe == "visualize" for suggestion in suggestions) and not _viz_available():
        lines.append("- Chart rendering needs the optional viz extra: `uv sync --extra viz`.")
    if capability.truncated:
        lines.append(
            "- Heavier ML should use a bounded re-fetch because the displayed rows are truncated."
        )
    console.print(Panel("\n".join(lines), title="Recommended next steps", border_style="magenta"))


def _viz_available() -> bool:
    return find_spec("matplotlib") is not None


def render_analysis_plan(plan: AnalysisPlan) -> None:
    table = Table(title="Analysis Plan")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Status", plan.status)
    table.add_row("Recipe", plan.recipe)
    table.add_row("Target", plan.target_column or "none")
    table.add_row("Rows", str(plan.row_count))
    table.add_row("Row scope", plan.row_scope)
    table.add_row("Confirmation required", "yes" if plan.confirmation_required else "no")
    table.add_row("Message", plan.message)
    if plan.feature_columns:
        table.add_row("Features", ", ".join(plan.feature_columns))
    if plan.excluded_columns:
        excluded = ", ".join(
            f"{item.column} ({item.reason})" for item in plan.excluded_columns[:8]
        )
        if len(plan.excluded_columns) > 8:
            excluded += f", +{len(plan.excluded_columns) - 8} more"
        table.add_row("Excluded", excluded)
    if plan.warnings:
        table.add_row("Warnings", "\n".join(plan.warnings))
    console.print(table)
    console.print("[dim]Plan only: no ML model was fitted or executed.[/dim]")


def render_analysis_execution(result: AnalysisExecutionResult) -> None:
    console.print(Panel(result.summary, title=result.title, border_style="cyan"))
    for result_table in result.tables:
        table = Table(title=result_table.title)
        for column in result_table.columns:
            table.add_column(column)
        for row in result_table.rows:
            table.add_row(*(format_cell(value) for value in row))
        console.print(table)
    if result.warnings:
        console.print(Panel("\n".join(result.warnings), title="Analysis warnings", border_style="yellow"))
    console.print("[dim]Deterministic analysis only: no ML model was fitted.[/dim]")


def render_supervised_preflight(preflight: SupervisedPreflight) -> None:
    table = Table(title="Supervised ML Preflight")
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Status", preflight.status)
    table.add_row("Recipe", preflight.recipe)
    table.add_row("Target", preflight.target_column or "none")
    table.add_row("Problem type", preflight.problem_type)
    table.add_row("Rows used", str(preflight.row_count))
    table.add_row("Row scope", preflight.row_scope)
    table.add_row("Eligible features", _join_preview(preflight.eligible_features))
    blockers = tuple(gate.reason for gate in preflight.gates if gate.status == "block")
    table.add_row("Blockers", "\n".join(blockers) if blockers else "none")
    table.add_row("Warnings", "\n".join(preflight.warnings) if preflight.warnings else "none")
    table.add_row("Baseline", preflight.baseline_recipe)
    table.add_row(
        "Next action",
        "Ready for a future explicit model run." if preflight.can_execute else "Fix blockers first.",
    )
    console.print(table)

    features = Table(title="Feature Assessment")
    features.add_column("Column")
    features.add_column("Status")
    features.add_column("Reason")
    for assessment in preflight.feature_assessments[:20]:
        features.add_row(assessment.column, assessment.status, assessment.reason)
    if len(preflight.feature_assessments) > 20:
        features.add_row(
            f"+{len(preflight.feature_assessments) - 20} more",
            "hidden",
            "feature list truncated for display",
        )
    console.print(features)
    console.print("[dim]Preflight only: no scikit-learn model was fitted.[/dim]")


def render_supervised_run_result(result: SupervisedRunResult) -> None:
    console.print(Panel(result.message, title="Supervised ML Run", border_style="cyan"))
    summary = Table(title="Run Summary")
    summary.add_column("Field", style="bold cyan")
    summary.add_column("Value")
    summary.add_row("Status", result.status)
    summary.add_row("Recipe", result.recipe)
    summary.add_row("Target", result.target_column or "none")
    summary.add_row("Problem type", result.problem_type)
    summary.add_row("Row scope", result.row_scope)
    summary.add_row("Features", _join_preview(result.feature_columns))
    summary.add_row("Baseline", result.baseline_name)
    summary.add_row("Model", result.model_name)
    summary.add_row("Rows used", str(result.rows_used))
    summary.add_row("Rows dropped missing target", str(result.rows_dropped_missing_target))
    summary.add_row("Train rows", str(result.train_rows))
    summary.add_row("Test rows", str(result.test_rows))
    console.print(summary)

    for result_table in result.tables:
        table = Table(title=result_table.title)
        for column in result_table.columns:
            table.add_column(column)
        for row in result_table.rows:
            table.add_row(*(format_cell(value) for value in row))
        console.print(table)
    if result.warnings:
        console.print(Panel("\n".join(result.warnings), title="ML warnings", border_style="yellow"))
    console.print("[dim]ML run digest only: no model, predictions, or materialized rows are saved.[/dim]")


def parse_analysis_run_request(arg: str) -> tuple[str, bool]:
    tokens = arg.split()
    run_requested = "--run" in tokens
    request = " ".join(token for token in tokens if token != "--run")
    return request.strip(), run_requested


def _join_preview(values: tuple[str, ...], *, limit: int = 8) -> str:
    if not values:
        return "none"
    text = ", ".join(values[:limit])
    if len(values) > limit:
        text += f", +{len(values) - limit} more"
    return text


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
