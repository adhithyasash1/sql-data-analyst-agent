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
    answer_question,
    create_openai_client,
    download_northwind,
    index_schema,
    read_query_logs,
    verify_sqlite_database,
)

app = typer.Typer(add_completion=False, help="Local Text-to-SQL for any SQLite database.")
console = Console()


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
            f"Database: {settings.source_db_path.name}\nType 'exit' to quit.",
            title="Text-to-SQL Assistant",
        )
    )
    while True:
        user_input = Prompt.ask("[bold]You[/bold]").strip()
        if user_input.lower() in {"exit", "quit", ":q"}:
            break
        if not user_input:
            continue
        run_question(settings, client, user_input, verbose)


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
