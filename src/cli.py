"""CLI entry point for claude-data-reporter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env before any os.getenv call anywhere in the pipeline
load_dotenv()

_DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
_DEFAULT_OUTPUT_DIR = "output"


@click.group()
@click.version_option(version="0.1.0", prog_name="claude-data-reporter")
def main() -> None:
    """Claude Data Reporter — AI-powered CSV analysis and report generation."""


@main.command()
@click.argument("csv_path", type=click.Path(dir_okay=False, path_type=str))
@click.option(
    "--output-dir",
    "-o",
    default=_DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory where the report and charts will be saved.",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Claude model ID (overrides DEFAULT_MODEL env var).",
)
@click.option(
    "--no-charts",
    is_flag=True,
    default=False,
    help="Skip chart generation (report only).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed progress output.",
)
def analyze(
    csv_path: str,
    output_dir: str,
    model: str | None,
    no_charts: bool,
    verbose: bool,
) -> None:
    """Analyze a CSV file and generate an AI-powered Markdown report.

    CSV_PATH is the path to the CSV file you want to analyze.

    \b
    Examples:
      claude-data-reporter analyze sales.csv
      claude-data-reporter analyze data.csv --output-dir reports/ --verbose
      claude-data-reporter analyze data.csv --no-charts --model claude-opus-4-6
    """
    # Lazy imports so CLI --help is instant even without dependencies installed
    from src.analyzer import DataAnalyzer
    from src.profiler import DataProfiler
    from src.reporter import ReportGenerator
    from src.utils import ensure_output_dir, validate_csv_path
    from src.visualizer import Visualizer

    # ------------------------------------------------------------------ #
    # 1. Validate inputs
    # ------------------------------------------------------------------ #
    try:
        csv_file = validate_csv_path(csv_path)
    except click.BadParameter as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        click.echo(
            click.style(
                "Error: ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or set the environment variable.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    # 3-tier model resolution: CLI flag > env var > hardcoded default
    resolved_model = model or os.getenv("DEFAULT_MODEL") or _DEFAULT_MODEL
    out_dir = ensure_output_dir(output_dir)

    if verbose:
        click.echo(f"Output directory : {out_dir}")
        click.echo(f"Model            : {resolved_model}")
        click.echo(f"Charts           : {'disabled' if no_charts else 'enabled'}")
        click.echo("")

    # ------------------------------------------------------------------ #
    # 2. Load and profile the CSV
    # ------------------------------------------------------------------ #
    _step("Loading CSV...", verbose)
    try:
        profiler = DataProfiler()
        df = profiler.load_csv(csv_file)
    except Exception as exc:
        _error(f"Failed to load CSV: {exc}")

    _step(
        f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.",
        verbose,
    )
    _step("Profiling data...", verbose)
    try:
        profile = profiler.profile()
    except Exception as exc:
        _error(f"Failed to profile data: {exc}")

    if verbose:
        click.echo(
            f"  Numeric: {len(profile.numeric_columns)} | "
            f"Categorical: {len(profile.categorical_columns)} | "
            f"Date: {len(profile.date_columns)} | "
            f"Missing: {profile.total_missing_pct}%"
        )

    # ------------------------------------------------------------------ #
    # 3. Generate charts
    # ------------------------------------------------------------------ #
    chart_paths: list[Path] = []
    if not no_charts:
        _step("Generating charts...", verbose)
        try:
            visualizer = Visualizer(out_dir)
            chart_paths = visualizer.generate_all(df, profile)
            _step(f"  {len(chart_paths)} chart(s) saved to {out_dir}/", verbose)
        except Exception as exc:
            click.echo(
                click.style(f"Warning: Chart generation failed: {exc}", fg="yellow"),
                err=True,
            )

    # ------------------------------------------------------------------ #
    # 4. Analyze with Claude
    # ------------------------------------------------------------------ #
    _step("Analyzing with Claude (this may take 30–60 seconds)...", verbose)
    try:
        analyzer = DataAnalyzer(profiler=profiler, model=resolved_model)
        analysis = analyzer.analyze(profile)
    except Exception as exc:
        _error(f"Claude API call failed: {exc}")

    # ------------------------------------------------------------------ #
    # 5. Write report
    # ------------------------------------------------------------------ #
    _step("Writing report...", verbose)
    try:
        reporter = ReportGenerator(out_dir)
        report_path = reporter.generate_report(
            csv_filename=csv_file.name,
            profile=profile,
            analysis=analysis,
            chart_paths=chart_paths,
        )
    except Exception as exc:
        _error(f"Failed to write report: {exc}")

    click.echo("")
    click.echo(click.style("Analysis complete!", fg="green", bold=True))
    click.echo(f"Report saved to: {report_path}")
    if chart_paths:
        click.echo(f"Charts saved to: {out_dir}/")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _step(message: str, verbose: bool) -> None:
    """Print a progress step if verbose mode is on."""
    if verbose:
        click.echo(message)
    else:
        # Always show major steps even in non-verbose mode
        if not message.startswith("  "):
            click.echo(message)


def _error(message: str) -> None:
    """Print a red error message and exit with code 1."""
    click.echo(click.style(f"Error: {message}", fg="red"), err=True)
    sys.exit(1)


if __name__ == "__main__":
    main()


