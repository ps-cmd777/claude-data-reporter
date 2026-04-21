# claude-data-reporter — AI Context File

## Project Overview

A Python CLI tool that profiles any CSV file with pandas, sends the analysis to
the Claude API using the **tool-use pattern**, and generates a professional
Markdown report with matplotlib/seaborn charts.

Built by Shushan as a portfolio project for transitioning into AI/prompt engineering roles.

## Tech Stack

| Layer | Library |
|-------|---------|
| CLI | `click` |
| Data profiling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| AI analysis | `anthropic` Python SDK |
| Env management | `python-dotenv` |
| Linting | `ruff` |
| Testing | `pytest` |

## Directory Structure

```
src/
  cli.py          — Click entry point; load_dotenv() here
  profiler.py     — DataProfiler class + DataProfile/ColumnProfile dataclasses
  visualizer.py   — Visualizer class; matplotlib.use("Agg") at top of file
  analyzer.py     — DataAnalyzer class; Claude tool-use loop lives here
  reporter.py     — ReportGenerator class; assembles Markdown sections
  utils.py        — Pure helpers: validate_csv_path, ensure_output_dir, etc.
tests/            — pytest suite; Claude API is always mocked
examples/         — sample_sales.csv + generate_sample_data.py
docs/             — architecture.md, learning-notes.md
output/           — generated reports and charts go here (gitignored)
```

## Common Commands

```bash
# Install (editable + dev deps)
pip install -e ".[dev]"

# Copy and fill in your API key
cp .env.example .env

# Run analysis on a CSV
claude-data-reporter analyze examples/sample_sales.csv

# Run analysis with all options
claude-data-reporter analyze data.csv --output-dir reports/ --verbose

# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Generate sample data
python examples/generate_sample_data.py
```

## Architecture Pattern

```
CSV → DataProfiler.profile() → DataProfile
                                    ├── Visualizer.generate_all() → List[Path] (PNGs)
                                    └── DataAnalyzer.analyze()
                                            ↕ tool-use loop
                                          Claude API
                                            ↓
                                        AnalysisResult
                                            ↓
                                      ReportGenerator.generate_report()
                                            ↓
                                      output/report_*.md
```

## Coding Standards

- **Type hints**: All function signatures must have type hints (params + return)
- **Docstrings**: Every public method/function needs a one-line docstring minimum
- **Ruff**: Format with `ruff format`, lint with `ruff check` before committing
- **No hardcoded secrets**: `ANTHROPIC_API_KEY` is read from env only — never in code
- **Matplotlib backend**: `matplotlib.use("Agg")` must be the first matplotlib call
  in visualizer.py to support headless/CI environments

## API Key Handling

```python
# CORRECT — reads from ANTHROPIC_API_KEY env var automatically
client = anthropic.Anthropic()

# ALSO CORRECT — explicit None is the same as above
client = anthropic.Anthropic(api_key=None)

# NEVER DO THIS
client = anthropic.Anthropic(api_key="sk-ant-...")
```

## Tool-Use Loop (analyzer.py)

Claude is given 4 tools that dispatch to DataProfiler methods:
- `get_column_stats(column_name)` → column statistics dict
- `get_correlations()` → top correlated pairs
- `get_outliers()` → outlier summary across numeric columns
- `get_missing_values()` → missing value counts/percentages

The loop continues until `response.stop_reason == "end_turn"` (max 10 iterations).

## Testing Approach

- `test_profiler.py` — real pandas operations on small fixture DataFrames
- `test_analyzer.py` — `unittest.mock` patches `anthropic.Anthropic`; no real API calls
- `test_visualizer.py` — real matplotlib with `tmp_path` fixture; checks PNGs exist
- `test_reporter.py` — mock profile + analysis data; asserts sections present in output

## Built With Claude Code

This project was scaffolded and developed using Claude Code CLI.
See the plan at `.claude/plans/` for the architectural decisions.
