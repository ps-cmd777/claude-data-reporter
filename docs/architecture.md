# Architecture

## Data Flow

```
CSV File (any size/schema)
      │
      ▼
┌─────────────┐
│   cli.py    │  click command parses args, validates input, orchestrates
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  profiler.py    │  DataProfiler
│  DataProfiler   │  ─ load_csv()          → pd.DataFrame
│                 │  ─ profile()           → DataProfile (dataclass)
│                 │  ─ detect_outliers()   → IQR-based outlier flags
│                 │  ─ compute_correlations() → correlation matrix
│                 │  ─ get_distributions() → bin counts for charts
│                 │
│  Tool handlers  │  ─ get_column_stats()  ← Claude tool call
│  (used by       │  ─ get_correlations()  ← Claude tool call
│   analyzer)     │  ─ get_outliers()      ← Claude tool call
│                 │  ─ get_missing_values() ← Claude tool call
└──────┬──────────┘
       │ DataProfile
       ├──────────────────────────────────────┐
       ▼                                      ▼
┌──────────────┐                   ┌──────────────────┐
│ visualizer.py│                   │  analyzer.py     │
│  Visualizer  │                   │  DataAnalyzer    │
│              │                   │                  │
│  5 chart     │                   │  ┌─────────────┐ │
│  types →     │                   │  │ Claude API  │ │
│  PNG files   │                   │  │ tool-use    │ │
│              │                   │  │ loop        │ │
│  Returns     │                   │  └──────┬──────┘ │
│  List[Path]  │                   │         │        │
└──────┬───────┘                   │  AnalysisResult  │
       │                           └──────────────────┘
       │ chart_paths                        │
       └──────────────┬─────────────────────┘
                      ▼
             ┌─────────────────┐
             │  reporter.py    │
             │ ReportGenerator │
             │                 │
             │  8 sections →   │
             │  Markdown file  │
             └────────┬────────┘
                      │
                      ▼
             output/report_*.md
             output/charts/*.png
```

## Module Responsibilities

### `src/cli.py`
Entry point. Uses `click` to parse CLI arguments. Calls `load_dotenv()` before anything else. Resolves model via 3-tier priority: flag → env var → default. Orchestrates the pipeline and handles top-level errors.

### `src/profiler.py`
The data foundation. `DataProfiler` loads the CSV and builds a `DataProfile` dataclass containing full column-level statistics. The same class exposes 4 tool handler methods that Claude calls during the analysis loop.

### `src/visualizer.py`
Chart generation. `matplotlib.use("Agg")` is set at the top of the file — this is critical for headless/CI environments. All 5 chart methods return `None` or `[]` when the chart is not applicable (no missing values, single numeric column, etc.) rather than raising exceptions.

### `src/analyzer.py`
Claude integration. `DataAnalyzer` runs the tool-use loop: sends the dataset overview, processes tool calls from Claude, and parses Claude's final structured response into an `AnalysisResult` dataclass. Never hardcodes API keys.

### `src/reporter.py`
Report assembly. `ReportGenerator` renders each of the 8 sections as Markdown strings and concatenates them into a final file. Chart paths are stored as relative paths so the report folder is portable.

### `src/utils.py`
Pure helper functions with no side effects and no cross-module imports. Used by cli.py for path validation and output directory setup.

## Tool-Use Loop Detail

```
messages = [{"role": "user", "content": "<dataset overview>"}]

while iteration < MAX_ITERATIONS:
    response = client.messages.create(
        model=model,
        tools=TOOLS,
        messages=messages
    )

    if response.stop_reason == "end_turn":
        # Extract text → parse into AnalysisResult
        break

    # stop_reason == "tool_use"
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            result = profiler.dispatch_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str)
            })

    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sync (not async) | CLI tool — user waits; sync is simpler and easier to test |
| DataProfiler as tool dispatcher | Avoids duplication; profiler already holds all data |
| None-returning charts | Pipeline never fails because of a chart edge case |
| Delimiter-based output parsing | More robust than JSON mode for end-of-conversation responses |
| `load_dotenv()` at module level | Guarantees env loads before any `os.getenv()` call |
| Sample data via script | Reproducible with `numpy.random.seed(42)` |
