"""Claude API integration — runs the tool-use loop and returns structured analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

from src.profiler import DataProfile, DataProfiler

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a World-Class Senior Data Analyst, BI Architect, and Data Storyteller.
You do not summarize CSV files. You analyze datasets as if you were hired to design \
the correct executive or operational dashboard from them.
You extract structure, meaning, KPIs, and insights directly from the data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY ANALYSIS FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — UNDERSTAND DATASET STRUCTURE FIRST
Before producing any insight, determine:
  • What one row represents (the entity level)
  • Which columns are dimensions vs metrics
  • Which columns represent time (trends are possible)
  • Which columns represent states, statuses, or lifecycle stages
  • Which columns are identifiers (exclude from aggregation)
  • What can be grouped, aggregated, or trended
Do NOT assume the domain — infer it from the column names and distributions.

STEP 2 — KPI DISCOVERY (NO TEMPLATES)
KPIs must be derived only from this dataset's structure using this reasoning:
  • What is being measured repeatedly across rows?
  • What changes over time?
  • What can be segmented into meaningful groups?
  • What represents volume, performance, movement, or distribution?
  • What questions would naturally arise when these columns are combined?
You are FORBIDDEN from inventing generic KPIs that do not emerge from the data.
Two datasets from the same domain MUST produce different KPIs if their structure differs.

STEP 3 — STRATEGIC TOOL USAGE
Use tools only to validate analytical thinking:
  • get_column_stats → confirm distributions and identify skew/outliers before claiming patterns
  • get_correlations → confirm relationships before asserting predictive relationships
  • get_outliers → detect anomalies before recommending exclusions
  • get_missing_values → assess quality before recommending imputation strategies
Never call tools randomly. Every tool call must answer a specific analytical question.

STEP 4 — INSIGHT DEPTH
For every dataset you must provide:
  • Major patterns with actual numbers
  • Trends (if time data exists — direction, magnitude, seasonality)
  • Segment comparisons (which groups outperform or underperform and by how much)
  • Anomalies (statistical outliers AND logical inconsistencies)
  • Data quality issues (missing %, duplicate risk, type mismatches, imputation feasibility)
  • Strategic implications (what decisions this data enables or blocks)
No generic observations. Every claim must cite a number or column.

STEP 5 — DASHBOARD THINKING
Structure your analysis as if designing a real BI dashboard:
  1. KPI Overview — headline metrics with context
  2. Trends & Changes — time-series patterns if applicable
  3. Segmentation & Breakdown — performance by dimension
  4. Risk Areas — data quality, anomalies, concentration risk
  5. Opportunity Areas — underexplored segments, predictive signals
  6. Recommendations — specific, owner-assigned actions

STEP 6 — VISUAL AWARENESS
While you do not generate visuals, recommend specific chart types for each finding:
  • Time series → Line chart or area chart
  • Distribution → Histogram or box plot
  • Segment comparison → Horizontal bar chart ranked by metric
  • Correlation → Scatter plot with regression line
  • Composition → Stacked bar or treemap (not pie chart for >4 categories)
  • KPI status → Gauge or scorecard with trend arrow
  • Missing values → Heatmap by column

STEP 7 — INTERNAL QUALITY CHECK
Before writing your final response, ask: "Would a Head of Data be impressed by this?"
If not, deepen the reasoning. Shallow observations are not acceptable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (EXACT HEADERS REQUIRED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Format your FINAL response using EXACTLY these section headers (including the ## markers):

## EXECUTIVE_SUMMARY ##
<3-4 paragraph narrative. Lead with what this dataset IS (entity, domain, scope). \
Then cover the most important pattern. Then data quality posture. \
End with what decisions this dataset enables. Write for a VP of Data or CDO.>

## KEY_FINDINGS ##
1. <quantified finding with segment or trend context>
2. <quantified finding with segment or trend context>
3. <quantified finding with segment or trend context>
4. <quantified finding with segment or trend context>
5. <quantified finding with segment or trend context>

## COLUMN_ANALYSES ##
### <column_name>
Summary: <role in the dataset — is this a dimension, metric, identifier, or status field?>
Quality: <missing %, outlier count, imputation feasibility, type issues>
Patterns: <distribution shape, top values, correlations, business meaning of the pattern>

(repeat for each column)

## ANOMALIES ##
- <specific anomaly with column, value range, row count, and business risk>
- <specific anomaly with column, value range, row count, and business risk>

## RECOMMENDATIONS ##
1. <actionable: who should do what, using which column, and what outcome to expect>
2. <actionable: who should do what, using which column, and what outcome to expect>
3. <actionable: who should do what, using which column, and what outcome to expect>

## METHODOLOGY ##
<Describe the analytical sequence you followed: what structure you identified first, \
which tools you used and why, what hypotheses you formed and tested, \
and what limitations exist in this profile-based analysis.>
"""

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "get_column_stats",
        "description": (
            "Returns detailed statistics for a specific column including data type, "
            "missing values, distribution metrics (mean, median, std, quartiles), "
            "outlier count, top values (for categorical), and date range (for dates)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "The exact name of the column to retrieve statistics for.",
                }
            },
            "required": ["column_name"],
        },
    },
    {
        "name": "get_correlations",
        "description": (
            "Returns correlation pairs between numeric columns where |r| > 0.5, "
            "sorted by absolute correlation strength (strongest first)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_outliers",
        "description": (
            "Returns a summary of outliers detected in all numeric columns using the "
            "IQR method (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_missing_values",
        "description": (
            "Returns missing value counts and percentages for all columns that have "
            "at least one missing value, sorted by missingness descending."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

_MAX_TOOL_ITERATIONS = 10


@dataclass
class ColumnAnalysis:
    """Analysis of a single column produced by Claude."""

    column_name: str
    summary: str
    quality: str
    patterns: str


@dataclass
class AnalysisResult:
    """Structured analysis result returned by DataAnalyzer."""

    executive_summary: str
    key_findings: list[str] = field(default_factory=list)
    column_analyses: list[ColumnAnalysis] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    methodology_notes: str = ""
    raw_response: str = ""  # preserved for debugging


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------


class DataAnalyzer:
    """Integrates with the Claude API to analyze a profiled dataset.

    Uses the tool-use pattern: Claude calls profiler methods as tools to
    gather column-level statistics before producing a structured report.
    """

    def __init__(
        self,
        profiler: DataProfiler,
        model: str,
        api_key: str | None = None,
    ) -> None:
        """Initialize with a profiler instance, model ID, and optional API key.

        If api_key is None, the SDK reads ANTHROPIC_API_KEY from the environment.
        """
        self._profiler = profiler
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def analyze(self, profile: DataProfile) -> AnalysisResult:
        """Run the full Claude tool-use loop and return a structured AnalysisResult.

        1. Builds an initial user message with high-level dataset metadata.
        2. Loops: when Claude calls tools, executes them and returns results.
        3. When Claude signals end_turn, parses its final response.
        """
        initial_message = self._build_initial_message(profile)
        messages: list[dict] = [{"role": "user", "content": initial_message}]
        raw_text = self._run_tool_loop(messages)
        return self._parse_analysis(raw_text)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_initial_message(self, profile: DataProfile) -> str:
        """Build the user message sent as the first turn of the conversation.

        Provides high-level metadata so Claude can orient itself before using
        tools to drill into specific columns and statistics.
        """
        col_descriptions = []
        for col in profile.columns:
            cp = profile.column_profiles[col]
            desc = f"  - {col} ({cp.dtype}): {cp.missing_pct}% missing, {cp.unique_count} unique values"
            if cp.mean is not None:
                desc += f", mean={cp.mean}"
            col_descriptions.append(desc)

        lines = [
            "Please analyze the following dataset:",
            "",
            f"Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns",
            f"Memory: {profile.memory_usage_mb} MB",
            f"Duplicate rows: {profile.duplicate_rows}",
            f"Overall missing values: {profile.total_missing_pct}%",
            "",
            f"Numeric columns ({len(profile.numeric_columns)}): "
            + ", ".join(profile.numeric_columns),
            f"Categorical columns ({len(profile.categorical_columns)}): "
            + ", ".join(profile.categorical_columns),
            f"Date columns ({len(profile.date_columns)}): "
            + (", ".join(profile.date_columns) if profile.date_columns else "none"),
            "",
            "Column overview:",
            *col_descriptions,
            "",
            "Use the available tools to investigate the data thoroughly before "
            "writing your final structured analysis.",
        ]
        return "\n".join(lines)

    def _run_tool_loop(self, messages: list[dict]) -> str:
        """Core agentic loop: call Claude, execute tools, repeat until end_turn.

        Returns Claude's final text response.
        """
        for _ in range(_MAX_TOOL_ITERATIONS):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=8096,
                system=_SYSTEM_PROMPT,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )

            if response.stop_reason == "end_turn":
                # Extract the final text content
                text_blocks = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_blocks)

            # stop_reason == "tool_use" — execute each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str = self._execute_tool(block.name, block.input)  # type: ignore[arg-type]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

            # Append assistant turn (with tool_use blocks) and tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # Fell out of the loop — return whatever the last text was
        last_texts = [
            block.text
            for msg in reversed(messages)
            if msg["role"] == "assistant"
            for block in (msg["content"] if isinstance(msg["content"], list) else [])
            if hasattr(block, "text")
        ]
        return last_texts[0] if last_texts else "Analysis incomplete — max iterations reached."

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Dispatch a Claude tool call to the appropriate DataProfiler method.

        Returns a JSON-serialised string of the result.
        """
        dispatch = {
            "get_column_stats": lambda: self._profiler.get_column_stats(
                tool_input.get("column_name", "")
            ),
            "get_correlations": lambda: self._profiler.get_correlations(),
            "get_outliers": lambda: self._profiler.get_outliers(),
            "get_missing_values": lambda: self._profiler.get_missing_values(),
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = handler()
        except Exception as exc:
            result = {"error": str(exc)}
        return json.dumps(result, default=str)

    def _parse_analysis(self, raw_text: str) -> AnalysisResult:
        """Parse Claude's delimiter-based response into an AnalysisResult dataclass.

        Falls back gracefully: if a section is missing, uses an empty default so
        the report can still be assembled.
        """
        result = AnalysisResult(
            executive_summary="",
            raw_response=raw_text,
        )

        def _extract(tag: str) -> str:
            """Extract content between ## TAG ## and the next ## ... ## marker."""
            pattern = rf"## {re.escape(tag)} ##\s*(.*?)(?=## \w[\w_ ]* ##|$)"
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""

        result.executive_summary = _extract("EXECUTIVE_SUMMARY") or raw_text[:500]
        result.methodology_notes = _extract("METHODOLOGY")

        # Key findings — numbered list
        kf_block = _extract("KEY_FINDINGS")
        if kf_block:
            result.key_findings = _parse_numbered_list(kf_block)

        # Anomalies — bullet list
        anomalies_block = _extract("ANOMALIES")
        if anomalies_block:
            result.anomalies = _parse_bullet_list(anomalies_block)

        # Recommendations — numbered list
        recs_block = _extract("RECOMMENDATIONS")
        if recs_block:
            result.recommendations = _parse_numbered_list(recs_block)

        # Column analyses — ### ColumnName subsections
        col_block = _extract("COLUMN_ANALYSES")
        if col_block:
            result.column_analyses = _parse_column_analyses(col_block)

        return result


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_numbered_list(text: str) -> list[str]:
    """Parse a numbered markdown list into a list of strings."""
    items = re.findall(r"^\d+\.\s+(.+)$", text, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def _parse_bullet_list(text: str) -> list[str]:
    """Parse a bullet markdown list into a list of strings."""
    items = re.findall(r"^[-*]\s+(.+)$", text, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def _parse_column_analyses(text: str) -> list[ColumnAnalysis]:
    """Parse the COLUMN_ANALYSES block into a list of ColumnAnalysis objects."""
    analyses: list[ColumnAnalysis] = []
    # Split on ### headers
    sections = re.split(r"^### (.+)$", text, flags=re.MULTILINE)
    # sections = ["", col1, col1_content, col2, col2_content, ...]
    it = iter(sections[1:])
    for col_name in it:
        try:
            content = next(it)
        except StopIteration:
            content = ""
        col_name = col_name.strip()
        summary = _extract_field(content, "Summary")
        quality = _extract_field(content, "Quality")
        patterns = _extract_field(content, "Patterns")
        analyses.append(
            ColumnAnalysis(
                column_name=col_name,
                summary=summary,
                quality=quality,
                patterns=patterns,
            )
        )
    return analyses


def _extract_field(text: str, field: str) -> str:
    """Extract a labelled field (e.g. 'Summary: ...') from a text block."""
    pattern = rf"^{re.escape(field)}:\s*(.+?)(?=\n[A-Z][a-z]+:|$)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""
