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
You are a senior data analyst with 10+ years of experience in exploratory data \
analysis, statistical modeling, and business intelligence.

Your task is to analyze a dataset profile and produce a structured, actionable report.

You have access to tools to query specific aspects of the dataset. Use them to gather \
the details you need. Investigate columns that look interesting, check correlations, \
review outliers, and assess data quality before writing your final analysis.

Guidelines:
- Lead with business impact, not technical detail
- Quantify findings wherever possible (use actual numbers from the data)
- Flag data quality issues that would affect downstream analysis or modeling
- Recommendations must be specific and actionable, not generic
- Write for a technical audience (data engineers, analysts, data scientists)
- Use precise statistical language (e.g., "right-skewed distribution" not "some high values")

Format your FINAL response using EXACTLY these section headers (including the ## markers):

## EXECUTIVE_SUMMARY ##
<2-3 paragraph narrative summary of the dataset>

## KEY_FINDINGS ##
1. <specific, quantified finding>
2. <specific, quantified finding>
3. <specific, quantified finding>
4. <specific, quantified finding>
5. <specific, quantified finding>

## COLUMN_ANALYSES ##
### <column_name>
Summary: <1-2 sentences describing the column's role and distribution>
Quality: <any data quality issues — missing values, outliers, inconsistencies>
Patterns: <notable patterns, correlations, or business-relevant observations>

(repeat for each column)

## ANOMALIES ##
- <specific anomaly with numbers>
- <specific anomaly with numbers>

## RECOMMENDATIONS ##
1. <actionable recommendation — who should do what and why>
2. <actionable recommendation>
3. <actionable recommendation>

## METHODOLOGY ##
<1-2 paragraphs describing the analytical approach and tools used>
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
