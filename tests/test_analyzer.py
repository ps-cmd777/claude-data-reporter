"""Tests for src/analyzer.py — Claude API is always mocked; no real API calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.analyzer import (
    AnalysisResult,
    DataAnalyzer,
    _parse_bullet_list,
    _parse_column_analyses,
    _parse_numbered_list,
)
from src.profiler import ColumnProfile, DataProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-5-20250514"


@pytest.fixture()
def mock_profile() -> DataProfile:
    """Minimal DataProfile used for analyzer tests."""
    cp_price = ColumnProfile(
        name="price",
        dtype="float64",
        missing_count=2,
        missing_pct=1.0,
        unique_count=98,
        mean=50.0,
        median=49.5,
        std=10.0,
        min_val=10.0,
        max_val=100.0,
        q25=43.0,
        q75=57.0,
        outlier_count=3,
    )
    cp_category = ColumnProfile(
        name="category",
        dtype="object",
        missing_count=0,
        missing_pct=0.0,
        unique_count=3,
        top_values={"A": 40, "B": 35, "C": 25},
    )
    return DataProfile(
        shape=(200, 2),
        columns=["price", "category"],
        column_profiles={"price": cp_price, "category": cp_category},
        numeric_columns=["price"],
        categorical_columns=["category"],
        date_columns=[],
        total_missing=2,
        total_missing_pct=0.5,
        correlation_matrix={"price": {"price": 1.0}},
        duplicate_rows=0,
        memory_usage_mb=0.02,
    )


def _make_text_block(text: str):
    """Create a mock text block mimicking anthropic SDK TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, input_dict: dict):
    """Create a mock tool_use block mimicking anthropic SDK ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_dict
    return block


def _sample_analysis_text() -> str:
    """Return a well-formed Claude response for parsing tests."""
    return """\
## EXECUTIVE_SUMMARY ##
This is a small dataset with 200 rows and 2 columns. The price column shows \
a normal distribution around $50. Category distribution is reasonably balanced.

## KEY_FINDINGS ##
1. Price mean is $50.0 with std of $10.0, indicating low variance.
2. Category column has 3 unique values with A being most common (40%).
3. Missing values are minimal at 0.5% overall.
4. 3 outliers detected in the price column via IQR method.
5. No date columns present — time-series analysis not applicable.

## COLUMN_ANALYSES ##
### price
Summary: Numeric column with a near-normal distribution centred around $50.
Quality: 2 missing values (1%). 3 outliers detected.
Patterns: Low coefficient of variation suggests stable pricing.

### category
Summary: Categorical column with 3 values — A, B, C.
Quality: No missing values.
Patterns: Category A is the most frequent at 40% of records.

## ANOMALIES ##
- 3 outliers in the price column exceed Q3 + 1.5*IQR.

## RECOMMENDATIONS ##
1. Impute or investigate the 2 missing price values before modeling.
2. Investigate the 3 price outliers — they may be data entry errors.
3. Consider encoding category as an ordered feature if A > B > C in business value.

## METHODOLOGY ##
Profiling was performed with pandas. Claude used get_column_stats and get_outliers \
tools to investigate the data before writing this report.
"""


# ---------------------------------------------------------------------------
# Mock anthropic client helper
# ---------------------------------------------------------------------------


def _make_mock_client(responses: list) -> MagicMock:
    """Create a mock anthropic.Anthropic client with pre-defined message responses."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses
    return mock_client


# ---------------------------------------------------------------------------
# DataAnalyzer — tool loop
# ---------------------------------------------------------------------------


class TestDataAnalyzerToolLoop:
    def test_single_end_turn_response(self, mock_profile):
        """When Claude returns end_turn on first call, no tools are executed."""
        analysis_text = _sample_analysis_text()
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [_make_text_block(analysis_text)]

        mock_profiler = MagicMock()
        mock_profiler.profile.return_value = mock_profile

        with patch("anthropic.Anthropic", return_value=_make_mock_client([response])):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
            result = analyzer.analyze(mock_profile)

        assert isinstance(result, AnalysisResult)
        assert "dataset" in result.executive_summary.lower()

    def test_tool_use_then_end_turn(self, mock_profile):
        """Claude calls one tool, then returns end_turn on the second iteration."""
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            _make_tool_use_block("tu_1", "get_missing_values", {}),
        ]

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [_make_text_block(_sample_analysis_text())]

        mock_profiler = MagicMock()
        mock_profiler.get_missing_values.return_value = {
            "columns_with_missing": [],
            "total_missing": 2,
            "total_missing_pct": 0.5,
        }

        mock_client = _make_mock_client([tool_response, final_response])
        with patch("anthropic.Anthropic", return_value=mock_client):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
            result = analyzer.analyze(mock_profile)

        assert mock_client.messages.create.call_count == 2
        mock_profiler.get_missing_values.assert_called_once()
        assert isinstance(result, AnalysisResult)

    def test_multiple_tools_in_one_response(self, mock_profile):
        """Claude returns two tool calls in one response; both are dispatched."""
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [
            _make_tool_use_block("tu_1", "get_correlations", {}),
            _make_tool_use_block("tu_2", "get_outliers", {}),
        ]

        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        final_response.content = [_make_text_block(_sample_analysis_text())]

        mock_profiler = MagicMock()
        mock_profiler.get_correlations.return_value = {"significant_pairs": [], "total_numeric_columns": 1}
        mock_profiler.get_outliers.return_value = {"columns_with_outliers": [], "total_columns_affected": 0}

        mock_client = _make_mock_client([tool_response, final_response])
        with patch("anthropic.Anthropic", return_value=mock_client):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
            analyzer.analyze(mock_profile)

        mock_profiler.get_correlations.assert_called_once()
        mock_profiler.get_outliers.assert_called_once()

    def test_api_key_not_hardcoded(self):
        """Verify no literal API key string appears in analyzer source."""
        import inspect
        import src.analyzer as analyzer_module

        source = inspect.getsource(analyzer_module)
        assert "sk-ant-" not in source


# ---------------------------------------------------------------------------
# DataAnalyzer — tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    def _make_analyzer(self, mock_profiler):
        with patch("anthropic.Anthropic"):
            return DataAnalyzer(profiler=mock_profiler, model=_MODEL)

    def test_dispatch_get_column_stats(self):
        mock_profiler = MagicMock()
        mock_profiler.get_column_stats.return_value = {"name": "price"}
        with patch("anthropic.Anthropic"):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
        result = analyzer._execute_tool("get_column_stats", {"column_name": "price"})
        assert '"name"' in result
        mock_profiler.get_column_stats.assert_called_once_with("price")

    def test_dispatch_get_correlations(self):
        mock_profiler = MagicMock()
        mock_profiler.get_correlations.return_value = {"significant_pairs": []}
        with patch("anthropic.Anthropic"):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
        result = analyzer._execute_tool("get_correlations", {})
        assert "significant_pairs" in result

    def test_dispatch_unknown_tool(self):
        mock_profiler = MagicMock()
        with patch("anthropic.Anthropic"):
            analyzer = DataAnalyzer(profiler=mock_profiler, model=_MODEL)
        result = analyzer._execute_tool("nonexistent_tool", {})
        assert "error" in result.lower()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


class TestParsingHelpers:
    def test_parse_numbered_list(self):
        text = "1. First item\n2. Second item\n3. Third"
        result = _parse_numbered_list(text)
        assert result == ["First item", "Second item", "Third"]

    def test_parse_bullet_list(self):
        text = "- Item one\n- Item two\n* Item three"
        result = _parse_bullet_list(text)
        assert "Item one" in result
        assert "Item two" in result

    def test_parse_empty_list(self):
        assert _parse_numbered_list("") == []
        assert _parse_bullet_list("") == []

    def test_parse_column_analyses(self):
        text = (
            "### price\n"
            "Summary: Normal distribution.\n"
            "Quality: 2 missing values.\n"
            "Patterns: Low variance.\n\n"
            "### category\n"
            "Summary: 3 unique values.\n"
            "Quality: Complete.\n"
            "Patterns: A is most common.\n"
        )
        result = _parse_column_analyses(text)
        assert len(result) == 2
        assert result[0].column_name == "price"
        assert "Normal" in result[0].summary
        assert result[1].column_name == "category"


class TestParseAnalysis:
    def test_full_parse(self, mock_profile):
        with patch("anthropic.Anthropic"):
            analyzer = DataAnalyzer(profiler=MagicMock(), model=_MODEL)
        result = analyzer._parse_analysis(_sample_analysis_text())

        assert "dataset" in result.executive_summary.lower()
        assert len(result.key_findings) == 5
        assert len(result.column_analyses) == 2
        assert len(result.anomalies) == 1
        assert len(result.recommendations) == 3
        assert result.methodology_notes != ""

    def test_fallback_on_empty_response(self, mock_profile):
        with patch("anthropic.Anthropic"):
            analyzer = DataAnalyzer(profiler=MagicMock(), model=_MODEL)
        result = analyzer._parse_analysis("")
        # Should not raise; executive_summary falls back to raw[:500] which is ""
        assert isinstance(result, AnalysisResult)
