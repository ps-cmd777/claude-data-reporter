"""Tests for src/reporter.py — verifies Markdown report structure and content."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analyzer import AnalysisResult, ColumnAnalysis
from src.profiler import ColumnProfile, DataProfile
from src.reporter import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_dir(tmp_path) -> Path:
    return tmp_path / "reports"


@pytest.fixture()
def mock_profile() -> DataProfile:
    """Minimal DataProfile for report tests."""
    cp_price = ColumnProfile(
        name="price",
        dtype="float64",
        missing_count=5,
        missing_pct=2.5,
        unique_count=195,
        mean=50.0,
        median=49.0,
        std=10.0,
        min_val=10.0,
        max_val=150.0,
        q25=42.0,
        q75=58.0,
        outlier_count=3,
    )
    cp_category = ColumnProfile(
        name="category",
        dtype="object",
        missing_count=0,
        missing_pct=0.0,
        unique_count=3,
        top_values={"A": 80, "B": 70, "C": 50},
    )
    return DataProfile(
        shape=(200, 2),
        columns=["price", "category"],
        column_profiles={"price": cp_price, "category": cp_category},
        numeric_columns=["price"],
        categorical_columns=["category"],
        date_columns=[],
        total_missing=5,
        total_missing_pct=1.25,
        correlation_matrix={"price": {"price": 1.0}},
        duplicate_rows=2,
        memory_usage_mb=0.05,
    )


@pytest.fixture()
def mock_analysis() -> AnalysisResult:
    """Minimal AnalysisResult for report tests."""
    return AnalysisResult(
        executive_summary="This is a test dataset with 200 rows and 2 columns.",
        key_findings=[
            "Price mean is $50 with std of $10.",
            "Category A is the most common value at 40%.",
            "2.5% of price values are missing.",
            "3 outliers detected in price.",
            "No date columns — time-series not applicable.",
        ],
        column_analyses=[
            ColumnAnalysis(
                column_name="price",
                summary="Numeric column with near-normal distribution.",
                quality="2.5% missing values, 3 outliers.",
                patterns="Low variance relative to mean.",
            ),
            ColumnAnalysis(
                column_name="category",
                summary="Categorical with 3 values.",
                quality="No missing values.",
                patterns="A is most frequent.",
            ),
        ],
        anomalies=["3 outliers in price exceed Q3 + 1.5*IQR."],
        recommendations=[
            "Impute the 5 missing price values.",
            "Investigate the 3 price outliers.",
        ],
        methodology_notes="Standard IQR-based outlier detection was applied.",
        raw_response="",
    )


# ---------------------------------------------------------------------------
# generate_report — existence and format
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_creates_file(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        assert path.exists()

    def test_output_is_markdown(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        assert path.suffix == ".md"

    def test_filename_contains_csv_stem(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("sales_data.csv", mock_profile, mock_analysis, [])
        assert "sales_data" in path.name

    def test_filename_contains_timestamp(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        # Timestamp pattern: 8 digits
        import re
        assert re.search(r"\d{8}", path.name)

    def test_filename_sanitizes_spaces(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("my data file.csv", mock_profile, mock_analysis, [])
        assert " " not in path.name


# ---------------------------------------------------------------------------
# generate_report — section presence
# ---------------------------------------------------------------------------


class TestReportSections:
    @pytest.fixture()
    def report_content(self, out_dir, mock_profile, mock_analysis) -> str:
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        return path.read_text(encoding="utf-8")

    def test_has_title(self, report_content):
        assert "Data Analysis Report" in report_content

    def test_has_executive_summary(self, report_content):
        assert "Executive Summary" in report_content
        assert "200 rows" in report_content

    def test_has_key_findings(self, report_content):
        assert "Key Findings" in report_content

    def test_has_data_quality_table(self, report_content):
        assert "Data Quality" in report_content
        assert "| `price`" in report_content

    def test_has_column_analyses(self, report_content):
        assert "Column-by-Column Analysis" in report_content
        assert "`price`" in report_content

    def test_has_anomalies(self, report_content):
        assert "Anomalies" in report_content

    def test_has_recommendations(self, report_content):
        assert "Recommendations" in report_content

    def test_has_methodology(self, report_content):
        assert "Methodology" in report_content

    def test_has_footer(self, report_content):
        assert "claude-data-reporter" in report_content


# ---------------------------------------------------------------------------
# generate_report — data quality table ordering
# ---------------------------------------------------------------------------


class TestDataQualityTable:
    def test_missing_columns_sorted_first(self, out_dir, mock_profile, mock_analysis):
        """Price (2.5% missing) should appear before category (0% missing)."""
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        content = path.read_text(encoding="utf-8")
        price_pos = content.index("`price`")
        category_pos = content.index("`category`")
        # price has more missing — should come first in the table
        assert price_pos < category_pos


# ---------------------------------------------------------------------------
# generate_report — chart embedding
# ---------------------------------------------------------------------------


class TestChartEmbedding:
    def test_embeds_charts_with_relative_paths(
        self, out_dir, mock_profile, mock_analysis, tmp_path
    ):
        fake_chart = out_dir / "20250101_000000_dist_1.png"
        fake_chart.parent.mkdir(parents=True, exist_ok=True)
        fake_chart.touch()

        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report(
            "test.csv", mock_profile, mock_analysis, [fake_chart]
        )
        content = path.read_text(encoding="utf-8")
        # Should reference filename only, not absolute path
        assert fake_chart.name in content
        assert str(fake_chart.parent) not in content

    def test_no_charts_section_when_empty(self, out_dir, mock_profile, mock_analysis):
        reporter = ReportGenerator(out_dir)
        path = reporter.generate_report("test.csv", mock_profile, mock_analysis, [])
        content = path.read_text(encoding="utf-8")
        assert "No charts were generated" in content
