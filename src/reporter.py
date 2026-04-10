"""Report generation module — assembles analysis results into a Markdown report."""

from __future__ import annotations

from pathlib import Path

from src.analyzer import AnalysisResult
from src.profiler import DataProfile
from src.utils import format_timestamp, sanitize_filename


class ReportGenerator:
    """Assembles a structured Markdown report from profiling and analysis data."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with the directory where the report file will be written."""
        self.output_dir = output_dir

    def generate_report(
        self,
        csv_filename: str,
        profile: DataProfile,
        analysis: AnalysisResult,
        chart_paths: list[Path],
    ) -> Path:
        """Assemble all report sections and write the Markdown file.

        Returns the path to the generated report file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = format_timestamp()
        stem = sanitize_filename(Path(csv_filename).stem)
        filename = f"report_{stem}_{timestamp}.md"
        output_path = self.output_dir / filename

        # Filter out any None values that chart generators may return
        valid_charts: list[Path] = [p for p in chart_paths if p is not None]

        sections = [
            self._render_header(csv_filename, profile, timestamp),
            self._render_executive_summary(analysis),
            self._render_key_findings(analysis),
            self._render_data_quality_table(profile),
            self._render_column_analyses(analysis),
            self._render_anomalies(analysis),
            self._render_visualizations(valid_charts),
            self._render_recommendations(analysis),
            self._render_methodology(profile, analysis),
            self._render_footer(),
        ]

        report_content = "\n\n---\n\n".join(s for s in sections if s.strip())
        output_path.write_text(report_content, encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_header(
        self, csv_filename: str, profile: DataProfile, timestamp: str
    ) -> str:
        """Render the report title and dataset metadata block."""
        rows, cols = profile.shape
        lines = [
            f"# Data Analysis Report — `{csv_filename}`",
            "",
            f"> Generated: {timestamp[:8]} {timestamp[9:].replace('_', ':')} UTC  ",
            f"> Source file: `{csv_filename}`  ",
            f"> Dataset: {rows:,} rows × {cols} columns  ",
            f"> Memory: {profile.memory_usage_mb} MB  ",
            f"> Missing values: {profile.total_missing_pct}%  ",
            f"> Duplicate rows: {profile.duplicate_rows}  ",
        ]
        if profile.date_columns:
            lines.append(f"> Date columns detected: {', '.join(profile.date_columns)}  ")
        lines += [
            "",
            "**Column summary**",
            f"- Numeric: {len(profile.numeric_columns)} column(s) — "
            + (", ".join(f"`{c}`" for c in profile.numeric_columns) or "none"),
            f"- Categorical: {len(profile.categorical_columns)} column(s) — "
            + (", ".join(f"`{c}`" for c in profile.categorical_columns) or "none"),
            f"- Date/Time: {len(profile.date_columns)} column(s) — "
            + (", ".join(f"`{c}`" for c in profile.date_columns) or "none"),
        ]
        return "\n".join(lines)

    def _render_executive_summary(self, analysis: AnalysisResult) -> str:
        """Render the executive summary section."""
        return f"## 1. Executive Summary\n\n{analysis.executive_summary}"

    def _render_key_findings(self, analysis: AnalysisResult) -> str:
        """Render the top 5 key findings as a numbered list."""
        if not analysis.key_findings:
            return "## 2. Key Findings\n\n_No key findings were returned._"
        items = "\n".join(
            f"{i+1}. {finding}" for i, finding in enumerate(analysis.key_findings[:5])
        )
        return f"## 2. Key Findings\n\n{items}"

    def _render_data_quality_table(self, profile: DataProfile) -> str:
        """Render a Markdown table showing data quality metrics per column.

        Sorted by missing percentage descending so problem columns appear first.
        """
        header = (
            "## 3. Data Quality Assessment\n\n"
            "| Column | Type | Missing # | Missing % | Unique | Outliers |\n"
            "|--------|------|----------:|----------:|-------:|---------:|"
        )
        rows = sorted(
            profile.column_profiles.values(),
            key=lambda cp: cp.missing_pct,
            reverse=True,
        )
        table_rows = []
        for cp in rows:
            outliers = cp.outlier_count if cp.outlier_count is not None else "—"
            table_rows.append(
                f"| `{cp.name}` | {cp.dtype} | {cp.missing_count:,} "
                f"| {cp.missing_pct:.1f}% | {cp.unique_count:,} | {outliers} |"
            )
        return header + "\n" + "\n".join(table_rows)

    def _render_column_analyses(self, analysis: AnalysisResult) -> str:
        """Render the column-by-column analysis section."""
        if not analysis.column_analyses:
            return "## 4. Column-by-Column Analysis\n\n_No column-level analysis available._"

        sections = ["## 4. Column-by-Column Analysis"]
        for ca in analysis.column_analyses:
            block = [
                f"### `{ca.column_name}`",
                "",
                f"**Summary:** {ca.summary}" if ca.summary else "",
                f"**Data Quality:** {ca.quality}" if ca.quality else "",
                f"**Patterns:** {ca.patterns}" if ca.patterns else "",
            ]
            sections.append("\n".join(line for line in block if line is not None))

        return "\n\n".join(sections)

    def _render_anomalies(self, analysis: AnalysisResult) -> str:
        """Render detected anomalies and outliers."""
        if not analysis.anomalies:
            return "## 5. Anomalies & Outliers\n\n_No significant anomalies detected._"
        items = "\n".join(f"- {anomaly}" for anomaly in analysis.anomalies)
        return f"## 5. Anomalies & Outliers\n\n{items}"

    def _render_visualizations(self, chart_paths: list[Path]) -> str:
        """Render chart embed section with relative paths.

        Uses relative paths so the report is portable if the whole output
        folder is moved or shared.
        """
        if not chart_paths:
            return "## 6. Visualizations\n\n_No charts were generated._"

        lines = ["## 6. Visualizations", ""]
        for path in chart_paths:
            chart_name = path.stem
            # Friendly display name — strip timestamp prefix
            parts = chart_name.split("_", 2)
            display = parts[2].replace("_", " ").title() if len(parts) >= 3 else chart_name
            relative = path.name  # charts are in the same output dir as the report
            lines.append(f"### {display}")
            lines.append(f"![{display}]({relative})")
            lines.append("")

        return "\n".join(lines)

    def _render_recommendations(self, analysis: AnalysisResult) -> str:
        """Render actionable recommendations as a numbered list."""
        if not analysis.recommendations:
            return "## 7. Recommendations\n\n_No recommendations were returned._"
        items = "\n".join(
            f"{i+1}. {rec}" for i, rec in enumerate(analysis.recommendations)
        )
        return f"## 7. Recommendations\n\n{items}"

    def _render_methodology(self, profile: DataProfile, analysis: AnalysisResult) -> str:
        """Render the methodology section describing how the analysis was done."""
        auto_method = (
            f"**Profiling:** Pandas v2+ was used to compute column-level statistics "
            f"for all {profile.shape[1]} columns, including missing value counts, "
            f"cardinality, numeric distributions (mean, median, std, IQR, quartiles), "
            f"and categorical value frequencies. Outliers were identified using the "
            f"IQR method (Q1 − 1.5·IQR, Q3 + 1.5·IQR). Date columns were detected "
            f"heuristically by column name and successful parse rate.\n\n"
            f"**AI Analysis:** The data profile was sent to the Claude API "
            f"(`claude-data-reporter`) using the tool-use pattern. Claude called "
            f"specialized profiler tools to retrieve column statistics, correlation "
            f"data, outlier summaries, and missing value details before generating "
            f"the narrative analysis above."
        )
        claude_method = analysis.methodology_notes or ""
        combined = auto_method
        if claude_method:
            combined += f"\n\n{claude_method}"
        return f"## 8. Methodology\n\n{combined}"

    def _render_footer(self) -> str:
        """Render the report footer."""
        return (
            "_Report generated by [claude-data-reporter]"
            "(https://github.com/shushan/claude-data-reporter) "
            "— Built with [Claude Code](https://claude.ai/code)_"
        )
