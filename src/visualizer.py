"""Chart generation module — produces PNG visualizations from a DataProfile."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Must be before any other matplotlib import (headless/CI safe)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import numpy as np
import pandas as pd

from src.profiler import DataProfile
from src.utils import format_timestamp


_CHART_DPI = 150
_CHART_STYLE = "seaborn-v0_8-whitegrid"
_MAX_SUBPLOTS_PER_FIGURE = 6
_TOP_N_VALUES = 10


class Visualizer:
    """Generates and saves PNG charts from a profiled dataset."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize with the directory where chart PNGs will be saved."""
        self.output_dir = output_dir
        self._timestamp = format_timestamp()
        try:
            plt.style.use(_CHART_STYLE)
        except OSError:
            pass  # Style not available in all environments; use default

    def generate_all(self, df: pd.DataFrame, profile: DataProfile) -> list[Path]:
        """Run all 5 chart generators and return the list of PNG paths created."""
        paths: list[Path] = []
        paths.extend(self.plot_distributions(df, profile))
        heatmap = self.plot_correlation_heatmap(profile)
        if heatmap is not None:
            paths.append(heatmap)
        missing_chart = self.plot_missing_values(profile)
        if missing_chart is not None:
            paths.append(missing_chart)
        paths.extend(self.plot_top_values(df, profile))
        paths.extend(self.plot_time_series(df, profile))
        return paths

    # ------------------------------------------------------------------
    # Chart methods
    # ------------------------------------------------------------------

    def plot_distributions(
        self, df: pd.DataFrame, profile: DataProfile
    ) -> list[Path]:
        """Plot histograms for all numeric columns (max 6 per figure).

        Returns list of saved PNG paths. Returns [] if no numeric columns.
        """
        cols = profile.numeric_columns
        if not cols:
            return []

        paths: list[Path] = []
        for chunk_start in range(0, len(cols), _MAX_SUBPLOTS_PER_FIGURE):
            chunk = cols[chunk_start : chunk_start + _MAX_SUBPLOTS_PER_FIGURE]
            n = len(chunk)
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
            )
            fig.suptitle("Numeric Column Distributions", fontsize=14, fontweight="bold")

            for idx, col in enumerate(chunk):
                ax = axes[idx // ncols][idx % ncols]
                data = df[col].dropna()
                ax.hist(data, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
                ax.set_title(col, fontsize=11)
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                # Annotate with mean/median
                if len(data) > 0:
                    ax.axvline(
                        data.mean(), color="#DD8452", linestyle="--", linewidth=1.2,
                        label=f"Mean: {data.mean():.2f}"
                    )
                    ax.axvline(
                        data.median(), color="#55A868", linestyle=":", linewidth=1.2,
                        label=f"Median: {data.median():.2f}"
                    )
                    ax.legend(fontsize=8)

            # Hide unused axes in the last row
            for idx in range(len(chunk), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)

            fig.tight_layout()
            chunk_label = f"dist_{chunk_start // _MAX_SUBPLOTS_PER_FIGURE + 1}"
            paths.append(self._save_figure(fig, chunk_label))

        return paths

    def plot_correlation_heatmap(self, profile: DataProfile) -> Path | None:
        """Plot a Pearson correlation heatmap for numeric columns.

        Returns None if fewer than 2 numeric columns exist.
        """
        if len(profile.numeric_columns) < 2:
            return None
        if not profile.correlation_matrix:
            return None

        corr_data = {
            col: profile.correlation_matrix[col]
            for col in profile.numeric_columns
            if col in profile.correlation_matrix
        }
        if len(corr_data) < 2:
            return None

        cols = list(corr_data.keys())
        matrix = [[corr_data[r].get(c, float("nan")) for c in cols] for r in cols]
        corr_df = pd.DataFrame(matrix, index=cols, columns=cols)

        size = max(6, len(cols) * 0.9)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 9},
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        fig.tight_layout()
        return self._save_figure(fig, "correlation_heatmap")

    def plot_missing_values(self, profile: DataProfile) -> Path | None:
        """Plot a horizontal bar chart showing missing value % per column.

        Returns None if no missing values exist.
        """
        missing = [
            (cp.name, cp.missing_pct)
            for cp in profile.column_profiles.values()
            if cp.missing_count > 0
        ]
        if not missing:
            return None

        missing.sort(key=lambda x: x[1], reverse=True)
        names, pcts = zip(*missing)

        fig_height = max(4, len(names) * 0.5)
        fig, ax = plt.subplots(figsize=(9, fig_height))
        bars = ax.barh(names, pcts, color="#DD8452", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Missing Values (%)")
        ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
        ax.set_xlim(0, max(pcts) * 1.15)

        for bar, pct in zip(bars, pcts):
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.invert_yaxis()
        fig.tight_layout()
        return self._save_figure(fig, "missing_values")

    def plot_top_values(
        self, df: pd.DataFrame, profile: DataProfile
    ) -> list[Path]:
        """Plot bar charts showing top N value counts for categorical columns.

        Returns [] if no categorical columns with more than 1 unique value.
        """
        paths: list[Path] = []
        for col in profile.categorical_columns:
            cp = profile.column_profiles[col]
            if cp.unique_count <= 1 or not cp.top_values:
                continue
            labels = list(cp.top_values.keys())[:_TOP_N_VALUES]
            values = list(cp.top_values.values())[:_TOP_N_VALUES]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = sns.color_palette("muted", len(labels))
            bars = ax.bar(labels, values, color=colors, edgecolor="white")
            ax.set_title(
                f"Top Values — {col}", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right", fontsize=9)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    str(val),
                    ha="center",
                    fontsize=8,
                )

            fig.tight_layout()
            safe_col = col.replace(" ", "_").replace("/", "_")[:30]
            paths.append(self._save_figure(fig, f"top_values_{safe_col}"))

        return paths

    def plot_time_series(
        self, df: pd.DataFrame, profile: DataProfile
    ) -> list[Path]:
        """Plot aggregated numeric trends over time for each date column.

        Returns [] if no date columns exist.
        """
        if not profile.date_columns or not profile.numeric_columns:
            return []

        paths: list[Path] = []
        for date_col in profile.date_columns:
            numeric_targets = [
                c for c in profile.numeric_columns
                if c != date_col
            ][:4]  # limit to 4 series per chart to keep it readable

            if not numeric_targets:
                continue

            ts_df = df[[date_col] + numeric_targets].dropna(subset=[date_col]).copy()
            ts_df = ts_df.set_index(date_col).sort_index()

            # Resample to a sensible frequency based on date range
            date_range_days = (ts_df.index.max() - ts_df.index.min()).days
            if date_range_days > 365:
                freq = "ME"
            elif date_range_days > 30:
                freq = "W"
            else:
                freq = "D"

            try:
                ts_agg = ts_df[numeric_targets].resample(freq).mean()
            except Exception:
                continue

            if ts_agg.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, 5))
            palette = sns.color_palette("tab10", len(numeric_targets))
            for col, color in zip(numeric_targets, palette):
                ax.plot(
                    ts_agg.index,
                    ts_agg[col],
                    label=col,
                    color=color,
                    linewidth=1.8,
                    marker="o",
                    markersize=3,
                )

            ax.set_title(
                f"Time Series — {date_col}", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel(date_col)
            ax.set_ylabel("Value (mean)")
            ax.legend(fontsize=9)
            plt.xticks(rotation=30, ha="right")
            fig.tight_layout()
            safe_col = date_col.replace(" ", "_")[:30]
            paths.append(self._save_figure(fig, f"time_series_{safe_col}"))

        return paths

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, name: str) -> Path:
        """Save a matplotlib figure as PNG and close it to free memory.

        Returns the path of the saved file.
        """
        filename = f"{self._timestamp}_{name}.png"
        path = self.output_dir / filename
        fig.savefig(path, dpi=_CHART_DPI, bbox_inches="tight")
        plt.close(fig)
        return path
