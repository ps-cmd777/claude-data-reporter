"""Tests for src/visualizer.py — generates real PNGs using tmp_path fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.profiler import ColumnProfile, DataProfile, DataProfiler
from src.visualizer import Visualizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_dir(tmp_path) -> Path:
    """Temporary output directory for chart PNGs."""
    d = tmp_path / "charts"
    d.mkdir()
    return d


def _make_profile(df: pd.DataFrame) -> tuple[DataProfiler, DataProfile]:
    """Helper: build a DataProfile from a DataFrame via a temporary CSV."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        tmp = Path(f.name)
    p = DataProfiler()
    p.load_csv(tmp)
    profile = p.profile()
    tmp.unlink(missing_ok=True)
    return p, profile


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """DataFrame with two numeric columns — supports heatmap."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "price": rng.normal(50, 10, 80),
            "quantity": rng.integers(1, 20, 80).astype(float),
        }
    )


@pytest.fixture()
def full_df() -> pd.DataFrame:
    """DataFrame with numeric, categorical, date columns, and some missing values."""
    rng = np.random.default_rng(1)
    n = 80
    prices = rng.normal(50, 10, n)
    prices[0] = np.nan  # introduce missing
    return pd.DataFrame(
        {
            "price": prices,
            "quantity": rng.integers(1, 20, n).astype(float),
            "category": (["A", "B", "C"] * 27)[:n],
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
        }
    )


# ---------------------------------------------------------------------------
# plot_distributions
# ---------------------------------------------------------------------------


class TestPlotDistributions:
    def test_creates_png_files(self, out_dir, numeric_df):
        _, profile = _make_profile(numeric_df)
        viz = Visualizer(out_dir)
        paths = viz.plot_distributions(numeric_df, profile)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"

    def test_returns_empty_list_when_no_numerics(self, out_dir):
        df = pd.DataFrame({"cat": ["A", "B", "C"]})
        _, profile = _make_profile(df)
        viz = Visualizer(out_dir)
        paths = viz.plot_distributions(df, profile)
        assert paths == []


# ---------------------------------------------------------------------------
# plot_correlation_heatmap
# ---------------------------------------------------------------------------


class TestPlotCorrelationHeatmap:
    def test_creates_heatmap_with_two_numeric_cols(self, out_dir, numeric_df):
        _, profile = _make_profile(numeric_df)
        viz = Visualizer(out_dir)
        path = viz.plot_correlation_heatmap(profile)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_returns_none_for_single_numeric_col(self, out_dir):
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        _, profile = _make_profile(df)
        viz = Visualizer(out_dir)
        result = viz.plot_correlation_heatmap(profile)
        assert result is None

    def test_returns_none_for_no_numeric_cols(self, out_dir):
        df = pd.DataFrame({"cat": ["A", "B", "C"]})
        _, profile = _make_profile(df)
        viz = Visualizer(out_dir)
        result = viz.plot_correlation_heatmap(profile)
        assert result is None


# ---------------------------------------------------------------------------
# plot_missing_values
# ---------------------------------------------------------------------------


class TestPlotMissingValues:
    def test_creates_chart_when_missing_values_exist(self, out_dir, full_df):
        _, profile = _make_profile(full_df)
        viz = Visualizer(out_dir)
        path = viz.plot_missing_values(profile)
        assert path is not None
        assert path.exists()

    def test_returns_none_when_no_missing_values(self, out_dir, numeric_df):
        _, profile = _make_profile(numeric_df)
        viz = Visualizer(out_dir)
        result = viz.plot_missing_values(profile)
        assert result is None


# ---------------------------------------------------------------------------
# plot_top_values
# ---------------------------------------------------------------------------


class TestPlotTopValues:
    def test_creates_charts_for_categorical_cols(self, out_dir, full_df):
        _, profile = _make_profile(full_df)
        viz = Visualizer(out_dir)
        paths = viz.plot_top_values(full_df, profile)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()

    def test_returns_empty_for_no_categorical_cols(self, out_dir, numeric_df):
        _, profile = _make_profile(numeric_df)
        viz = Visualizer(out_dir)
        paths = viz.plot_top_values(numeric_df, profile)
        assert paths == []


# ---------------------------------------------------------------------------
# plot_time_series
# ---------------------------------------------------------------------------


class TestPlotTimeSeries:
    def test_creates_chart_with_date_col(self, out_dir, full_df):
        _, profile = _make_profile(full_df)
        viz = Visualizer(out_dir)
        paths = viz.plot_time_series(full_df, profile)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()

    def test_returns_empty_without_date_cols(self, out_dir, numeric_df):
        _, profile = _make_profile(numeric_df)
        viz = Visualizer(out_dir)
        paths = viz.plot_time_series(numeric_df, profile)
        assert paths == []


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------


class TestGenerateAll:
    def test_returns_list_of_paths(self, out_dir, full_df):
        _, profile = _make_profile(full_df)
        viz = Visualizer(out_dir)
        paths = viz.generate_all(full_df, profile)
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)
        assert all(p.exists() for p in paths)

    def test_all_paths_are_png(self, out_dir, full_df):
        _, profile = _make_profile(full_df)
        viz = Visualizer(out_dir)
        paths = viz.generate_all(full_df, profile)
        for p in paths:
            assert p.suffix == ".png"
