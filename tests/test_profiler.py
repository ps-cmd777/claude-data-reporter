"""Tests for src/profiler.py — uses real pandas operations on small DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.profiler import DataProfile, DataProfiler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small DataFrame covering numeric, categorical, and date column types."""
    rng = np.random.default_rng(42)
    n = 100
    prices = rng.normal(50, 10, n)
    prices[5] = 200.0   # outlier
    prices[10] = -50.0  # outlier
    return pd.DataFrame(
        {
            "id": range(n),
            "price": prices,
            "quantity": rng.integers(1, 20, n).astype(float),
            "category": (["A", "B", "C"] * 34)[:n],
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "notes": [None] * 20 + ["text"] * 80,
        }
    )


@pytest.fixture()
def csv_file(tmp_path, sample_df) -> "Path":
    """Write sample_df to a temporary CSV and return its path."""
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def profiler_with_data(csv_file) -> DataProfiler:
    """DataProfiler that has loaded the sample CSV."""
    p = DataProfiler()
    p.load_csv(csv_file)
    return p


@pytest.fixture()
def profile(profiler_with_data) -> DataProfile:
    """Fully built DataProfile from the sample data."""
    return profiler_with_data.profile()


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_returns_dataframe(self, csv_file):
        p = DataProfiler()
        df = p.load_csv(csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, csv_file):
        p = DataProfiler()
        df = p.load_csv(csv_file)
        assert df.shape[0] == 100

    def test_date_column_parsed(self, csv_file):
        p = DataProfiler()
        df = p.load_csv(csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_missing_file_raises(self, tmp_path):
        p = DataProfiler()
        with pytest.raises(Exception):
            p.load_csv(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


class TestProfile:
    def test_shape_correct(self, profile, sample_df):
        assert profile.shape == sample_df.shape

    def test_numeric_columns_detected(self, profile):
        assert "price" in profile.numeric_columns
        assert "quantity" in profile.numeric_columns

    def test_categorical_columns_detected(self, profile):
        assert "category" in profile.categorical_columns
        assert "notes" in profile.categorical_columns

    def test_date_columns_detected(self, profile):
        assert "date" in profile.date_columns

    def test_missing_values_counted(self, profile):
        assert profile.column_profiles["notes"].missing_count == 20

    def test_missing_pct_computed(self, profile):
        # 20 / 100 = 20%
        assert profile.column_profiles["notes"].missing_pct == pytest.approx(20.0)

    def test_numeric_stats_present(self, profile):
        cp = profile.column_profiles["price"]
        assert cp.mean is not None
        assert cp.median is not None
        assert cp.std is not None
        assert cp.min_val is not None
        assert cp.max_val is not None

    def test_top_values_for_categorical(self, profile):
        cp = profile.column_profiles["category"]
        assert "A" in cp.top_values or "B" in cp.top_values

    def test_date_range_computed(self, profile):
        cp = profile.column_profiles["date"]
        assert cp.date_range_days == 99  # 100 daily steps → 99-day range

    def test_duplicate_rows_zero(self, profile):
        assert profile.duplicate_rows == 0

    def test_memory_usage_positive(self, profile):
        assert profile.memory_usage_mb > 0

    def test_profile_without_load_raises(self):
        p = DataProfiler()
        with pytest.raises(RuntimeError, match="load_csv"):
            p.profile()


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    def test_detects_known_outliers(self, profiler_with_data, profile):
        count, indices = profiler_with_data.detect_outliers("price")
        # We injected two obvious outliers (200 and -50)
        assert count >= 2

    def test_returns_indices_as_list(self, profiler_with_data, profile):
        _, indices = profiler_with_data.detect_outliers("price")
        assert isinstance(indices, list)

    def test_no_outliers_in_id_column(self, profiler_with_data, profile):
        count, _ = profiler_with_data.detect_outliers("id")
        # id is sequential 0–99, so IQR method should find no outliers
        assert count == 0


# ---------------------------------------------------------------------------
# compute_correlations
# ---------------------------------------------------------------------------


class TestComputeCorrelations:
    def test_returns_nested_dict(self, profiler_with_data, profile):
        corr = profiler_with_data.compute_correlations()
        assert isinstance(corr, dict)

    def test_symmetric(self, profiler_with_data, profile):
        corr = profiler_with_data.compute_correlations()
        for col_a in corr:
            for col_b, val in corr[col_a].items():
                if col_b in corr and col_a in corr[col_b]:
                    assert abs(corr[col_a][col_b] - corr[col_b][col_a]) < 0.0001

    def test_self_correlation_is_one(self, profiler_with_data, profile):
        corr = profiler_with_data.compute_correlations()
        for col in corr:
            assert corr[col][col] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_get_column_stats_returns_dict(self, profiler_with_data, profile):
        result = profiler_with_data.get_column_stats("price")
        assert isinstance(result, dict)
        assert "mean" in result
        assert "missing_count" in result

    def test_get_column_stats_unknown_column(self, profiler_with_data, profile):
        result = profiler_with_data.get_column_stats("nonexistent_col")
        assert "error" in result

    def test_get_correlations_returns_pairs(self, profiler_with_data, profile):
        result = profiler_with_data.get_correlations()
        assert "significant_pairs" in result
        assert "total_numeric_columns" in result

    def test_get_outliers_structure(self, profiler_with_data, profile):
        result = profiler_with_data.get_outliers()
        assert "columns_with_outliers" in result
        assert "total_columns_affected" in result

    def test_get_missing_values_structure(self, profiler_with_data, profile):
        result = profiler_with_data.get_missing_values()
        assert "columns_with_missing" in result
        assert "total_missing" in result
        # notes column has missing values
        cols = [c["column"] for c in result["columns_with_missing"]]
        assert "notes" in cols

    def test_tool_handlers_require_profile(self):
        p = DataProfiler()
        with pytest.raises(RuntimeError):
            p.get_column_stats("any")
