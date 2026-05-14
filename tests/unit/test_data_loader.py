"""Unit tests for data_loader module."""
import numpy as np
import pandas as pd
import pytest

from src.layer1_timegan.data_loader import (
    generate_seed_data, normalize_data, create_sequences, split_dataset,
)
from src.shared.constants import ALL_COLUMNS, NUMERIC_FEATURES


class TestGenerateSeedData:
    def test_generates_correct_shape(self):
        df = generate_seed_data(n_days=2, intersections=["carrera_11_norte"], seed=42)
        expected_rows = 2 * 96  # 2 days × 96 windows
        assert len(df) == expected_rows

    def test_all_columns_present(self):
        df = generate_seed_data(n_days=1, intersections=["carrera_11_norte"], seed=42)
        for col in ALL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_value_ranges(self):
        df = generate_seed_data(n_days=3, intersections=["carrera_11_norte"], seed=42)
        assert df["hour"].min() >= 0 and df["hour"].max() <= 23
        assert df["vehicle_flow"].min() >= 0 and df["vehicle_flow"].max() <= 300
        assert df["congestion_level"].min() >= 0 and df["congestion_level"].max() <= 1
        assert df["avg_speed_kmh"].min() >= 0

    def test_reproducibility(self):
        df1 = generate_seed_data(n_days=2, intersections=["carrera_11_norte"], seed=42)
        df2 = generate_seed_data(n_days=2, intersections=["carrera_11_norte"], seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_multiple_intersections(self):
        intersections = ["carrera_11_norte", "av_castellana_entrada"]
        df = generate_seed_data(n_days=1, intersections=intersections, seed=42)
        assert len(df) == 96 * len(intersections)


class TestNormalizeData:
    def test_normalized_range(self, sample_dataframe):
        df_norm, scaler = normalize_data(sample_dataframe)
        for col in NUMERIC_FEATURES:
            if col in df_norm.columns:
                assert df_norm[col].min() >= -0.01
                assert df_norm[col].max() <= 1.01


class TestCreateSequences:
    def test_sequence_shape(self, sample_dataframe):
        df_norm, _ = normalize_data(sample_dataframe)
        seqs = create_sequences(df_norm, seq_len=10, group_by_intersection=False)
        assert seqs.ndim == 3
        assert seqs.shape[1] == 10
        assert seqs.shape[2] == len(NUMERIC_FEATURES)


class TestSplitDataset:
    def test_split_ratio(self):
        data = np.random.randn(100, 24, 11)
        train, val = split_dataset(data, train_ratio=0.8)
        assert len(train) == 80
        assert len(val) == 20
