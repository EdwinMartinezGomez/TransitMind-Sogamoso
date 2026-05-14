"""Integration test: data generation pipeline."""
import pytest
from src.layer1_timegan.data_loader import generate_seed_data, normalize_data, create_sequences
from src.shared.constants import NUMERIC_FEATURES


class TestPipelineGenerate:
    def test_full_data_pipeline(self):
        df = generate_seed_data(n_days=2, intersections=["carrera_11_norte"], seed=42)
        assert len(df) > 0
        df_norm, scaler = normalize_data(df)
        seqs = create_sequences(df_norm, seq_len=24, group_by_intersection=True)
        assert seqs.shape[1] == 24
        assert seqs.shape[2] == len(NUMERIC_FEATURES)
