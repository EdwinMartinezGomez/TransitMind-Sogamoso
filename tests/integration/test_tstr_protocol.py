"""Integration test: TSTR protocol components."""
import numpy as np
import pandas as pd
import pytest

from src.layer1_timegan.evaluator import compute_fidelity_metrics
from src.shared.constants import NUMERIC_FEATURES


class TestTSTRProtocol:
    def test_fidelity_metrics_structure(self, sample_dataframe):
        df2 = sample_dataframe.copy()
        df2[NUMERIC_FEATURES] = df2[NUMERIC_FEATURES] + np.random.normal(0, 0.01, df2[NUMERIC_FEATURES].shape)
        metrics = compute_fidelity_metrics(sample_dataframe, df2)
        assert "js_divergence_avg" in metrics
        assert "correlation_diff" in metrics
        assert metrics["js_divergence_avg"] >= 0
