"""Pipeline: TSTR Evaluation (Fase 4)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.layer1_timegan.evaluator import (
    tstr_evaluation, discriminative_score, visualize_distributions,
    compute_fidelity_metrics, generate_evaluation_report,
)
from src.layer1_timegan.generator import TrafficDataGenerator
from src.shared.logger import setup_logger
from src.shared.utils import resolve_path

logger = setup_logger("pipeline_evaluate")

def main():
    logger.info("pipeline_evaluate_start")

    # Load real validation data
    real_path = resolve_path("data/validation/pilot_carrera11.csv")
    real_df = pd.read_csv(real_path, parse_dates=["timestamp"])

    # Generate synthetic data
    gen = TrafficDataGenerator()
    synthetic_df = gen.generate(n_samples=200, intersection_id="carrera_11_norte")

    # Run evaluations
    tstr_metrics = tstr_evaluation(synthetic_df, real_df)
    disc_metrics = discriminative_score(synthetic_df, real_df)
    fidelity = compute_fidelity_metrics(synthetic_df, real_df)
    viz_paths = visualize_distributions(synthetic_df, real_df)

    all_metrics = {"tstr": tstr_metrics, "discriminative": disc_metrics, "fidelity": fidelity}
    report = generate_evaluation_report(all_metrics)

    logger.info("pipeline_evaluate_complete", tstr_score=tstr_metrics.get("tstr_score"))

if __name__ == "__main__":
    main()
