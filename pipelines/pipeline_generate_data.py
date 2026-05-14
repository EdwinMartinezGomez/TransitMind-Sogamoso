"""Pipeline: Generate seed data (Fase 0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.layer1_timegan.data_loader import generate_seed_data, generate_validation_data
from src.shared.logger import setup_logger

logger = setup_logger("pipeline_generate")

def main():
    logger.info("pipeline_generate_data_start")
    df = generate_seed_data(n_days=90, seed=42)
    logger.info("seed_data_complete", rows=len(df))
    generate_validation_data(n_days=14, seed=123)
    logger.info("validation_data_complete")
    logger.info("pipeline_generate_data_done")

if __name__ == "__main__":
    main()
