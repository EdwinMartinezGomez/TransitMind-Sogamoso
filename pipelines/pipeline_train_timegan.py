"""Pipeline: Train TimeGAN model (Fases 1-2)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.layer1_timegan.data_loader import normalize_data, create_sequences, split_dataset
from src.layer1_timegan.timegan_model import build_timegan
from src.layer1_timegan.trainer import TimeGANTrainer, create_dataloader
from src.shared.logger import setup_logger
from src.shared.utils import get_config, get_device, resolve_path, set_seed

logger = setup_logger("pipeline_train")

def main():
    config = get_config()
    set_seed(config.get("data", {}).get("seed", 42))
    device = get_device()
    logger.info("pipeline_train_start", device=device)

    # Load seed data
    csv_path = resolve_path("data/processed/train_seed.csv")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    logger.info("data_loaded", rows=len(df))

    # Normalize and create sequences
    df_norm, scaler = normalize_data(df)
    sequences = create_sequences(df_norm, seq_len=config.get("model", {}).get("seq_len", 24))
    train_seq, val_seq = split_dataset(sequences)

    # Create dataloader
    batch_size = config.get("training", {}).get("batch_size", 32)
    train_loader = create_dataloader(train_seq, batch_size=batch_size)

    # Build model and train
    components = build_timegan(config)
    trainer = TimeGANTrainer(components, config, device=device)
    metrics = trainer.train(train_loader, use_mlflow=True)

    logger.info("pipeline_train_complete", metrics=metrics)

if __name__ == "__main__":
    main()
