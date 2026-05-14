"""Pipeline: Train TimeGAN model (Fases 1-2)."""
import sys
from pathlib import Path
import re
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import mlflow
from src.layer1_timegan.data_loader import normalize_data, create_sequences, split_dataset
from src.layer1_timegan.timegan_model import build_timegan
from src.layer1_timegan.trainer import TimeGANTrainer, create_dataloader
from src.shared.logger import setup_logger
from src.shared.utils import get_config, get_device, resolve_path, set_seed

logger = setup_logger("pipeline_train")


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    candidates = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_checkpoint_name(path: Path) -> tuple[str, int] | None:
    match = re.match(r"checkpoint_(phase_[abc])_epoch(\d+)", path.stem)
    if not match:
        return None
    return match.group(1), int(match.group(2))

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
    use_mlflow = True

    checkpoint_dir = resolve_path(
        config.get("paths", {}).get("checkpoints", "models/timegan/checkpoints")
    )
    latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint is None:
        metrics = trainer.train(train_loader, use_mlflow=use_mlflow)
    else:
        parsed = _parse_checkpoint_name(latest_checkpoint)
        if parsed is None:
            logger.warning("checkpoint_name_unrecognized", path=str(latest_checkpoint))
            metrics = trainer.train(train_loader, use_mlflow=use_mlflow)
        else:
            phase, epoch = parsed
            trainer.load_checkpoint(str(latest_checkpoint))
            logger.info(
                "resume_from_checkpoint",
                path=str(latest_checkpoint),
                phase=phase,
                epoch=epoch,
            )

            if use_mlflow:
                mlflow.set_experiment(
                    config.get("mlflow", {}).get(
                        "experiment_name", "transitmind-timegan-layer1"
                    )
                )

            train_cfg = config.get("training", {})
            epochs_a = train_cfg.get("epochs_autoencoder", 200)
            epochs_b = train_cfg.get("epochs_supervisor", 200)
            epochs_c = train_cfg.get("epochs_joint", 500)

            metrics = {}

            def run_phase(phase_name, phase_fn, n_epochs):
                if n_epochs <= 0:
                    return None
                return trainer._run_phase_with_mlflow(
                    phase_name,
                    phase_fn,
                    train_loader,
                    n_epochs,
                    use_mlflow,
                )

            if phase == "phase_a":
                metrics["phase_a"] = run_phase(
                    "phase_a_autoencoder",
                    trainer.phase_a_autoencoder,
                    epochs_a - epoch,
                )
                metrics["phase_b"] = run_phase(
                    "phase_b_supervisor",
                    trainer.phase_b_supervisor,
                    epochs_b,
                )
                metrics["phase_c"] = run_phase(
                    "phase_c_joint_training",
                    trainer.phase_c_joint,
                    epochs_c,
                )
            elif phase == "phase_b":
                metrics["phase_b"] = run_phase(
                    "phase_b_supervisor",
                    trainer.phase_b_supervisor,
                    epochs_b - epoch,
                )
                metrics["phase_c"] = run_phase(
                    "phase_c_joint_training",
                    trainer.phase_c_joint,
                    epochs_c,
                )
            elif phase == "phase_c":
                metrics["phase_c"] = run_phase(
                    "phase_c_joint_training",
                    trainer.phase_c_joint,
                    epochs_c - epoch,
                )
            else:
                logger.warning("checkpoint_phase_unrecognized", phase=phase)
                metrics = trainer.train(train_loader, use_mlflow=use_mlflow)

    logger.info("pipeline_train_complete", metrics=metrics)

if __name__ == "__main__":
    main()
