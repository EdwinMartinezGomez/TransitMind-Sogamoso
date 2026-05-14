"""
TransitMind Sogamoso — TimeGAN Trainer (Fase 2)
=================================================
3-phase training protocol for TimeGAN:
    Phase A: Autoencoder pre-training (Embedder + Recovery)
    Phase B: Supervisor pre-training (Generator + Supervisor)
    Phase C: Joint adversarial training (all components)

Reference:
    Yoon, Jarrett, & van der Schaar (2019).
    "Time-series Generative Adversarial Networks." NeurIPS.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.shared.logger import setup_logger
from src.shared.utils import ensure_dir, format_duration, get_config, resolve_path

logger = setup_logger("trainer")


class TimeGANTrainer:
    """
    Trainer for the TimeGAN model with 3-phase training protocol.

    Integrates with MLflow for experiment tracking and supports
    checkpoint saving/loading for training resumption.
    """

    def __init__(
        self,
        components: Dict[str, nn.Module],
        config: Optional[Dict] = None,
        device: str = "cpu",
    ):
        """
        Initialize the TimeGAN trainer.

        Args:
            components: Dictionary with keys 'embedder', 'recovery',
                'generator', 'supervisor', 'discriminator'.
            config: Training configuration. Loads from YAML if None.
            device: PyTorch device ('cpu' or 'cuda').
        """
        if config is None:
            config = get_config()

        self.config = config
        self.device = torch.device(device)

        # Unpack components and move to device
        self.embedder = components["embedder"].to(self.device)
        self.recovery = components["recovery"].to(self.device)
        self.generator = components["generator"].to(self.device)
        self.supervisor = components["supervisor"].to(self.device)
        self.discriminator = components["discriminator"].to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.lr_gen = train_cfg.get("lr_generator", 0.001)
        self.lr_disc = train_cfg.get("lr_discriminator", 0.001)
        self.gamma = train_cfg.get("gamma", 1.0)
        self.checkpoint_every = train_cfg.get("checkpoint_every", 50)
        self.log_every_ab = train_cfg.get("log_every_phase_ab", 10)
        self.log_every_c = train_cfg.get("log_every_phase_c", 25)

        # Model config
        model_cfg = config.get("model", {})
        self.hidden_dim = model_cfg.get("hidden_dim", 64)
        self.noise_dim = model_cfg.get("noise_dim", 64)
        self.seq_len = model_cfg.get("seq_len", 24)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Optimizers (initialized per phase)
        self._setup_optimizers()

        # Paths
        paths_cfg = config.get("paths", {})
        self.checkpoint_dir = resolve_path(
            paths_cfg.get("checkpoints", "models/timegan/checkpoints")
        )
        self.best_model_dir = resolve_path(
            paths_cfg.get("best_model", "models/timegan/best_model")
        )
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.best_model_dir)

        # Training history
        self.history: Dict[str, list] = {
            "phase_a_loss": [],
            "phase_b_loss": [],
            "phase_c_d_loss": [],
            "phase_c_g_loss": [],
            "phase_c_recon_loss": [],
        }

        logger.info(
            "trainer_initialized",
            device=str(self.device),
            lr_gen=self.lr_gen,
            lr_disc=self.lr_disc,
            gamma=self.gamma,
        )

    def _setup_optimizers(self) -> None:
        """Set up optimizers for each training phase."""
        # Phase A: Autoencoder
        self.opt_autoencoder = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.lr_gen,
        )

        # Phase B: Supervisor
        self.opt_supervisor = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.lr_gen,
        )

        # Phase C: Generator side (embedder + recovery + generator + supervisor)
        self.opt_generator = torch.optim.Adam(
            list(self.embedder.parameters())
            + list(self.recovery.parameters())
            + list(self.generator.parameters())
            + list(self.supervisor.parameters()),
            lr=self.lr_gen,
        )

        # Phase C: Discriminator
        self.opt_discriminator = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr_disc,
        )

    def phase_a_autoencoder(
        self,
        dataloader: DataLoader,
        n_epochs: int = 200,
    ) -> Dict[str, float]:
        """
        Phase A: Train the Autoencoder (Embedder + Recovery).

        Loss: reconstruction_loss = MSE(Recovery(Embedder(X)), X)
        Optimizes: Embedder + Recovery parameters.

        Args:
            dataloader: DataLoader providing real sequences.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary with final metrics.
        """
        logger.info("phase_a_start", n_epochs=n_epochs)
        start_time = time.time()

        self.embedder.train()
        self.recovery.train()

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in dataloader:
                batch = batch.to(self.device)

                # Forward pass
                h = self.embedder(batch)
                x_hat = self.recovery(h)

                # Reconstruction loss
                loss = self.mse_loss(x_hat, batch)

                # Backward pass
                self.opt_autoencoder.zero_grad()
                loss.backward()
                self.opt_autoencoder.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["phase_a_loss"].append(avg_loss)

            if epoch % self.log_every_ab == 0:
                logger.info(
                    "phase_a_epoch",
                    epoch=epoch,
                    reconstruction_loss=round(avg_loss, 6),
                )

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch, "phase_a", {"reconstruction_loss": avg_loss})

        elapsed = time.time() - start_time
        final_loss = self.history["phase_a_loss"][-1] if self.history["phase_a_loss"] else 0.0

        logger.info(
            "phase_a_complete",
            final_loss=round(final_loss, 6),
            duration=format_duration(elapsed),
        )

        return {"reconstruction_loss": final_loss, "duration_s": elapsed}

    def phase_b_supervisor(
        self,
        dataloader: DataLoader,
        n_epochs: int = 200,
    ) -> Dict[str, float]:
        """
        Phase B: Train the Supervisor in latent space.

        Loss: supervised_loss = MSE(Supervisor(H_t), H_{t+1})
        where H = Embedder(X).
        Optimizes: Generator + Supervisor parameters.

        Args:
            dataloader: DataLoader providing real sequences.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary with final metrics.
        """
        logger.info("phase_b_start", n_epochs=n_epochs)
        start_time = time.time()

        self.embedder.eval()  # Freeze embedder weights (use trained embedder)
        self.generator.train()
        self.supervisor.train()

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for (batch,) in dataloader:
                batch = batch.to(self.device)

                # Embed real data
                with torch.no_grad():
                    h_real = self.embedder(batch)

                # Supervised prediction: predict h_{t+1} from h_{t}
                h_supervised = self.supervisor(h_real)

                # Supervised loss: compare shifted sequences
                loss = self.mse_loss(
                    h_supervised[:, :-1, :],  # Predicted h_{t+1}
                    h_real[:, 1:, :],         # Actual h_{t+1}
                )

                # Backward pass
                self.opt_supervisor.zero_grad()
                loss.backward()
                self.opt_supervisor.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.history["phase_b_loss"].append(avg_loss)

            if epoch % self.log_every_ab == 0:
                logger.info(
                    "phase_b_epoch",
                    epoch=epoch,
                    supervised_loss=round(avg_loss, 6),
                )

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch, "phase_b", {"supervised_loss": avg_loss})

        elapsed = time.time() - start_time
        final_loss = self.history["phase_b_loss"][-1] if self.history["phase_b_loss"] else 0.0

        logger.info(
            "phase_b_complete",
            final_loss=round(final_loss, 6),
            duration=format_duration(elapsed),
        )

        return {"supervised_loss": final_loss, "duration_s": elapsed}

    def phase_c_joint(
        self,
        dataloader: DataLoader,
        n_epochs: int = 500,
    ) -> Dict[str, float]:
        """
        Phase C: Joint adversarial training of all components.

        Losses:
            - discriminator_loss = BCE(D(H_real), 1) + BCE(D(H_fake), 0)
            - generator_loss = adversarial + gamma * supervised_loss
            - reconstruction_loss (maintains autoencoder fidelity)

        Optimizes: All components alternately.

        Args:
            dataloader: DataLoader providing real sequences.
            n_epochs: Number of training epochs.

        Returns:
            Dictionary with final metrics.
        """
        logger.info("phase_c_start", n_epochs=n_epochs, gamma=self.gamma)
        start_time = time.time()

        # Set all to training mode
        self.embedder.train()
        self.recovery.train()
        self.generator.train()
        self.supervisor.train()
        self.discriminator.train()

        for epoch in range(1, n_epochs + 1):
            d_losses, g_losses, r_losses = [], [], []

            for (batch,) in dataloader:
                batch = batch.to(self.device)
                batch_size = batch.size(0)

                # === Generate fake data ===
                z = torch.randn(
                    batch_size, self.seq_len, self.noise_dim, device=self.device
                )

                # Real embeddings
                h_real = self.embedder(batch)

                # Fake latent from generator + supervisor
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)

                # Labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # === Train Discriminator ===
                d_real = self.discriminator(h_real.detach())
                d_fake = self.discriminator(h_fake.detach())

                d_loss_real = self.bce_loss(d_real, real_labels)
                d_loss_fake = self.bce_loss(d_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake

                self.opt_discriminator.zero_grad()
                d_loss.backward()
                self.opt_discriminator.step()

                # === Train Generator (+ Embedder + Recovery + Supervisor) ===
                # Re-embed and regenerate (fresh computation graph)
                h_real = self.embedder(batch)
                h_fake_raw = self.generator(z)
                h_fake = self.supervisor(h_fake_raw)

                # Adversarial loss: fool discriminator
                d_fake_for_g = self.discriminator(h_fake)
                g_adversarial_loss = self.bce_loss(d_fake_for_g, real_labels)

                # Supervised loss
                h_supervised = self.supervisor(h_real)
                supervised_loss = self.mse_loss(
                    h_supervised[:, :-1, :],
                    h_real[:, 1:, :],
                )

                # Reconstruction loss
                x_hat = self.recovery(h_real)
                reconstruction_loss = self.mse_loss(x_hat, batch)

                # Moment matching loss (mean + variance alignment)
                mean_loss = torch.mean(
                    torch.abs(torch.mean(h_real, dim=0) - torch.mean(h_fake, dim=0))
                )
                var_loss = torch.mean(
                    torch.abs(torch.var(h_real, dim=0) - torch.var(h_fake, dim=0))
                )

                # Combined generator loss
                g_loss = (
                    g_adversarial_loss
                    + self.gamma * supervised_loss
                    + 10.0 * reconstruction_loss
                    + mean_loss
                    + var_loss
                )

                self.opt_generator.zero_grad()
                g_loss.backward()
                self.opt_generator.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                r_losses.append(reconstruction_loss.item())

            avg_d = np.mean(d_losses)
            avg_g = np.mean(g_losses)
            avg_r = np.mean(r_losses)

            self.history["phase_c_d_loss"].append(avg_d)
            self.history["phase_c_g_loss"].append(avg_g)
            self.history["phase_c_recon_loss"].append(avg_r)

            if epoch % self.log_every_c == 0:
                logger.info(
                    "phase_c_epoch",
                    epoch=epoch,
                    d_loss=round(avg_d, 6),
                    g_loss=round(avg_g, 6),
                    recon_loss=round(avg_r, 6),
                )

            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(
                    epoch,
                    "phase_c",
                    {"d_loss": avg_d, "g_loss": avg_g, "recon_loss": avg_r},
                )

        elapsed = time.time() - start_time

        final_metrics = {
            "d_loss": float(avg_d),
            "g_loss": float(avg_g),
            "recon_loss": float(avg_r),
            "duration_s": elapsed,
        }

        logger.info(
            "phase_c_complete",
            d_loss=round(avg_d, 6),
            g_loss=round(avg_g, 6),
            recon_loss=round(avg_r, 6),
            duration=format_duration(elapsed),
        )

        return final_metrics

    def train(
        self,
        dataloader: DataLoader,
        epochs_a: Optional[int] = None,
        epochs_b: Optional[int] = None,
        epochs_c: Optional[int] = None,
        use_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full 3-phase training protocol: A → B → C.

        Args:
            dataloader: DataLoader providing real sequences.
            epochs_a: Override epochs for Phase A.
            epochs_b: Override epochs for Phase B.
            epochs_c: Override epochs for Phase C.
            use_mlflow: Whether to log to MLflow.

        Returns:
            Dictionary with metrics from all phases.
        """
        train_cfg = self.config.get("training", {})
        ea = epochs_a or train_cfg.get("epochs_autoencoder", 200)
        eb = epochs_b or train_cfg.get("epochs_supervisor", 200)
        ec = epochs_c or train_cfg.get("epochs_joint", 500)

        logger.info(
            "training_start",
            epochs_a=ea,
            epochs_b=eb,
            epochs_c=ec,
        )

        total_start = time.time()
        all_metrics: Dict[str, Any] = {}

        # === Phase A ===
        if use_mlflow:
            mlflow.set_experiment(
                self.config.get("mlflow", {}).get(
                    "experiment_name", "transitmind-timegan-layer1"
                )
            )

        phase_a_metrics = self._run_phase_with_mlflow(
            "phase_a_autoencoder",
            self.phase_a_autoencoder,
            dataloader,
            ea,
            use_mlflow,
        )
        all_metrics["phase_a"] = phase_a_metrics

        # === Phase B ===
        phase_b_metrics = self._run_phase_with_mlflow(
            "phase_b_supervisor",
            self.phase_b_supervisor,
            dataloader,
            eb,
            use_mlflow,
        )
        all_metrics["phase_b"] = phase_b_metrics

        # === Phase C ===
        phase_c_metrics = self._run_phase_with_mlflow(
            "phase_c_joint_training",
            self.phase_c_joint,
            dataloader,
            ec,
            use_mlflow,
        )
        all_metrics["phase_c"] = phase_c_metrics

        # Save best model
        self._save_best_model()

        total_elapsed = time.time() - total_start
        all_metrics["total_duration_s"] = total_elapsed

        logger.info(
            "training_complete",
            total_duration=format_duration(total_elapsed),
        )

        return all_metrics

    def _run_phase_with_mlflow(
        self,
        phase_name: str,
        phase_fn,
        dataloader: DataLoader,
        n_epochs: int,
        use_mlflow: bool,
    ) -> Dict[str, float]:
        """Run a training phase with optional MLflow tracking."""
        if use_mlflow:
            try:
                with mlflow.start_run(run_name=phase_name, nested=True):
                    # Log hyperparameters
                    mlflow.log_params({
                        "phase": phase_name,
                        "n_epochs": n_epochs,
                        "lr_generator": self.lr_gen,
                        "lr_discriminator": self.lr_disc,
                        "gamma": self.gamma,
                        "hidden_dim": self.hidden_dim,
                        "seq_len": self.seq_len,
                        "device": str(self.device),
                    })

                    metrics = phase_fn(dataloader, n_epochs)

                    # Log final metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)

                    return metrics
            except Exception as e:
                logger.warning("mlflow_error", error=str(e))
                return phase_fn(dataloader, n_epochs)
        else:
            return phase_fn(dataloader, n_epochs)

    def save_checkpoint(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, float],
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            phase: Current training phase ('phase_a', 'phase_b', 'phase_c').
            metrics: Current metrics to save with checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "phase": phase,
            "metrics": metrics,
            "embedder_state": self.embedder.state_dict(),
            "recovery_state": self.recovery.state_dict(),
            "generator_state": self.generator.state_dict(),
            "supervisor_state": self.supervisor.state_dict(),
            "discriminator_state": self.discriminator.state_dict(),
            "opt_autoencoder_state": self.opt_autoencoder.state_dict(),
            "opt_supervisor_state": self.opt_supervisor.state_dict(),
            "opt_generator_state": self.opt_generator.state_dict(),
            "opt_discriminator_state": self.opt_discriminator.state_dict(),
            "history": self.history,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_{phase}_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        logger.debug("checkpoint_saved", path=str(path), epoch=epoch, phase=phase)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load a training checkpoint and restore state.

        Args:
            path: Path to the checkpoint file.

        Returns:
            Dictionary with checkpoint metadata (epoch, phase, metrics).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.embedder.load_state_dict(checkpoint["embedder_state"])
        self.recovery.load_state_dict(checkpoint["recovery_state"])
        self.generator.load_state_dict(checkpoint["generator_state"])
        self.supervisor.load_state_dict(checkpoint["supervisor_state"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state"])

        self.opt_autoencoder.load_state_dict(checkpoint["opt_autoencoder_state"])
        self.opt_supervisor.load_state_dict(checkpoint["opt_supervisor_state"])
        self.opt_generator.load_state_dict(checkpoint["opt_generator_state"])
        self.opt_discriminator.load_state_dict(checkpoint["opt_discriminator_state"])

        self.history = checkpoint.get("history", self.history)

        logger.info(
            "checkpoint_loaded",
            path=path,
            epoch=checkpoint["epoch"],
            phase=checkpoint["phase"],
        )

        return {
            "epoch": checkpoint["epoch"],
            "phase": checkpoint["phase"],
            "metrics": checkpoint["metrics"],
        }

    def _save_best_model(self) -> None:
        """Save the final trained model as the best model."""
        model_state = {
            "embedder": self.embedder.state_dict(),
            "recovery": self.recovery.state_dict(),
            "generator": self.generator.state_dict(),
            "supervisor": self.supervisor.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "config": self.config,
            "history": self.history,
        }

        path = self.best_model_dir / "timegan_best.pt"
        torch.save(model_state, path)
        logger.info("best_model_saved", path=str(path))


def create_dataloader(
    sequences: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from numpy sequences.

    Args:
        sequences: Array of shape (n_samples, seq_len, n_features).
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle the data.

    Returns:
        PyTorch DataLoader.
    """
    tensor = torch.FloatTensor(sequences)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
