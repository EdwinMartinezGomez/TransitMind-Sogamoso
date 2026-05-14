"""
TransitMind Sogamoso — TimeGAN Model Architecture (Fase 1)
============================================================
Complete TimeGAN implementation in PyTorch with 5 components:
    1. Embedder  — Maps real space to latent space H
    2. Recovery  — Reconstructs real space from latent space
    3. Generator — Generates latent representations from noise
    4. Supervisor — Learns temporal dynamics in latent space
    5. Discriminator — Distinguishes real from synthetic latent sequences

Reference:
    Yoon, Jarrett, & van der Schaar (2019).
    "Time-series Generative Adversarial Networks." NeurIPS.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.shared.logger import setup_logger
from src.shared.utils import get_config

logger = setup_logger("timegan_model")


class Embedder(nn.Module):
    """
    Embedder network: maps real-space sequences to a latent space H.

    Architecture:
        - Multi-layer GRU
        - Linear projection to hidden_dim
        - LayerNorm for stable training

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_features: Number of input features.
            hidden_dim: Dimension of the latent space.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features).

        Returns:
            Latent representation H of shape (batch, seq_len, hidden_dim).
        """
        h, _ = self.gru(x)
        h = self.linear(h)
        h = self.layer_norm(h)
        h = self.activation(h)
        return h


class Recovery(nn.Module):
    """
    Recovery network: reconstructs real-space sequences from latent space H.

    Architecture:
        - Multi-layer GRU (inverse of Embedder)
        - Linear projection back to n_features

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, n_features)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_features: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of the latent space.
            n_features: Number of output features.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, n_features)
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Latent tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Reconstructed tensor of shape (batch, seq_len, n_features).
        """
        out, _ = self.gru(h)
        out = self.linear(out)
        out = self.activation(out)
        return out


class Generator(nn.Module):
    """
    Generator network: produces latent representations from random noise.

    Architecture:
        - Multi-layer GRU processing noise input
        - Linear projection to hidden_dim
        - Sigmoid activation

    Input:  (batch, seq_len, noise_dim)
    Output: (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        noise_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            noise_dim: Dimension of the noise input.
            hidden_dim: Dimension of the latent output.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Noise tensor of shape (batch, seq_len, noise_dim).

        Returns:
            Generated latent tensor of shape (batch, seq_len, hidden_dim).
        """
        out, _ = self.gru(z)
        out = self.linear(out)
        out = self.activation(out)
        return out


class Supervisor(nn.Module):
    """
    Supervisor network: learns temporal dynamics in the latent space.

    Captures the transition distribution p(h_{t+1} | h_t) so that
    generated sequences follow realistic temporal patterns.

    Architecture:
        - 2-layer GRU
        - Linear projection
        - Sigmoid activation

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of the latent space.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Latent tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Supervised latent tensor of shape (batch, seq_len, hidden_dim).
        """
        out, _ = self.gru(h)
        out = self.linear(out)
        out = self.activation(out)
        return out


class Discriminator(nn.Module):
    """
    Discriminator network: classifies sequences as real or synthetic.

    Architecture:
        - Bidirectional GRU for full temporal context
        - Linear(1) + Sigmoid for binary classification
        - Uses the last hidden state for classification

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, 1) — probability of being real
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Dimension of the latent space.
            num_layers: Number of GRU layers.
            dropout: Dropout rate between GRU layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Bidirectional doubles the output dim
        self.linear = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            h: Latent tensor of shape (batch, seq_len, hidden_dim).

        Returns:
            Logit tensor of shape (batch, 1). Raw logits (no sigmoid).
        """
        out, _ = self.gru(h)
        # Use the last timestep output
        last_output = out[:, -1, :]
        logit = self.linear(last_output)
        return logit


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_timegan(config: Optional[Dict] = None) -> Dict[str, nn.Module]:
    """
    Factory function to build all 5 TimeGAN components.

    Args:
        config: Configuration dictionary. If None, loads from timegan_config.yaml.
            Expected keys under 'model':
                - n_features (int): Number of input features
                - hidden_dim (int): Latent space dimension
                - noise_dim (int): Generator noise dimension
                - num_layers (int): GRU layers for Embedder/Recovery/Generator
                - dropout (float): Dropout rate

    Returns:
        Dictionary with keys: 'embedder', 'recovery', 'generator',
        'supervisor', 'discriminator', each containing an nn.Module.
    """
    if config is None:
        config = get_config()

    model_cfg = config.get("model", {})
    n_features = model_cfg.get("n_features", 11)
    hidden_dim = model_cfg.get("hidden_dim", 64)
    noise_dim = model_cfg.get("noise_dim", 64)
    num_layers = model_cfg.get("num_layers", 3)
    dropout = model_cfg.get("dropout", 0.1)

    embedder = Embedder(
        n_features=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    recovery = Recovery(
        hidden_dim=hidden_dim,
        n_features=n_features,
        num_layers=num_layers,
        dropout=dropout,
    )

    generator = Generator(
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    supervisor = Supervisor(
        hidden_dim=hidden_dim,
        num_layers=2,  # Supervisor uses 2 layers as specified
        dropout=dropout,
    )

    discriminator = Discriminator(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    components = {
        "embedder": embedder,
        "recovery": recovery,
        "generator": generator,
        "supervisor": supervisor,
        "discriminator": discriminator,
    }

    # Log model sizes
    total_params = 0
    for name, model in components.items():
        n_params = count_parameters(model)
        total_params += n_params
        logger.info("model_component", name=name, parameters=n_params)

    logger.info("timegan_built", total_parameters=total_params)

    return components
