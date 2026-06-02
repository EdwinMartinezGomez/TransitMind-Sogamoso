"""
TransitMind Sogamoso — Utility Functions
=========================================
General-purpose helper functions used across all layers.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root (where pyproject.toml lives).
    """
    current = Path(__file__).resolve()
    # Navigate up from src/shared/utils.py → src/shared → src → project root
    return current.parent.parent.parent


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file from the configs/ directory.

    Args:
        config_name: Name of the config file (e.g., 'timegan_config.yaml').

    Returns:
        Dictionary with parsed YAML contents.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = get_project_root() / "configs" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create.

    Returns:
        The Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path against the project root.

    Args:
        relative_path: Path relative to project root.

    Returns:
        Absolute Path object.
    """
    return get_project_root() / relative_path


def get_device() -> str:
    """
    Detect the best available device for PyTorch.

    Returns:
        One of: 'mps', 'cuda', or 'cpu'.
    """
    import torch

    # Prefer Apple's MPS backend on macOS Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"

    # Then CUDA if available
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and Python.

    Args:
        seed: Integer seed value.
    """
    import numpy as np
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA: set all devices
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Fail-safe: some builds/environments may not expose CUDA APIs
        pass

    # For reproducibility across backends, enable deterministic algorithms if requested
    # (left commented — enable only if needed and after testing performance)
    # torch.use_deterministic_algorithms(True)


def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the main TimeGAN configuration, optionally returning a specific section.

    Args:
        section: Optional section key (e.g., 'model', 'training', 'data').

    Returns:
        Full config dict or the specified section.
    """
    config = load_yaml_config("timegan_config.yaml")
    if section:
        return config.get(section, {})
    return config


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., '2h 15m 30s').
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def get_autocast(device: str):
    """
    Return an autocast context manager appropriate for the device.

    Usage:
        with get_autocast(device):
            out = model(x)
    """
    try:
        # torch.autocast accepts a device_type argument in newer PyTorch
        import torch
        return torch.autocast(device_type=device)
    except Exception:
        # Fallback: no-op context manager
        from contextlib import nullcontext

        return nullcontext()


class _NoOpGradScaler:
    """A thin no-op stand-in for torch.cuda.amp.GradScaler on unsupported devices."""

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def unscale_(self, optimizer):
        return

    def update(self):
        return


def get_grad_scaler(device: str):
    """
    Return a GradScaler instance when available for the device, otherwise a no-op scaler.
    """
    try:
        import torch
        if device == "cuda":
            return torch.cuda.amp.GradScaler()
    except Exception:
        pass

    return _NoOpGradScaler()
