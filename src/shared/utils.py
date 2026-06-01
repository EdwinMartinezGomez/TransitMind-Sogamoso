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
        'cuda' if GPU is available, otherwise 'cpu'.
    """
    import torch
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
