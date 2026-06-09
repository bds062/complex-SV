"""
Shared utilities for the complex_sv project.

Includes checkpoint I/O, reproducibility helpers, logging setup, genomic
coordinate conversion, L2 normalisation, and percentile-threshold helpers.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOGGER_NAME = "complex_sv"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any],
    **extra_arrays: Any,
) -> None:
    """
    Save a model checkpoint as a .pt dictionary.

    The checkpoint always contains:
        - model_state_dict
        - config

    Additional keyword arguments are stored at the top level, for example
    scaler_center, scaler_scale, val_losses, train_losses, or novelty thresholds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    checkpoint.update(extra_arrays)
    torch.save(checkpoint, path)


def torch_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load trusted project checkpoints across PyTorch 2.6 weights_only default changes."""
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a .pt checkpoint into model and return the full checkpoint dict.

    Use the returned dict to access stored config values, scaler statistics,
    validation curves, or novelty thresholds.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch_load_checkpoint(path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint is missing 'model_state_dict': {path}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def offsets_to_coords(
    start_offset: float,
    end_offset: float,
    region_start_bp: int,
    region_end_bp: int,
) -> tuple[int, int]:
    """
    Convert fractional boundary offsets back to 0-based half-open coordinates.

    Offsets are clipped to [0, 1]. Returned coordinates are ordered so start is
    never greater than end, and both are clipped to the input candidate region.
    """
    if region_end_bp < region_start_bp:
        raise ValueError(
            "region_end_bp must be greater than or equal to region_start_bp "
            f"({region_start_bp=}, {region_end_bp=})"
        )

    start_offset = float(np.clip(start_offset, 0.0, 1.0))
    end_offset = float(np.clip(end_offset, 0.0, 1.0))

    span = int(region_end_bp) - int(region_start_bp)
    start_bp = int(round(region_start_bp + start_offset * span))
    end_bp = int(round(region_start_bp + end_offset * span))

    start_bp, end_bp = sorted((start_bp, end_bp))
    start_bp = max(int(region_start_bp), min(start_bp, int(region_end_bp)))
    end_bp = max(int(region_start_bp), min(end_bp, int(region_end_bp)))

    return start_bp, end_bp


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure project logging to stdout and optionally to a file.

    Repeated calls replace existing handlers on the project logger to prevent
    duplicate messages in notebooks, tests, or CLI re-entry.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "complex_sv.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_percentile_thresholds(
    errors: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Return a finite percentile threshold from an array of reconstruction errors."""
    arr = np.asarray(errors, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute a percentile threshold from an empty error array")
    return float(np.percentile(arr, percentile))


def l2_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    L2-normalise embeddings before storage, prototype operations, or distance use.
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance for already L2-normalised embeddings.

    Broadcasting follows torch.matmul semantics. For common cases:
        - a: [D], b: [D] returns scalar
        - a: [B, D], b: [C, D] returns [B, C]
    """
    if a.ndim == 1 and b.ndim == 1:
        return 1.0 - torch.dot(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return 1.0 - a @ b.T
    return 1.0 - torch.sum(a * b, dim=-1)


def reciprocal_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """
    Return the minimum reciprocal overlap between two half-open intervals.

    This is useful for candidate-region merging and prediction/ground-truth
    matching where both intervals must overlap substantially.
    """
    if end_a <= start_a or end_b <= start_b:
        return 0.0

    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    if inter == 0:
        return 0.0

    len_a = end_a - start_a
    len_b = end_b - start_b
    return float(min(inter / len_a, inter / len_b))


def interval_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """Return standard interval IoU for two 0-based half-open intervals."""
    if end_a <= start_a or end_b <= start_b:
        return 0.0

    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return float(inter / union) if union > 0 else 0.0
