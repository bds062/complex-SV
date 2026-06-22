"""
Central configuration for the complex_sv project.

This module is intentionally dependency-light and should be imported by every
other project module that needs defaults. Do not duplicate configurable numeric
defaults elsewhere in the codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CNEncoderConfig:
    """Configuration for Stage 2a: masked copy-number Transformer."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    n_bins_arm: int = 256
    n_bins_region: int = 128
    mask_prob: float = 0.15


@dataclass(slots=True)
class GraphEncoderConfig:
    """Configuration for Stage 2b: heterogeneous graph masked autoencoder."""

    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    embed_dim: int = 64
    dropout: float = 0.1
    proximity_bp: int = 1_000_000
    mask_prob: float = 0.15
    edge_attr_dim: int = 3


@dataclass(slots=True)
class FusionConfig:
    """Configuration for Stage 4: multimodal fusion Transformer."""

    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dropout: float = 0.2

    # Projection input dimensions for the four fusion streams.
    cn_embed_dim: int = 256
    graph_embed_dim: int = 64
    graph_global_dim: int = 64
    graph_stats_dim: int = 22
    segment_stats_dim: int = 18


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for supervised Phase 3 and Phase 4 training."""

    phase3_lr: float = 1e-3
    phase4_lr_encoder_cn: float = 1e-5
    phase4_lr_encoder_graph: float = 5e-6
    phase4_lr_other: float = 1e-4
    phase3_epochs: int = 100
    phase4_epochs: int = 30
    batch_size: int = 32
    weight_decay: float = 0.01
    early_stop_patience: int = 10


@dataclass(slots=True)
class EpisodeConfig:
    """Configuration for N-way K-shot episodic training."""

    n_way: int = 2
    k_shot: int = 5
    n_query: int = 10


@dataclass(slots=True)
class InferenceConfig:
    """Configuration for prototype classification and novelty scoring."""

    tau: float = 0.5
    novelty_cn_p95: float | None = None
    novelty_graph_p95: float | None = None


@dataclass(slots=True)
class RegionProposalConfig:
    """Configuration for Stage 3 candidate-region proposal."""

    min_segments: int = 6
    max_cn_states: int = 3
    cn_tolerance: float = 0.3
    min_breakpoints_per_10mb: float = 5.0
    min_loh_or_imbalance: float = 0.4
    graph_proximity_bp: int = 500_000
    min_junctions: int = 3
    min_span_bp: int = 1_000_000
    overlap_threshold: float = 0.5


@dataclass(slots=True)
class SVConfig:
    """Top-level project configuration containing every stage sub-config."""

    cn_encoder: CNEncoderConfig = field(default_factory=CNEncoderConfig)
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    region_proposal: RegionProposalConfig = field(default_factory=RegionProposalConfig)


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any]) -> Any:
    """Recursively construct a dataclass instance from a JSON-loaded dict."""
    field_types = {name: field_def.type for name, field_def in cls.__dataclass_fields__.items()}
    kwargs: dict[str, Any] = {}

    for name, field_def in cls.__dataclass_fields__.items():
        if name not in values:
            continue

        default_value = getattr(cls(), name) if cls is SVConfig else None
        raw_value = values[name]

        # For the top-level config, use the concrete default instance type to
        # avoid relying on postponed annotation strings at runtime.
        if cls is SVConfig and is_dataclass(default_value) and isinstance(raw_value, dict):
            kwargs[name] = _dataclass_from_dict(type(default_value), raw_value)
        else:
            kwargs[name] = raw_value

    return cls(**kwargs)


def load_config(path: str | Path) -> SVConfig:
    """
    Load an SVConfig from a JSON file.

    Missing fields are filled from dataclass defaults so older config files can
    still be loaded after new options are added.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"Config JSON must contain an object at top level: {path}")

    return _dataclass_from_dict(SVConfig, data)


def save_config(cfg: SVConfig, path: str | Path) -> None:
    """Save an SVConfig as human-readable JSON."""
    if not is_dataclass(cfg):
        raise TypeError("save_config expects an SVConfig dataclass instance")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(cfg), fh, indent=2, sort_keys=True)
        fh.write("\n")
