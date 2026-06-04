"""
Phase 1 CN encoder pretraining entry point.

Trains a BERT-style masked copy-number autoencoder on unlabeled Wakhan CNV
calls. Parsing is delegated to data.wakhan_parser, and arm-level resampling is
delegated to data.cn_resampler.

Compared with the older version, this script follows the working MAE pattern:

    dataset returns x_masked, x_target, mask
    loss is computed only at masked bins
    training uses one shared feature space across all samples
    checkpoint also stores arm metadata and extracted embeddings when possible
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from config import CNEncoderConfig
    from data.cn_resampler import build_bp_window_tensors
    from data.wakhan_parser import parse_all_wakhan
    from utils import get_device, save_checkpoint, set_seed, setup_logging
    from pretrain.cn_encoder import CNMaskedAutoencoder
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import CNEncoderConfig
    from data.cn_resampler import build_bp_window_tensors
    from data.wakhan_parser import parse_all_wakhan
    from utils import get_device, save_checkpoint, set_seed, setup_logging
    from pretrain.cn_encoder import CNMaskedAutoencoder

log = logging.getLogger(__name__)


def _resolve_inputs(
    input_patterns: list[str] | None,
    input_list: str | None,
) -> list[str]:
    """
    Collect Wakhan paths from --input and/or --input_list.
    """
    paths: list[str] = []

    if input_patterns:
        for pattern in input_patterns:
            expanded = glob.glob(pattern)
            if expanded:
                paths.extend(expanded)
            else:
                paths.append(pattern)

    if input_list:
        list_path = Path(input_list)
        if not list_path.exists():
            raise FileNotFoundError(f"--input_list not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = line.strip()
                if item and not item.startswith("#"):
                    paths.append(item)

    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)

    missing = [p for p in unique if not Path(p).exists()]
    if missing:
        log.warning("%d input path(s) do not exist; examples: %s", len(missing), missing[:5])

    valid = [p for p in unique if Path(p).exists()]
    if not valid:
        raise FileNotFoundError("No valid Wakhan inputs found. Use --input or --input_list.")

    return valid


class CNWindowDataset(Dataset):
    """
    One item is one bp-window resampled to [n_bins, 5].
    """

    def __init__(self, tensors: list[torch.Tensor], mask_prob: float) -> None:
        if not tensors:
            raise ValueError("CNWindowDataset requires at least one tensor")
        self.tensors = [x.to(dtype=torch.float32) for x in tensors]
        self.mask_prob = float(mask_prob)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_target = self.tensors[idx].clone()

        mask = torch.rand(x_target.shape[0]) < self.mask_prob
        if not mask.any():
            mask[torch.randint(0, x_target.shape[0], (1,)).item()] = True

        x_masked = x_target.clone()
        x_masked[mask] = 0.0

        return x_masked, x_target, mask


def _build_training_tensors(
    wakhan_paths: list[str],
    window_bp_sizes: list[int],
    n_bins: int,
    windows_per_chrom_per_size: int,
    min_covered_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    """
    Parse Wakhan inputs and build fixed-bp genomic windows.

    Each training example is a genomic interval of size window_bp_size, resampled
    to [n_bins, 5]. Long CN segments therefore occupy proportionally more bins.
    """
    df = parse_all_wakhan(wakhan_paths)
    rng = np.random.default_rng(seed)

    tensors, meta = build_bp_window_tensors(
        df,
        window_bp_sizes=window_bp_sizes,
        n_bins=n_bins,
        windows_per_chrom_per_size=windows_per_chrom_per_size,
        min_covered_fraction=min_covered_fraction,
        rng=rng,
    )

    if not tensors:
        raise RuntimeError(
            "No bp-window CN tensors were generated. "
            "Try smaller --window_bp_sizes or lower --min_covered_fraction."
        )

    log.info(
        "Built %d bp-window CN tensor(s) across %d sample(s).",
        len(tensors),
        meta["sample_id"].nunique() if not meta.empty else 0,
    )

    return tensors, meta


def _masked_cn_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked reconstruction loss over all CN channels/features.
    """
    mask_f = mask.unsqueeze(-1).to(dtype=recon.dtype)
    return ((recon - target) ** 2 * mask_f).sum() / (
        mask_f.sum() * target.shape[-1] + 1e-9
    )


def _make_encoder_cfg(base: CNEncoderConfig) -> SimpleNamespace:
    """
    Provide project-spec names plus legacy prototype names.
    """
    return SimpleNamespace(
        d_model=base.d_model,
        n_heads=base.n_heads,
        n_layers=base.n_layers,
        ff_dim=base.ff_dim,
        d_ff=base.ff_dim,
        dropout=base.dropout,
        n_bins_arm=base.n_bins_arm,
        n_bins_region=base.n_bins_region,
        seq_len=base.n_bins_arm,
        mask_prob=base.mask_prob,
        embed_dim=base.d_model,
    )


def _forward_cn(
    model: nn.Module,
    x_masked: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Normalize CN autoencoder forward outputs across compatible implementations.

    Preferred call:
        model(x_masked, mask)

    Supported returns:
        recon, cls_emb
        recon, cls_emb, bin_embs
    """
    out = model(x_masked, mask)

    if not isinstance(out, tuple):
        raise TypeError("CNMaskedAutoencoder.forward must return a tuple")

    if len(out) == 3:
        recon, cls_emb, bin_embs = out
    elif len(out) == 2:
        recon, cls_emb = out
        bin_embs = None
    else:
        raise ValueError(f"Unexpected CNMaskedAutoencoder output length: {len(out)}")

    return recon, cls_emb, bin_embs


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0

    for x_masked, x_target, mask in tqdm(
        loader,
        desc="train" if training else "val",
        leave=False,
    ):
        x_masked = x_masked.to(device=device, dtype=torch.float32)
        x_target = x_target.to(device=device, dtype=torch.float32)
        mask = mask.to(device=device, dtype=torch.bool)

        if training:
            optimizer.zero_grad(set_to_none=True)

        recon, _cls_emb, _bin_embs = _forward_cn(model, x_masked, mask)
        loss = _masked_cn_mse(recon, x_target, mask)

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("DataLoader produced no batches")

    return total_loss / n_batches


class WarmupCosineScheduler:
    """
    Epoch-level warmup followed by cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        epochs: int,
        warmup_epochs: int = 5,
        min_lr_fraction: float = 0.01,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.epochs = max(int(epochs), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.min_lr_fraction = float(min_lr_fraction)

    def step(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            denom = max(self.epochs - self.warmup_epochs, 1)
            progress = min(max(epoch - self.warmup_epochs, 0), denom) / denom
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_lr = self.base_lr * self.min_lr_fraction
            lr = min_lr + (self.base_lr - min_lr) * cosine

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        return lr


def _extract_embeddings(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Extract CLS embeddings using unmasked inputs.
    """
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeds: list[np.ndarray] = []

    with torch.no_grad():
        for _x_masked, x_target, _mask in tqdm(loader, desc="embed", leave=False):
            x_target = x_target.to(device=device, dtype=torch.float32)
            no_mask = torch.zeros(
                x_target.shape[:2],
                dtype=torch.bool,
                device=device,
            )

            _recon, cls_emb, _bin_embs = _forward_cn(model, x_target, no_mask)
            all_embeds.append(cls_emb.detach().cpu().numpy())

    if not all_embeds:
        raise RuntimeError("No embeddings extracted")

    return np.concatenate(all_embeds, axis=0)


def train(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    logger.info("Starting CN encoder pretraining")

    cfg = CNEncoderConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        n_bins_arm=args.n_bins_arm,
        n_bins_region=args.n_bins_region,
        mask_prob=args.mask_prob,
    )

    inputs = _resolve_inputs(args.input, args.input_list)

    window_tensors, window_meta = _build_training_tensors(
        wakhan_paths=inputs,
        window_bp_sizes=args.window_bp_sizes,
        n_bins=cfg.n_bins_region,
        windows_per_chrom_per_size=args.windows_per_chrom_per_size,
        min_covered_fraction=args.min_covered_fraction,
        seed=args.seed,
    )

    dataset = CNWindowDataset(window_tensors, mask_prob=cfg.mask_prob)

    val_size = max(1, int(round(args.val_fraction * len(dataset)))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size

    if val_size > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = get_device()
    logger.info("Device: %s", device)

    model = CNMaskedAutoencoder(_make_encoder_cfg(cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CN model parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=args.lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        lr = scheduler.step(epoch)

        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss = _run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
            )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            "Epoch %3d/%d  train=%.6f  val=%.6f  lr=%.3e",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            lr,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "cn_encoder": asdict(cfg),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
            "window_bp_sizes": args.window_bp_sizes,
            "windows_per_chrom_per_size": args.windows_per_chrom_per_size,
            "min_covered_fraction": args.min_covered_fraction,
        },
        "input_files": inputs,
    }

    save_checkpoint(
        output_dir / "cn_encoder.pt",
        model,
        config_dict,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val,
    )
    logger.info("Saved CN checkpoint to %s", output_dir / "cn_encoder.pt")

    window_meta.to_csv(output_dir / "cn_bp_window_metadata.tsv", sep="\t", index=False)
    logger.info("Saved bp-window metadata to %s", output_dir / "cn_bp_window_metadata.tsv")

    if args.save_embeddings:
        logger.info("Extracting arm embeddings ...")
        embeddings = _extract_embeddings(
            model,
            dataset,
            device,
            batch_size=max(args.batch_size, 128),
        )

        np.savez(
            output_dir / "cn_arm_embeddings.npz",
            embeddings=embeddings,
            sample_id=arm_meta["sample_id"].values.astype(str),
            chrom=arm_meta["chrom"].values.astype(str),
            arm=arm_meta["arm"].values.astype(str),
            arm_start=arm_meta["arm_start"].values,
            arm_end=arm_meta["arm_end"].values,
            n_segments=arm_meta["n_segments"].values,
            train_losses=np.asarray(train_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
        )

        logger.info(
            "Saved arm embeddings to %s shape=%s",
            output_dir / "cn_arm_embeddings.npz",
            embeddings.shape,
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked CN MSE")
    ax.set_title("CN encoder pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cn_training_loss.png", dpi=150)
    plt.close(fig)

    logger.info("Saved loss curve to %s", output_dir / "cn_training_loss.png")
    logger.info("Done.")

    return train_losses, val_losses


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    defaults = CNEncoderConfig()

    parser = argparse.ArgumentParser(
        description="Pretrain the masked CN Transformer encoder on Wakhan outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inp = parser.add_argument_group("Input")
    inp.add_argument("--input", nargs="+", metavar="WAKHAN")
    inp.add_argument("--input_list", metavar="FILE")

    parser.add_argument("--output_dir", default=".")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--mask_prob", type=float, default=defaults.mask_prob)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Also save unmasked arm-level CLS embeddings to cn_arm_embeddings.npz.",
    )

    parser.add_argument("--d_model", type=int, default=defaults.d_model)
    parser.add_argument("--n_heads", type=int, default=defaults.n_heads)
    parser.add_argument("--n_layers", type=int, default=defaults.n_layers)
    parser.add_argument("--ff_dim", type=int, default=defaults.ff_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--n_bins_arm", type=int, default=defaults.n_bins_arm)
    parser.add_argument("--n_bins_region", type=int, default=defaults.n_bins_region)

    parser.add_argument(
        "--window_bp_sizes",
        type=int,
        nargs="+",
        default=[100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000],
        help="Fixed genomic bp window sizes used for CN pretraining.",
    )
    parser.add_argument(
        "--windows_per_chrom_per_size",
        type=int,
        default=40,
        help="Random bp windows sampled per sample/chromosome/window size.",
    )
    parser.add_argument(
        "--min_covered_fraction",
        type=float,
        default=0.10,
        help="Minimum fraction of a bp window overlapped by observed CN segments.",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())