"""
Phase 2 graph encoder pretraining entry point.

Trains the Severus heterogeneous graph masked autoencoder on fixed genomic bp
windows sampled from VCF-derived breakpoint graphs.  This mirrors the CN
encoder's bp-window workflow while preserving graph topology inside each region.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from torch_geometric.data import Batch
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "PyTorch Geometric is required for complex_sv.pretrain.pretrain_graph. "
        "Install torch-geometric and matching PyG wheels for your torch build."
    ) from exc

try:
    from config import GraphEncoderConfig
    from data.severus_parser import BINARY_COLS, CONTINUOUS_COLS, N_CONT, N_FEAT, build_node_features, parse_all_severus
    from data.sv_region_sampler import GRAPH_AUX_TARGET_NAMES, build_region_graphs, build_sv_bp_windows, window_metadata_frame
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, save_checkpoint, set_seed, setup_logging
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import GraphEncoderConfig
    from data.severus_parser import BINARY_COLS, CONTINUOUS_COLS, N_CONT, N_FEAT, build_node_features, parse_all_severus
    from data.sv_region_sampler import GRAPH_AUX_TARGET_NAMES, build_region_graphs, build_sv_bp_windows, window_metadata_frame
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, save_checkpoint, set_seed, setup_logging


LOG_NAME = "graph_pretrain.log"
log = logging.getLogger(__name__)


def _status(message: str) -> None:
    print(f"[pretrain_graph] {message}", flush=True)


def _setup_script_logging(output_dir: Path) -> logging.Logger:
    logger = setup_logging(output_dir)
    dedicated = output_dir / LOG_NAME
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(dedicated)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Writing dedicated graph pretraining log to %s", dedicated)
    return logger


def _resolve_input_paths(input_list: str) -> list[str]:
    list_path = Path(input_list)
    if not list_path.exists():
        raise FileNotFoundError(f"--input_list not found: {list_path}")

    paths: list[str] = []
    with list_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = line.strip()
            if item and not item.startswith("#"):
                paths.append(item)

    unique = list(dict.fromkeys(paths))
    valid = [p for p in unique if Path(p).exists()]
    if not valid:
        raise FileNotFoundError("--input_list did not contain any valid Severus VCF paths")
    missing = [p for p in unique if not Path(p).exists()]
    if missing:
        log.warning("Skipping %d missing VCF path(s) from input list", len(missing))
    return valid


class SVRegionGraphDataset(Dataset):
    """One item is one fixed-bp regional SV graph with a random node mask."""

    def __init__(self, graphs: list, mask_prob: float) -> None:
        if not graphs:
            raise ValueError("SVRegionGraphDataset requires at least one graph")
        self.graphs = graphs
        self.mask_prob = float(mask_prob)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        n_nodes = int(graph["sv"].x.shape[0])
        if n_nodes <= 0:
            raise ValueError("Regional graph has no SV nodes")
        mask = torch.rand(n_nodes) < self.mask_prob
        if not mask.any():
            mask[torch.randint(0, n_nodes, (1,)).item()] = True
        return graph, mask


def collate_region_graphs(batch):
    graphs, masks = zip(*batch)
    batched = Batch.from_data_list(list(graphs))
    return batched, torch.cat(list(masks), dim=0)


def _masked_node_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(dtype=recon.dtype)
    return ((recon - target) ** 2 * mask_f).sum() / (mask_f.sum() * target.shape[-1] + 1e-9)


def _aux_targets(data) -> torch.Tensor:
    if not hasattr(data, "graph_targets"):
        raise KeyError("Regional graphs are missing graph_targets; rebuild them with sv_region_sampler.")
    target = data.graph_targets
    if target.ndim == 1:
        target = target.view(-1, len(GRAPH_AUX_TARGET_NAMES))
    return target.to(dtype=torch.float32)


def _graph_aux_losses(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "density": F.mse_loss(pred[:, 0:2], target[:, 0:2]),
        "foldback": F.binary_cross_entropy_with_logits(pred[:, 2], target[:, 2])
        + F.mse_loss(torch.sigmoid(pred[:, 3]), target[:, 3]),
        "interchrom": F.binary_cross_entropy_with_logits(pred[:, 4], target[:, 4])
        + F.mse_loss(torch.sigmoid(pred[:, 5]), target[:, 5]),
    }


class WarmupCosineScheduler:
    """Epoch-level warmup followed by cosine decay."""

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


def _run_epoch(
    model: SVGraphMAE,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    density_loss_weight: float = 1.0,
    foldback_loss_weight: float = 0.5,
    interchrom_loss_weight: float = 0.5,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0
    for batch_graph, mask in tqdm(loader, desc="train" if training else "val", leave=False):
        batch_graph = batch_graph.to(device)
        mask = mask.to(device=device, dtype=torch.bool)
        target = batch_graph["sv"].x.to(device=device, dtype=torch.float32)

        if training:
            optimizer.zero_grad(set_to_none=True)

        recon, node_h = model(batch_graph, mask)
        recon_loss = _masked_node_mse(recon, target, mask)
        aux_pred = model.graph_aux(node_h, batch_graph["sv"].batch)
        aux_loss = _graph_aux_losses(aux_pred, _aux_targets(batch_graph).to(device=device))
        loss = (
            recon_loss
            + float(density_loss_weight) * aux_loss["density"]
            + float(foldback_loss_weight) * aux_loss["foldback"]
            + float(interchrom_loss_weight) * aux_loss["interchrom"]
        )

        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("DataLoader produced no batches")
    return total_loss / n_batches


def _extract_embeddings(
    model: SVGraphMAE,
    graphs: list,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Batch.from_data_list)
    all_embeds: list[np.ndarray] = []

    with torch.no_grad():
        for batch_graph in tqdm(loader, desc="embed", leave=False):
            batch_graph = batch_graph.to(device)
            mask = torch.zeros(batch_graph["sv"].x.shape[0], dtype=torch.bool, device=device)
            _recon, node_h = model(batch_graph, mask)
            batch_vec = batch_graph["sv"].batch
            batch_embeds: list[torch.Tensor] = []
            for graph_idx in range(int(batch_graph.num_graphs)):
                node_idx = torch.nonzero(batch_vec == graph_idx, as_tuple=False).flatten().tolist()
                if not node_idx:
                    continue
                batch_embeds.append(l2_normalize(model.regional_embed(node_h, node_idx), dim=-1))
            if batch_embeds:
                all_embeds.append(torch.stack(batch_embeds, dim=0).cpu().numpy())

    if not all_embeds:
        raise RuntimeError("No graph window embeddings extracted")
    return np.concatenate(all_embeds, axis=0)


def _build_training_graphs(
    input_paths: list[str],
    cfg: GraphEncoderConfig,
    window_bp_sizes: list[int],
    windows_per_chrom_per_size: int,
    cluster_windows_per_chrom_per_size: int,
    min_sv_per_window: int,
    max_windows: int | None,
    seed: int,
    include_mate_context: bool,
    max_mate_context_nodes: int | None,
) -> tuple[list, object, object, object]:
    _status(f"Parsing {len(input_paths):,} Severus VCF file(s)")
    df = parse_all_severus(input_paths)
    _status(
        f"Parsed {len(df):,} SV record(s) across "
        f"{df['sample_id'].nunique():,} sample(s)"
    )

    feat_matrix, scaler = build_node_features(df)
    n_chrom_groups = df.groupby(["sample_id", "chrom"], sort=False).ngroups
    upper_bound = n_chrom_groups * len(window_bp_sizes) * (
        int(windows_per_chrom_per_size) + int(cluster_windows_per_chrom_per_size)
    )
    _status(
        "Building SV bp windows: "
        f"sizes={window_bp_sizes}, random_per_chrom={windows_per_chrom_per_size}, "
        f"dense_per_chrom={cluster_windows_per_chrom_per_size}, "
        f"min_sv_per_window={min_sv_per_window}, upper_bound~{upper_bound:,}, "
        f"max_windows={max_windows}"
    )

    rng = np.random.default_rng(seed)
    windows = build_sv_bp_windows(
        df,
        window_bp_sizes=window_bp_sizes,
        windows_per_chrom_per_size=windows_per_chrom_per_size,
        cluster_windows_per_chrom_per_size=cluster_windows_per_chrom_per_size,
        min_sv_per_window=min_sv_per_window,
        rng=rng,
        max_windows=max_windows,
        progress=True,
    )
    if not windows:
        raise RuntimeError(
            "No SV bp windows were generated. Try smaller --window_bp_sizes "
            "or lower --min_sv_per_window."
        )
    _status(f"Built {len(windows):,} accepted SV bp window(s)")

    graphs = build_region_graphs(
        df,
        feat_matrix,
        windows,
        proximity_bp=cfg.proximity_bp,
        progress=True,
        include_mate_context=include_mate_context,
        max_mate_context_nodes=max_mate_context_nodes,
    )
    if not graphs:
        raise RuntimeError("No regional graph objects were built")
    _status(f"Built {len(graphs):,} regional graph object(s)")
    return graphs, windows, df, scaler


def train(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_script_logging(output_dir)
    logger.info("Starting graph encoder pretraining")
    _status("Starting graph encoder pretraining")

    cfg = GraphEncoderConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        proximity_bp=args.proximity_bp,
        mask_prob=args.mask_prob,
    )

    input_paths = _resolve_input_paths(args.input_list)
    _status(f"Loaded {len(input_paths):,} VCF path(s) from {args.input_list}")

    graphs, windows, _df, scaler = _build_training_graphs(
        input_paths=input_paths,
        cfg=cfg,
        window_bp_sizes=args.window_bp_sizes,
        windows_per_chrom_per_size=args.windows_per_chrom_per_size,
        cluster_windows_per_chrom_per_size=args.cluster_windows_per_chrom_per_size,
        min_sv_per_window=args.min_sv_per_window,
        max_windows=args.max_windows,
        seed=args.seed,
        include_mate_context=not args.no_mate_context,
        max_mate_context_nodes=args.max_mate_context_nodes,
    )

    dataset = SVRegionGraphDataset(graphs, mask_prob=cfg.mask_prob)
    val_size = max(1, int(round(args.val_fraction * len(dataset)))) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    _status(f"Train/val split: {train_size:,} train window(s), {val_size:,} val window(s)")

    if val_size > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_ds, val_ds = dataset, dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_region_graphs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_region_graphs,
    )

    device = get_device()
    logger.info("Device: %s", device)
    _status(f"Using device: {device}")

    model = SVGraphMAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Graph model parameters: %s", f"{n_params:,}")
    _status(f"Graph model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            density_loss_weight=args.density_loss_weight,
            foldback_loss_weight=args.foldback_loss_weight,
            interchrom_loss_weight=args.interchrom_loss_weight,
        )
        with torch.no_grad():
            val_loss = _run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                density_loss_weight=args.density_loss_weight,
                foldback_loss_weight=args.foldback_loss_weight,
                interchrom_loss_weight=args.interchrom_loss_weight,
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
        _status(
            f"Epoch {epoch:3d}/{args.epochs} train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={lr:.3e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    config_dict = {
        "graph_encoder": asdict(cfg),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "seed": args.seed,
            "window_bp_sizes": args.window_bp_sizes,
            "windows_per_chrom_per_size": args.windows_per_chrom_per_size,
            "cluster_windows_per_chrom_per_size": args.cluster_windows_per_chrom_per_size,
            "min_sv_per_window": args.min_sv_per_window,
            "max_windows": args.max_windows,
            "include_mate_context": not args.no_mate_context,
            "max_mate_context_nodes": args.max_mate_context_nodes,
            "density_loss_weight": args.density_loss_weight,
            "foldback_loss_weight": args.foldback_loss_weight,
            "interchrom_loss_weight": args.interchrom_loss_weight,
            "graph_aux_target_names": list(GRAPH_AUX_TARGET_NAMES),
        },
        "input_vcfs": input_paths,
        "severus_continuous_cols": list(CONTINUOUS_COLS),
        "severus_binary_cols": list(BINARY_COLS),
        "n_feat": N_FEAT,
    }

    save_checkpoint(
        output_dir / "graph_encoder.pt",
        model,
        config_dict,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val,
        scaler_center=scaler.center_,
        scaler_scale=scaler.scale_,
        scaler_n_features_in=getattr(scaler, "n_features_in_", N_CONT),
        n_cont=N_CONT,
    )
    logger.info("Saved graph checkpoint to %s", output_dir / "graph_encoder.pt")

    meta = window_metadata_frame(windows, include_node_indices=True)
    meta.to_csv(output_dir / "graph_bp_window_metadata.tsv", sep="\t", index=False)
    logger.info("Saved graph window metadata to %s", output_dir / "graph_bp_window_metadata.tsv")

    if args.save_embeddings:
        logger.info("Extracting graph window embeddings ...")
        embeddings = _extract_embeddings(model, graphs, device=device, batch_size=max(args.batch_size, 128))
        embed_meta = meta.reset_index(drop=True).copy()
        if len(embed_meta) != embeddings.shape[0]:
            raise RuntimeError(
                "Embedding count does not match window metadata count: "
                f"{embeddings.shape[0]} vs {len(embed_meta)}"
            )
        meta_arrays = {}
        for col in embed_meta.columns:
            values = embed_meta[col].values
            if values.dtype == object:
                values = values.astype(str)
            meta_arrays[col] = values
        np.savez(
            output_dir / "graph_window_embeddings.npz",
            embeddings=embeddings,
            train_losses=np.asarray(train_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
            **meta_arrays,
        )
        logger.info(
            "Saved graph window embeddings to %s shape=%s",
            output_dir / "graph_window_embeddings.npz",
            embeddings.shape,
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total pretraining loss")
    ax.set_title("Graph encoder pretraining")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "graph_training_loss.png", dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve to %s", output_dir / "graph_training_loss.png")
    logger.info("Done.")
    _status(f"Done. Dedicated log: {output_dir / LOG_NAME}")
    return train_losses, val_losses


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    defaults = GraphEncoderConfig()
    parser = argparse.ArgumentParser(
        description="Pretrain the Severus graph masked autoencoder on genomic bp windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_list",
        required=True,
        metavar="FILE",
        help="Text file with one Severus VCF path per line.",
    )
    parser.add_argument("--output_dir", default=".")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--mask_prob", type=float, default=defaults.mask_prob)
    parser.add_argument("--density_loss_weight", type=float, default=1.0)
    parser.add_argument("--foldback_loss_weight", type=float, default=0.5)
    parser.add_argument("--interchrom_loss_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Also save unmasked graph bp-window embeddings to graph_window_embeddings.npz.",
    )

    parser.add_argument("--d_model", type=int, default=defaults.d_model)
    parser.add_argument("--n_heads", type=int, default=defaults.n_heads)
    parser.add_argument("--n_layers", type=int, default=defaults.n_layers)
    parser.add_argument("--embed_dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--proximity_bp", type=int, default=defaults.proximity_bp)

    parser.add_argument(
        "--window_bp_sizes",
        type=int,
        nargs="+",
        default=[100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000],
        help="Fixed genomic bp window sizes used for graph pretraining.",
    )
    parser.add_argument(
        "--windows_per_chrom_per_size",
        type=int,
        default=20,
        help="Random bp windows sampled per sample/chromosome/window size.",
    )
    parser.add_argument(
        "--cluster_windows_per_chrom_per_size",
        type=int,
        default=20,
        help="Dense breakpoint windows retained per sample/chromosome/window size.",
    )
    parser.add_argument(
        "--min_sv_per_window",
        type=int,
        default=2,
        help="Minimum SV nodes required for a training window.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Optional cap on accepted graph windows before training.",
    )
    parser.add_argument(
        "--no_mate_context",
        action="store_true",
        help="Do not add explicit mate records outside the anchor bp window to regional graphs.",
    )
    parser.add_argument(
        "--max_mate_context_nodes",
        type=int,
        default=512,
        help="Maximum explicit mate nodes added outside each anchor bp window; use -1 for no cap.",
    )

    args = parser.parse_args(argv)
    if args.max_mate_context_nodes is not None and args.max_mate_context_nodes < 0:
        args.max_mate_context_nodes = None
    return args


if __name__ == "__main__":
    train(parse_args())
