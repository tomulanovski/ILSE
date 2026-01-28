#!/usr/bin/env python3
"""
STS GIN/MLP/Weighted training with precomputed embeddings.
Loads from HDF5 files created by precompute_sts.py for fast training.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import time

from .gnn_datasets import PrecomputedSTSGraphDataset, PrecomputedSTSTensorDataset, PrecomputedSTSLayerwiseDataset
from .gnn_models import LayerGINEncoder, PairCosineSimScore, MLPEncoder, LearnedWeightingEncoder, DeepSetEncoder, DWAttEncoder
from experiments.utils.gpu_tracking import GPUTracker, log_gpu_summary


def _cos_to_score(cos: torch.Tensor, min_score: float, max_score: float) -> torch.Tensor:
    """
    Map cosine similarity [-1, 1] to score range [min_score, max_score].

    For STSBenchmark (0-5): cos=-1 -> 0, cos=0 -> 2.5, cos=1 -> 5
    """
    # First map [-1, 1] to [0, 1]
    normalized = (cos + 1.0) * 0.5
    # Then scale to [min_score, max_score]
    return min_score + normalized * (max_score - min_score)


def train_epoch(model, train_loader, optimizer, criterion, device, min_score, max_score):
    model.train()
    total_loss = 0.0
    all_score_preds = []
    all_y = []
    total = 0

    for batch in train_loader:
        optimizer.zero_grad()
        a, b, y = batch
        a, b, y = a.to(device), b.to(device), y.to(device)
        cos_preds = model(a, b)
        score_preds = _cos_to_score(cos_preds, min_score, max_score)
        loss = criterion(score_preds, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total += bs
        total_loss += loss.item() * bs
        all_score_preds.append(score_preds.detach().cpu())
        all_y.append(y.detach().cpu())

    all_preds = torch.cat(all_score_preds).numpy()
    all_y_np = torch.cat(all_y).numpy()
    sp, _ = spearmanr(all_y_np, all_preds)
    if not np.isfinite(sp):
        sp = 0.0

    return total_loss / max(1, total), float(sp)


def validate_epoch(model, val_loader, criterion, device, min_score, max_score):
    model.eval()
    total_loss = 0.0
    total = 0
    all_score_preds = []
    all_y = []

    with torch.no_grad():
        for batch in val_loader:
            a, b, y = batch
            a, b, y = a.to(device), b.to(device), y.to(device)
            cos_preds = model(a, b)
            score_preds = _cos_to_score(cos_preds, min_score, max_score)
            loss = criterion(score_preds, y)
            bs = y.size(0)
            total += bs
            total_loss += loss.item() * bs
            all_score_preds.append(score_preds.detach().cpu())
            all_y.append(y.detach().cpu())

    all_preds = torch.cat(all_score_preds).numpy()
    all_y_np = torch.cat(all_y).numpy()
    sp, _ = spearmanr(all_y_np, all_preds)
    if not np.isfinite(sp):
        sp = 0.0

    return total_loss / max(1, total), float(sp)


def main():
    parser = argparse.ArgumentParser(description="STS training with precomputed embeddings")

    # Task and model info
    parser.add_argument("--task", type=str, required=True, help="STS task name (e.g., STSBenchmark)")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., Pythia, TinyLlama)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (e.g., 410m, 1.1B)")

    # Precomputed embeddings path
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory containing precomputed embeddings (e.g., precomputed_embeddings_sts/Pythia_410m_mean_pooling)")

    # Encoder choice
    parser.add_argument("--encoder", type=str, default="gin", choices=["gin", "mlp", "weighted", "deepset", "dwatt"])

    # GIN hyperparams
    parser.add_argument("--gin_hidden_dim", type=int, default=256)
    parser.add_argument("--gin_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gin_mlp_layers", type=int, default=1)
    parser.add_argument("--node_to_choose", type=str, default="mean")
    parser.add_argument("--graph_type", type=str, default="fully_connected",
                        choices=["linear", "fully_connected", "virtual_node", "cayley"])

    # MLP hyperparams
    parser.add_argument("--mlp_input", type=str, default="last", choices=["last", "mean", "flatten"])
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)

    # DeepSet hyperparams
    parser.add_argument("--deepset_hidden_dim", type=int, default=256)
    parser.add_argument("--deepset_pre_pooling_layers", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--deepset_post_pooling_layers", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--deepset_pooling_type", type=str, default="mean", choices=["mean", "sum"])

    # DWAtt hyperparams (Depth-Wise Attention from ElNokrashy et al. 2024)
    # Paper: https://arxiv.org/abs/2209.15168
    parser.add_argument("--dwatt_hidden_dim", type=int, default=None,
                        help="If provided, projects to this dim first (for fair comparison with GIN). "
                             "None (default) = paper-faithful, operates in layer_dim.")
    parser.add_argument("--dwatt_bottleneck_ratio", type=float, default=0.5,
                        help="γ in paper, ratio for MLP bottleneck (default: 0.5, paper Table 4)")
    parser.add_argument("--dwatt_pos_embed_dim", type=int, default=24,
                        help="d_pos in paper, dimension for positional embeddings (default: 24, paper Table 4)")

    # Linear mode (no ReLU activations)
    parser.add_argument("--use_linear", action="store_true",
                        help="Remove all ReLU activations for fair comparison with linear probing baselines")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Other
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Cayley graphs always use pool_real_nodes_only and train_eps
    if args.graph_type == "cayley":
        args.pool_real_nodes_only = True
        args.train_eps = True
    else:
        args.pool_real_nodes_only = False
        args.train_eps = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Encoder: {args.encoder} | Task: {args.task}")

    # Construct HDF5 path
    h5_path = Path(args.embeddings_dir) / f"{args.task}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Precomputed embeddings not found: {h5_path}")

    print(f"Loading precomputed embeddings from: {h5_path}")

    # Load datasets based on encoder type
    if args.encoder == "gin":
        train_dataset = PrecomputedSTSGraphDataset(
            h5_path=str(h5_path),
            split="train",
            graph_type=args.graph_type,
            keep_embedding_layer=args.pool_real_nodes_only  # Keep embedding layer when pooling real nodes only
        )
        val_dataset = PrecomputedSTSGraphDataset(
            h5_path=str(h5_path),
            split="validation",
            graph_type=args.graph_type,
            keep_embedding_layer=args.pool_real_nodes_only  # Keep embedding layer when pooling real nodes only
        )
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get dimensions
        num_layers, layer_dim = train_dataset.num_layers, train_dataset.layer_dim
        print(f"Layer dimensions: {num_layers} layers, {layer_dim} features per layer")

        # Create GIN encoder
        gin_encoder = LayerGINEncoder(
            in_dim=layer_dim,
            hidden_dim=args.gin_hidden_dim,
            num_layers=args.gin_layers,
            dropout=args.dropout,
            gin_mlp_layers=args.gin_mlp_layers,
            use_linear=args.use_linear,
            node_to_choose=args.node_to_choose,
            graph_type=args.graph_type,
            pool_real_nodes_only=args.pool_real_nodes_only,            train_eps=args.train_eps        )
        model = PairCosineSimScore(gin_encoder).to(device)

    elif args.encoder == "mlp":
        train_dataset = PrecomputedSTSTensorDataset(
            h5_path=str(h5_path),
            split="train",
            mode=args.mlp_input
        )
        val_dataset = PrecomputedSTSTensorDataset(
            h5_path=str(h5_path),
            split="validation",
            mode=args.mlp_input
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get input dimension
        in_dim = train_dataset.input_dim
        print(f"MLP input mode: {args.mlp_input}, Input dim: {in_dim}")

        # Create MLP encoder
        mlp_encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_layers=args.mlp_layers,
            dropout=args.dropout,
            use_linear=args.use_linear
        )
        model = PairCosineSimScore(mlp_encoder).to(device)

    elif args.encoder == "weighted":
        train_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="train"
        )
        val_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="validation"
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get dimensions
        num_layers, layer_dim = train_dataset.num_layers, train_dataset.layer_dim
        print(f"Layer dimensions: {num_layers} layers, {layer_dim} features per layer")

        # Create weighted encoder
        weighted_encoder = LearnedWeightingEncoder(num_layers=num_layers, layer_dim=layer_dim)
        model = PairCosineSimScore(weighted_encoder).to(device)

    elif args.encoder == "deepset":
        train_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="train"
        )
        val_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="validation"
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get dimensions
        num_layers, layer_dim = train_dataset.num_layers, train_dataset.layer_dim
        print(f"DeepSet config: pre_layers={args.deepset_pre_pooling_layers}, post_layers={args.deepset_post_pooling_layers}, pooling={args.deepset_pooling_type}")
        print(f"Layer dimensions: {num_layers} layers, {layer_dim} features per layer")

        # Create DeepSet encoder
        deepset_encoder = DeepSetEncoder(
            num_layers=num_layers,
            layer_dim=layer_dim,
            hidden_dim=args.deepset_hidden_dim,
            pre_pooling_layers=args.deepset_pre_pooling_layers,
            post_pooling_layers=args.deepset_post_pooling_layers,
            pooling_type=args.deepset_pooling_type,
            dropout=args.dropout,
            use_linear=args.use_linear
        )
        model = PairCosineSimScore(deepset_encoder).to(device)

    elif args.encoder == "dwatt":
        # DWAtt: Depth-Wise Attention (ElNokrashy et al. 2024)
        # Uses layerwise data like weighted/deepset
        train_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="train"
        )
        val_dataset = PrecomputedSTSLayerwiseDataset(
            h5_path=str(h5_path),
            split="validation"
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get dimensions
        num_layers, layer_dim = train_dataset.num_layers, train_dataset.layer_dim
        print(f"DWAtt config: bottleneck_ratio={args.dwatt_bottleneck_ratio}, pos_embed_dim={args.dwatt_pos_embed_dim}")
        print(f"Layer dimensions: {num_layers} layers, {layer_dim} features per layer")
        if args.dwatt_hidden_dim:
            print(f"Using input projection to {args.dwatt_hidden_dim} dims (controlled comparison mode)")
        else:
            print(f"Paper-faithful mode: operating in {layer_dim} dims (no input projection)")

        # Create DWAtt encoder
        dwatt_encoder = DWAttEncoder(
            num_layers=num_layers,
            layer_dim=layer_dim,
            hidden_dim=args.dwatt_hidden_dim,  # None = paper-faithful
            bottleneck_ratio=args.dwatt_bottleneck_ratio,
            pos_embed_dim=args.dwatt_pos_embed_dim,
            dropout=args.dropout
        )
        model = PairCosineSimScore(dwatt_encoder).to(device)

    else:
        raise ValueError(f"Unsupported encoder: {args.encoder}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Detect score range based on task
    if args.task == "SICK-R":
        min_score, max_score = 1.0, 5.0
    elif args.task == "STSBenchmark":
        min_score, max_score = 0.0, 5.0
    elif args.task == "BIOSSES":
        min_score, max_score = 0.0, 4.0
    else:
        # Default to STSBenchmark range
        min_score, max_score = 0.0, 5.0
    print(f"Score range: [{min_score}, {max_score}]")

    # Training setup
    best_val_spearman = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # History
    history = {
        "epoch": [], "train_loss": [], "train_spearman": [],
        "val_loss": [], "val_spearman": [], "lr": []
    }
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_epoch, best_path = 0, None

    # Start training
    print(f"\nStarting training for {args.epochs} epochs...\n" + "-"*60)

    # Initialize robust GPU tracker
    gpu_tracker = GPUTracker(device, require_gpu=False)
    gpu_tracker.start()

    for epoch in range(args.epochs):
        tr_loss, tr_spearman = train_epoch(model, train_loader, optimizer, criterion, device, min_score, max_score)
        va_loss, va_spearman = validate_epoch(model, val_loader, criterion, device, min_score, max_score)
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["train_loss"].append(tr_loss)
        history["train_spearman"].append(tr_spearman)
        history["val_loss"].append(va_loss)
        history["val_spearman"].append(va_spearman)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch+1:2d}: Train Loss {tr_loss:.4f} Spearman {tr_spearman:.4f} | "
              f"Val Loss {va_loss:.4f} Spearman {va_spearman:.4f} | LR {lr_now:.2e}")

        if va_spearman > best_val_spearman:
            best_val_spearman, best_epoch = va_spearman, epoch+1

            # Determine method name and config suffix (matches classification naming)
            linear_prefix = "linear_" if args.use_linear else ""
            if args.encoder == "gin":
                base_method = "gcn" if args.gin_mlp_layers == 0 else "gin"
                method = f"{linear_prefix}{base_method}"
                config_suffix = f"{args.graph_type}_{args.node_to_choose}"
            elif args.encoder == "mlp":
                method = f"{linear_prefix}mlp"
                config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
            elif args.encoder == "weighted":
                method = "weighted"  # weighted is already linear, no prefix needed
                config_suffix = "softmax"
            elif args.encoder == "deepset":
                method = f"{linear_prefix}deepset"
                config_suffix = f"{args.deepset_pooling_type}_pre{args.deepset_pre_pooling_layers}_post{args.deepset_post_pooling_layers}"
            elif args.encoder == "dwatt":
                method = "dwatt"  # DWAtt paper-faithful
                # Config: paper-faithful (no projection) or with hidden_dim
                if args.dwatt_hidden_dim:
                    config_suffix = f"proj{args.dwatt_hidden_dim}"
                else:
                    config_suffix = "paper"  # Paper-faithful (no projection)
            else:
                method = f"{linear_prefix}{args.encoder}"
                config_suffix = "unknown"

            # Match classification naming: {method}_{task}_{model_family}_{model_size}_{config}.pt
            best_path = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}.pt"

            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_spearman": tr_spearman,
                "val_spearman": va_spearman,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "args": vars(args),
            }, best_path)
            print(f"★ New best model saved! Val Spearman: {va_spearman:.4f}")

    # Get runtime and GPU memory metrics
    metrics = gpu_tracker.stop()

    # Update best checkpoint with training metrics
    if best_path and best_path.exists():
        checkpoint = torch.load(best_path)
        checkpoint["train_time_sec"] = metrics["runtime_sec"]
        checkpoint["peak_memory_mb"] = metrics["peak_memory_mb"]
        checkpoint["peak_memory_gb"] = metrics["peak_memory_gb"]
        torch.save(checkpoint, best_path)

    print("-"*60)
    print(f"Done. Best Val Spearman: {best_val_spearman:.4f} @ epoch {best_epoch}")
    if best_path:
        print(f"Saved: {best_path}")

    # Log GPU summary
    log_gpu_summary(metrics, prefix="Training ")

    # Determine method name and config suffix for history/plot filenames
    if args.encoder == "gin":
        method = "gcn" if args.gin_mlp_layers == 0 else "gin"
        config_suffix = args.graph_type
    elif args.encoder == "mlp":
        method = "mlp"
        config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
    elif args.encoder == "weighted":
        method = "weighted"
        config_suffix = "softmax"
    else:
        method = args.encoder
        config_suffix = "unknown"

    # Save history
    import csv
    hist_path = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(history.keys())
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"Saved history CSV: {hist_path}")

    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch — {args.task} ({method})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_png = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_loss.png"
    plt.savefig(loss_png, dpi=160)
    plt.close()
    print(f"Saved loss plot: {loss_png}")

    # Spearman plot
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_spearman"], label="Train Spearman")
    plt.plot(history["epoch"], history["val_spearman"], label="Val Spearman")
    plt.xlabel("Epoch")
    plt.ylabel("Spearman Correlation")
    plt.title(f"Spearman vs Epoch — {args.task} ({method})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    sp_png = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_spearman.png"
    plt.savefig(sp_png, dpi=160)
    plt.close()
    print(f"Saved Spearman plot: {sp_png}")


if __name__ == "__main__":
    main()
