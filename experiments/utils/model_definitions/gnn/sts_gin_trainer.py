#!/usr/bin/env python3
"""
Basic GIN implementation for layer-wise embeddings.
Simple, clean, and focused on the essentials.
"""
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr


from ..text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from .gnn_datasets import load_task_data, compute_layerwise, PairGraphRegressionDataset, PairTensorRegressionDataset
from .gnn_models import LayerGINEncoder, PairCosineSimScore, MLPEncoder


def _cos_to_01(cos: torch.Tensor) -> torch.Tensor:
    """DEPRECATED: Maps [-1, 1] -> [0, 1]. Use _cos_to_score instead."""
    # map [-1, 1] -> [0, 1]
    return (cos + 1.0) * 0.5

def _cos_to_score(cos: torch.Tensor, min_score: float, max_score: float) -> torch.Tensor:
    """
    Map cosine similarity [-1, 1] to score range [min_score, max_score].

    Args:
        cos: Cosine similarity in [-1, 1]
        min_score: Minimum score in dataset (e.g., 0 for STSBenchmark)
        max_score: Maximum score in dataset (e.g., 5 for STSBenchmark)

    Returns:
        Predicted scores in [min_score, max_score]

    Example:
        cos=-1 -> min_score, cos=1 -> max_score
        For STSBenchmark (0-5): cos=-1 -> 0, cos=0 -> 2.5, cos=1 -> 5
    """
    # First map [-1, 1] to [0, 1]
    normalized = (cos + 1.0) * 0.5
    # Then scale to [min_score, max_score]
    return min_score + normalized * (max_score - min_score)


def train_epoch(model, train_loader, optimizer, criterion, device, mode, min_score, max_score):
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
        # Map cosine similarity to score range
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
    if not np.isfinite(sp):  # handle degenerate constant vectors
        sp = 0.0

    return total_loss / max(1, total), float(sp)

def validate_epoch(model, val_loader, criterion, device, mode, min_score, max_score):
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
            # Map cosine similarity to score range
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
    parser = argparse.ArgumentParser(description="STS GIN/MLP Training on layer embeddings")
    # Task and model
    parser.add_argument("--task", type=str, default="SICK-R")
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="14m")
    # Encoder choice
    parser.add_argument("--encoder", type=str, default="gin", choices=["gin", "mlp", "weighted"])
    # GIN hyperparams
    parser.add_argument("--gin_hidden_dim", type=int, default=256)
    parser.add_argument("--gin_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gin_mlp_layers", type=int, default=1)
    parser.add_argument("--node_to_choose", type=str, default="mean")
    parser.add_argument("--graph_type", type=str, default="fully_connected",
                        choices=["linear", "fully_connected", "virtual_node", "cayley", "cayley"])
    parser.add_argument("--pool_real_nodes_only", action="store_true",
                        help="For cayley: exclude virtual nodes from pooling (only pool real layer nodes)")
    parser.add_argument("--train_eps", action="store_true",
                        help="Learnable epsilon in GIN aggregation (adds 1 param per GIN layer)")
    # MLP hyperparams
    parser.add_argument("--mlp_input", type=str, default="last", choices=["last", "mean", "flatten"],
                        help="What the MLP consumes: last layer (D), mean over layers (D), or flattened (L*D).")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)
    # Linear mode (no ReLU activations)
    parser.add_argument("--use_linear", action="store_true",
                        help="Remove all ReLU activations for fair comparison with linear probing baselines")
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # Other
    parser.add_argument("--save_dir", type=str, default="./basic_gin_models")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Encoder: {args.encoder} | Task: {args.task}")

    # Load data
    train_data = load_task_data(args.task, "train")
    try:
        val_data = load_task_data(args.task, "validation")
    except Exception:
        raise KeyError(f"No validation data for task: {args.task}")

    # Wrapper for layerwise
    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)

    print("Extracting layer-wise embeddings...")
    train_texts_a, train_texts_b = train_data["text_a"], train_data["text_b"]    
    train_lw_a = compute_layerwise(wrapper, train_texts_a, batch_size=256, token_pooling_method="mean")
    train_lw_b = compute_layerwise(wrapper, train_texts_b, batch_size=256, token_pooling_method="mean")

    val_texts_a, val_texts_b = val_data["text_a"], val_data["text_b"]    
    val_lw_a = compute_layerwise(wrapper, val_texts_a, batch_size=256, token_pooling_method="mean")
    val_lw_b = compute_layerwise(wrapper, val_texts_b, batch_size=256, token_pooling_method="mean")

    if args.encoder == "gin":
        train_dataset = PairGraphRegressionDataset(train_lw_a, train_lw_b, train_data["original_scores"], graph_type=args.graph_type) # Use original scores for training
        val_dataset = PairGraphRegressionDataset(val_lw_a, val_lw_b, val_data["original_scores"], graph_type=args.graph_type)
        train_loader  = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = PyGDataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        L, D = train_lw_a[0].shape
        print(f"Layer dimensions: {L} layers, {D} features per layer")
        
        gin_encoder = LayerGINEncoder(in_dim=D, hidden_dim=args.gin_hidden_dim,
                                      num_layers=args.gin_layers, dropout=args.dropout,
                                      gin_mlp_layers=args.gin_mlp_layers, node_to_choose=args.node_to_choose,
                                      graph_type=args.graph_type, use_linear=args.use_linear,
                                      pool_real_nodes_only=args.pool_real_nodes_only,
                                      train_eps=args.train_eps)
        model = PairCosineSimScore(gin_encoder).to(device)
        train_mode = "gin"

    elif args.encoder == "mlp":  # MLP baseline (no graph)
        # Build plain tensor datasets
        # Decide input dimension by mode
        L, D = train_lw_a[0].shape
        if args.mlp_input in ("last", "mean"):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        train_dataset = PairTensorRegressionDataset(train_lw_a, train_lw_b, train_data["original_scores"], mode=args.mlp_input) # Use original scores for training
        val_dataset = PairTensorRegressionDataset(val_lw_a, val_lw_b, val_data["original_scores"], mode=args.mlp_input)
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        mlp_encoder = MLPEncoder(in_dim=in_dim, hidden_dim=args.mlp_hidden_dim,
                                 num_layers=args.mlp_layers, dropout=args.dropout,
                                 use_linear=args.use_linear)
        model = PairCosineSimScore(mlp_encoder).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":  # Weighted baseline (learned layer weighting)
        from .gnn_models import LearnedWeightingEncoder
        from .gnn_datasets import PairLayerwiseRegressionDataset

        L, D = train_lw_a[0].shape
        print(f"Layer dimensions: {L} layers, {D} features per layer")

        # Create datasets that keep full layerwise info
        train_dataset = PairLayerwiseRegressionDataset(train_lw_a, train_lw_b, train_data["original_scores"])
        val_dataset = PairLayerwiseRegressionDataset(val_lw_a, val_lw_b, val_data["original_scores"])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Create weighted encoder
        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = PairCosineSimScore(weighted_encoder).to(device)
        train_mode = "weighted"

    else:
        raise ValueError(f"Unsupported encoder: {args.encoder}")

    # if train_mode == "gin":
    #     print("---------------------------")
    #     ys = []
    #     for i in range(len(train_dataset)):
    #         ys.append(int(train_dataset[i].y))
    #     ys = torch.tensor(ys)
    #     print("Class distribution (train):", torch.bincount(ys))
    #     print("Majority baseline (train):", torch.bincount(ys).max().item() / len(ys))
    #     print("---------------------------")

    #     ys_val = []
    #     for i in range(len(val_dataset)):
    #         ys_val.append(int(val_dataset[i].y))
    #     ys_val = torch.tensor(ys_val)
    #     print("Class distribution (validation):", torch.bincount(ys_val))
    #     print("Majority baseline (validation):", torch.bincount(ys_val).max().item() / len(ys_val))
    #     print("---------------------------")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get score range from training data (pre-detected by load_task_data)
    min_score = train_data["min_score"]
    max_score = train_data["max_score"]
    print(f"Score range: [{min_score}, {max_score}]")

    best_val_spearman = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # History & training (unchanged, but call with train_mode)
    history = {"epoch": [], "train_loss": [], "train_spearman": [], "val_loss": [], "val_spearman": [], "lr": []}
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_epoch, best_path = 0, None

    # Track GPU memory and training time
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    print(f"\nStarting training for {args.epochs} epochs...\n" + "-"*60)
    for epoch in range(args.epochs):
        tr_loss, tr_spearman = train_epoch(model, train_loader, optimizer, criterion, device, train_mode, min_score, max_score)
        va_loss, va_spearman = validate_epoch(model, val_loader, criterion, device, train_mode, min_score, max_score)
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["train_loss"].append(tr_loss); history["train_spearman"].append(tr_spearman)
        history["val_loss"].append(va_loss);   history["val_spearman"].append(va_spearman)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch+1:2d}: Train Loss {tr_loss:.4f} spearman {tr_spearman:.4f} | "
              f"Val Loss {va_loss:.4f} spearman {va_spearman:.4f} | LR {lr_now:.2e}")

        if va_spearman > best_val_spearman:
            best_val_spearman, best_epoch = va_spearman, epoch+1

            # Determine method and config_suffix (match classification naming pattern)
            if args.encoder == "gin":
                method = "gin" if args.gin_mlp_layers > 0 else "gcn"
                config_suffix = args.graph_type  # e.g., "cayley", "linear", etc.
            elif args.encoder == "mlp":
                method = "mlp"
                config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
            elif args.encoder == "weighted":
                method = "weighted"
                config_suffix = "softmax"
            else:
                method = args.encoder
                config_suffix = "unknown"

            # Match classification naming: {method}_{task}_{model_family}_{model_size}_{config}.pt
            best_path = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}.pt"

            # Compute current metrics for checkpoint
            current_time = time.time() - start_time
            if torch.cuda.is_available():
                current_peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            else:
                current_peak_memory_mb = 0.0

            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_spearman": tr_spearman, "val_spearman": va_spearman,
                "train_loss": tr_loss, "val_loss": va_loss,
                "args": vars(args),
                "training_time_sec": current_time,
                "peak_memory_mb": current_peak_memory_mb,
                "param_count": sum(p.numel() for p in model.parameters()),
            }, best_path)
            print(f"★ New best model saved! Val spearman: {va_spearman:.4f}")

    # Compute training metrics
    training_time = time.time() - start_time
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory_mb = 0.0

    print("-"*60)
    print(f"Done. Best Val spearman: {best_val_spearman:.4f} @ epoch {best_epoch}")
    print(f"Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if best_path: print(f"Saved: {best_path}")

    # Save history + plots (same as before)
    import csv
    hist_path = save_dir / f"{args.task}_{args.model_family}_{args.model_size}_{args.encoder}_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(history.keys())
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"Saved history CSV: {hist_path}")

    # Loss plot
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch — {args.task} ({args.encoder})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    loss_png = save_dir / f"{args.task}_{args.model_family}_{args.model_size}_{args.encoder}_loss.png"
    plt.savefig(loss_png, dpi=160); plt.close()
    print(f"Saved loss plot: {loss_png}")

    # spearman plot
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_spearman"], label="Train spearman")
    plt.plot(history["epoch"], history["val_spearman"], label="Val spearman")
    plt.xlabel("Epoch"); plt.ylabel("spearman")
    plt.title(f"spearman vs Epoch — {args.task} ({args.encoder})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    sp_png = save_dir / f"{args.task}_{args.model_family}_{args.model_size}_{args.encoder}_spearman.png"
    plt.savefig(sp_png, dpi=160); plt.close()
    print(f"Saved spearman plot: {sp_png}")

if __name__ == "__main__":
    main()