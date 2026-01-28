#!/usr/bin/env python3
"""
Basic GIN implementation for layer-wise embeddings.
Simple, clean, and focused on the essentials.
"""
import os
import argparse
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from ..text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from .gnn_datasets import load_task_data, compute_layerwise, SingleGraphDataset, SimpleTensorDataset, LayerwiseTensorDataset, single_collate
from .gnn_models import LayerGINEncoder, SingleClassifier, MLPEncoder, SingleClassifierLinearHead, LearnedWeightingEncoder, DeepSetEncoder


def train_epoch(model, train_loader, optimizer, criterion, device, mode: str):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in train_loader:
        optimizer.zero_grad()
        if mode == "gin":
            batch = batch.to(device)
            preds = model(batch)
            labels = batch.y
        else:  # mlp or weighted: batch is (X, y)
            X, y = batch
            X, y = X.to(device), y.to(device)
            preds = model((X, y))
            labels = y

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_loss += loss.item()
        correct += (preds.argmax(dim=1) == labels).sum().item()

    return total_loss / max(1, len(train_loader)), correct / max(1, total)

def validate_epoch(model, val_loader, criterion, device, mode: str):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            if mode == "gin":
                batch = batch.to(device)
                preds = model(batch)
                labels = batch.y
            else:
                X, y = batch
                X, y = X.to(device), y.to(device)
                preds = model((X, y))
                labels = y

            total += labels.size(0)
            total_loss += criterion(preds, labels).item()
            correct += (preds.argmax(dim=1) == labels).sum().item()

    return total_loss / max(1, len(val_loader)), correct / max(1, total)

def main():
    parser = argparse.ArgumentParser(description="Basic GIN/MLP Training on layer embeddings")
    # Task and model
    parser.add_argument("--task", type=str, default="AmazonCounterfactualClassification")
    parser.add_argument("--model_family", type=str, default="Pythia")
    parser.add_argument("--model_size", type=str, default="14m")
    # Encoder choice
    parser.add_argument("--encoder", type=str, default="gin", choices=["gin", "mlp", "weighted", "deepset"])
    # GIN hyperparams
    parser.add_argument("--gin_hidden_dim", type=int, default=256)
    parser.add_argument("--gin_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gin_mlp_layers", type=int, default=1)
    parser.add_argument("--node_to_choose", type=str, default="mean")
    parser.add_argument("--graph_type", type=str, default="fully_connected",
                        choices=["linear", "fully_connected", "virtual_node", "cayley", "cayley"])
    # MLP hyperparams
    parser.add_argument("--mlp_input", type=str, default="last", choices=["last", "mean", "flatten"],
                        help="What the MLP consumes: last layer (D), mean over layers (D), or flattened (L*D).")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)
    # DeepSet hyperparams
    parser.add_argument("--deepset_pre_pooling_layers", type=int, default=0, help="Number of pre-pooling MLP layers (φ)")
    parser.add_argument("--deepset_post_pooling_layers", type=int, default=1, help="Number of post-pooling MLP layers (ρ)")
    parser.add_argument("--deepset_pooling_type", type=str, default="mean", choices=["mean", "sum"], help="Pooling type for DeepSet")
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

    # Try to load validation data, if not available, split from train
    val_split_ratio = 0.15  # Default 15% for validation
    try:
        val_data = load_task_data(args.task, "validation")
        print(f"Using existing validation split for {args.task}")
    except Exception:
        print(f"No validation split found for {args.task}. Splitting {val_split_ratio*100:.0f}% from train data.")

        # Perform stratified train-validation split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_data["text"],
            train_data["labels"],
            test_size=val_split_ratio,
            random_state=args.seed,
            stratify=train_data["labels"]  # Maintain class distribution
        )

        # Update train_data with split
        train_data["text"] = train_texts
        train_data["labels"] = train_labels

        # Create val_data dictionary
        val_data = {
            "text": val_texts,
            "labels": val_labels,
            "num_classes": train_data["num_classes"],
            "type": train_data["type"],
            "task_type": train_data["task_type"]
        }

        print(f"Split created: {len(train_texts)} train samples, {len(val_texts)} validation samples")

    # Wrapper for layerwise
    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)

    print("Extracting layer-wise embeddings...")
    train_layerwise = compute_layerwise(wrapper, train_data["text"], batch_size=256, token_pooling_method="mean")
    val_layerwise   = compute_layerwise(wrapper, val_data["text"],   batch_size=256, token_pooling_method="mean")
    num_classes = train_data["num_classes"]

    # === Branch: GIN vs MLP vs Weighted ===
    if args.encoder == "gin":
        # Graph datasets + PyG DataLoader
        train_dataset = SingleGraphDataset(train_layerwise, train_data["labels"], graph_type=args.graph_type)
        val_dataset   = SingleGraphDataset(val_layerwise,   val_data["labels"],   graph_type=args.graph_type)
        train_loader  = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = PyGDataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        # dims from one sample
        L, D = train_layerwise[0].shape
        gin_encoder = LayerGINEncoder(in_dim=D, hidden_dim=args.gin_hidden_dim,
                                      num_layers=args.gin_layers, dropout=args.dropout,
                                      gin_mlp_layers=args.gin_mlp_layers, node_to_choose=args.node_to_choose,
                                      graph_type=args.graph_type, use_linear=args.use_linear)
        model = SingleClassifier(gin_encoder, args.gin_hidden_dim, num_classes).to(device)
        train_mode = "gin"

    elif args.encoder == "mlp":  # MLP baseline
        # Build plain tensor datasets
        # Decide input dimension by mode
        L, D = train_layerwise[0].shape
        if args.mlp_input in ("last", "mean"):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        train_dataset = SimpleTensorDataset(train_layerwise, train_data["labels"], mode=args.mlp_input)
        val_dataset   = SimpleTensorDataset(val_layerwise,   val_data["labels"],   mode=args.mlp_input)

        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        mlp_encoder = MLPEncoder(in_dim=in_dim, hidden_dim=args.mlp_hidden_dim,
                                 num_layers=args.mlp_layers, dropout=args.dropout,
                                 use_linear=args.use_linear)
        model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":  # Weighted baseline (learned layer weighting)
        # Build layerwise tensor datasets (keeps all layers)
        L, D = train_layerwise[0].shape

        train_dataset = LayerwiseTensorDataset(train_layerwise, train_data["labels"])
        val_dataset   = LayerwiseTensorDataset(val_layerwise,   val_data["labels"])

        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = SingleClassifierLinearHead(weighted_encoder, weighted_encoder.out_dim, num_classes).to(device)
        train_mode = "weighted"

    else:  # DeepSet baseline
        # Build layerwise tensor datasets (keeps all layers)
        L, D = train_layerwise[0].shape

        train_dataset = LayerwiseTensorDataset(train_layerwise, train_data["labels"])
        val_dataset   = LayerwiseTensorDataset(val_layerwise,   val_data["labels"])

        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        deepset_encoder = DeepSetEncoder(
            num_layers=L,
            layer_dim=D,
            hidden_dim=args.gin_hidden_dim,  # Use same hidden_dim as GIN for fair comparison
            pre_pooling_layers=args.deepset_pre_pooling_layers,
            post_pooling_layers=args.deepset_post_pooling_layers,
            pooling_type=args.deepset_pooling_type,
            dropout=args.dropout,
            use_linear=args.use_linear
        )
        model = SingleClassifierLinearHead(deepset_encoder, deepset_encoder.out_dim, num_classes).to(device)
        train_mode = "deepset"

    if train_mode == "gin":
        print("---------------------------")
        ys = []
        for i in range(len(train_dataset)):
            ys.append(int(train_dataset[i].y))
        ys = torch.tensor(ys)
        print("Class distribution (train):", torch.bincount(ys))
        print("Majority baseline (train):", torch.bincount(ys).max().item() / len(ys))
        print("---------------------------")

        ys_val = []
        for i in range(len(val_dataset)):
            ys_val.append(int(val_dataset[i].y))
        ys_val = torch.tensor(ys_val)
        print("Class distribution (validation):", torch.bincount(ys_val))
        print("Majority baseline (validation):", torch.bincount(ys_val).max().item() / len(ys_val))
        print("---------------------------")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)

    # History & training
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"; checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc, best_epoch, best_path = 0.0, 0, None
    epochs_since_improvement = 0
    patience_for_checkpoints = 5  # Stop saving checkpoints after 5 epochs without improvement

    # Early stopping configuration
    early_stopping_patience = 15
    min_delta = 0.001
    no_improvement_count = 0

    # Reset GPU memory stats and start timer
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    import time
    start_time = time.time()

    print(f"\nStarting training for {args.epochs} epochs (early stopping patience={early_stopping_patience})...")
    print("-"*60)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, train_mode)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device, train_mode)
        scheduler.step(va_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        print(f"Epoch {epoch+1:2d}: Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.4f} | LR {lr_now:.2e}")

        # Check for improvement (with min_delta threshold)
        improved = False
        if va_acc > best_val_acc + min_delta:
            best_val_acc, best_epoch = va_acc, epoch+1
            epochs_since_improvement = 0
            improved = True
            no_improvement_count = 0

            # Determine method name and config suffix (supports weighted, mlp, gin, deepset)
            linear_prefix = "linear_" if args.use_linear else ""
            if args.encoder == "gin":
                base_method = "gcn" if args.gin_mlp_layers == 0 else "gin"
                method = f"{linear_prefix}{base_method}"
                config_suffix = args.graph_type
            elif args.encoder == "mlp":
                method = f"{linear_prefix}mlp"
                config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
            elif args.encoder == "weighted":
                method = "weighted"  # weighted is already linear, no prefix needed
                config_suffix = "softmax"
            elif args.encoder == "deepset":
                method = f"{linear_prefix}deepset"
                config_suffix = f"{args.deepset_pooling_type}_pre{args.deepset_pre_pooling_layers}_post{args.deepset_post_pooling_layers}"
            else:
                method = f"{linear_prefix}{args.encoder}"
                config_suffix = "unknown"

            best_path = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}.pt"

            # Get current metrics for checkpoint
            elapsed_time = time.time() - start_time
            current_peak_mem = 0.0
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
                current_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # Save best model
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": tr_acc, "val_acc": va_acc,
                "train_loss": tr_loss, "val_loss": va_loss,
                "train_time_sec": elapsed_time,
                "peak_memory_mb": current_peak_mem,
                "param_count": sum(p.numel() for p in model.parameters()),
                "args": vars(args),
            }, best_path)
            print(f"★ New best model saved! Val Acc: {va_acc:.4f}")
        else:
            no_improvement_count += 1
            epochs_since_improvement += 1

        # Save epoch checkpoint if still within patience window (from collaborator)
        if epochs_since_improvement < patience_for_checkpoints:
            checkpoint_path = checkpoints_dir / f"{args.task}_{args.model_family}_{args.model_size}_{args.encoder}_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": tr_acc, "val_acc": va_acc,
                "train_loss": tr_loss, "val_loss": va_loss,
                "args": vars(args),
            }, checkpoint_path)
            if improved:
                print(f"  → Checkpoint saved: epoch{epoch+1}.pt")
            else:
                print(f"  → Checkpoint saved: epoch{epoch+1}.pt ({epochs_since_improvement}/{patience_for_checkpoints} epochs since improvement)")
        elif epochs_since_improvement == patience_for_checkpoints:
            print(f"  ⚠ Stopped saving checkpoints (no improvement for {patience_for_checkpoints} epochs)")

        # Early stopping check
        if no_improvement_count >= early_stopping_patience:
            print(f"\n⏹️  Early stopping triggered: no improvement for {early_stopping_patience} epochs")
            break

    # Calculate final metrics
    train_time_sec = time.time() - start_time
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)

    print("-"*60)
    print(f"Done. Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Training time: {train_time_sec:.1f}s | Peak GPU memory: {peak_memory_mb:.1f} MB")
    if best_path: print(f"Saved: {best_path}")

    # Save history + plots (use same naming as model checkpoint)
    import csv
    # Determine method name and config suffix to match model checkpoint naming
    if args.encoder == "gin":
        method = "gcn" if args.gin_mlp_layers == 0 else "gin"
        config_suffix = args.graph_type
        method_display = f"{method.upper()} {config_suffix}"
    elif args.encoder == "mlp":
        method = "mlp"
        config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
        method_display = f"MLP {args.mlp_input} layers={args.mlp_layers}"
    elif args.encoder == "weighted":
        method = "weighted"
        config_suffix = "softmax"
        method_display = f"Weighted (softmax)"
    else:
        method = args.encoder
        config_suffix = "unknown"
        method_display = method.upper()

    hist_path = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_history.csv"
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
    plt.title(f"Loss vs Epoch — {args.task} ({method_display})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    loss_png = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_loss.png"
    plt.savefig(loss_png, dpi=160); plt.close()
    print(f"Saved loss plot: {loss_png}")

    # Acc plot
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epoch — {args.task} ({method_display})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    acc_png = save_dir / f"{method}_{args.task}_{args.model_family}_{args.model_size}_{config_suffix}_acc.png"
    plt.savefig(acc_png, dpi=160); plt.close()
    print(f"Saved acc plot: {acc_png}")

if __name__ == "__main__":
    main()