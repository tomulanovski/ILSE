#!/usr/bin/env python3
"""
Basic GIN/MLP training using precomputed embeddings from H5 files.
No LLM loading required - fast training!
"""
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ...precompute.h5_utils import load_embeddings_from_h5, get_h5_metadata
from .gnn_datasets import SingleGraphDataset, SimpleTensorDataset, LayerwiseTensorDataset
from .gnn_models import LayerGINEncoder, SingleClassifier, MLPEncoder, SingleClassifierLinearHead, LearnedWeightingEncoder, DeepSetEncoder, DWAttEncoder


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
        else:  # mlp: batch is (X, y)
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
    parser = argparse.ArgumentParser(description="Basic GIN/MLP Training using precomputed embeddings")
    # Task and embeddings
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (must match precomputed embeddings directory)")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings (e.g., precomputed_embeddings/Pythia_410m_mean_pooling)")
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
    parser.add_argument("--mlp_input", type=str, default="last", choices=["last", "mean", "flatten"],
                        help="What the MLP consumes: last layer (D), mean over layers (D), or flattened (L*D).")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--mlp_layers", type=int, default=2)
    # DeepSet hyperparams
    parser.add_argument("--deepset_hidden_dim", type=int, default=256)
    parser.add_argument("--deepset_pre_pooling_layers", type=int, default=0)
    parser.add_argument("--deepset_post_pooling_layers", type=int, default=1)
    parser.add_argument("--deepset_pooling_type", type=str, default="mean", choices=["mean", "sum"])
    # DWAtt hyperparams (paper-faithful defaults from Table 4)
    parser.add_argument("--dwatt_hidden_dim", type=int, default=None,
                        help="Hidden dim for DWAtt. None=paper-faithful (operates in layer_dim), 256=fair comparison with GIN")
    parser.add_argument("--dwatt_bottleneck_ratio", type=float, default=0.5,
                        help="Bottleneck ratio gamma (paper default: 0.5)")
    parser.add_argument("--dwatt_pos_embed_dim", type=int, default=24,
                        help="Positional embedding dimension d_pos (paper default: 24)")
    # Linear mode (no ReLU activations)
    parser.add_argument("--use_linear", action="store_true",
                        help="Remove all ReLU activations for fair comparison with linear probing baselines")
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # Other
    parser.add_argument("--save_dir", type=str, default="./basic_gin_models_precomputed")
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

    print("="*70)
    print("GIN/MLP TRAINING WITH PRECOMPUTED EMBEDDINGS")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Embeddings: {args.embeddings_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"Linear mode: {args.use_linear}")
    print(f"Device: {device}")
    print("="*70)

    # Load precomputed embeddings
    embeddings_base = Path(args.embeddings_dir)
    task_dir = embeddings_base / args.task

    train_h5 = task_dir / "train.h5"
    val_h5 = task_dir / "validation.h5"

    if not train_h5.exists():
        raise FileNotFoundError(f"Train embeddings not found: {train_h5}")
    if not val_h5.exists():
        raise FileNotFoundError(f"Validation embeddings not found: {val_h5}")

    print(f"\nLoading precomputed embeddings...")
    train_emb, train_labels, train_meta = load_embeddings_from_h5(train_h5, load_metadata=True)
    val_emb, val_labels, val_meta = load_embeddings_from_h5(val_h5, load_metadata=True)

    print(f"  Train: {train_emb.shape} ({len(train_labels)} samples)")
    print(f"  Validation: {val_emb.shape} ({len(val_labels)} samples)")
    print(f"  Metadata: pooling={train_meta.get('pooling_method')}, model={train_meta.get('model_family')}-{train_meta.get('model_size')}")

    # Get dimensions
    N_train, L, D = train_emb.shape
    num_classes = train_meta.get('num_classes', len(np.unique(train_labels)))

    # Convert to list of (L, D) arrays for dataset compatibility
    # Create proper numpy array copies (not views) to avoid torch.from_numpy() issues
    train_layerwise = [np.array(train_emb[i], copy=True) for i in range(len(train_emb))]
    val_layerwise = [np.array(val_emb[i], copy=True) for i in range(len(val_emb))]

    # === Branch: GIN vs MLP ===
    if args.encoder == "gin":
        print(f"\nBuilding GIN model (graph_type={args.graph_type})...")
        train_dataset = SingleGraphDataset(train_layerwise, train_labels.tolist(), graph_type=args.graph_type,
                                           keep_embedding_layer=args.pool_real_nodes_only)
        val_dataset = SingleGraphDataset(val_layerwise, val_labels.tolist(), graph_type=args.graph_type,
                                         keep_embedding_layer=args.pool_real_nodes_only)
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        gin_encoder = LayerGINEncoder(
            in_dim=D,
            hidden_dim=args.gin_hidden_dim,
            num_layers=args.gin_layers,
            dropout=args.dropout,
            gin_mlp_layers=args.gin_mlp_layers,
            node_to_choose=args.node_to_choose,
            graph_type=args.graph_type,
            use_linear=args.use_linear,
            pool_real_nodes_only=args.pool_real_nodes_only,
            train_eps=args.train_eps
        )
        model = SingleClassifier(gin_encoder, args.gin_hidden_dim, num_classes).to(device)
        train_mode = "gin"

    elif args.encoder == "mlp":
        print(f"\nBuilding MLP model (input_mode={args.mlp_input})...")
        if args.mlp_input in ("last", "mean"):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        train_dataset = SimpleTensorDataset(train_layerwise, train_labels.tolist(), mode=args.mlp_input)
        val_dataset = SimpleTensorDataset(val_layerwise, val_labels.tolist(), mode=args.mlp_input)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        mlp_encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_layers=args.mlp_layers,
            dropout=args.dropout,
            use_linear=args.use_linear
        )
        model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":
        print(f"\nBuilding Weighted model...")
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_labels.tolist())
        val_dataset = LayerwiseTensorDataset(val_layerwise, val_labels.tolist())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = SingleClassifierLinearHead(weighted_encoder, D, num_classes).to(device)
        train_mode = "weighted"

    elif args.encoder == "deepset":
        print(f"\nBuilding DeepSet model (pooling={args.deepset_pooling_type}, pre={args.deepset_pre_pooling_layers}, post={args.deepset_post_pooling_layers})...")
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_labels.tolist())
        val_dataset = LayerwiseTensorDataset(val_layerwise, val_labels.tolist())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        deepset_encoder = DeepSetEncoder(
            num_layers=L,
            layer_dim=D,
            hidden_dim=args.deepset_hidden_dim,
            pre_pooling_layers=args.deepset_pre_pooling_layers,
            post_pooling_layers=args.deepset_post_pooling_layers,
            pooling_type=args.deepset_pooling_type,
            dropout=args.dropout,
            use_linear=args.use_linear
        )
        model = SingleClassifierLinearHead(deepset_encoder, deepset_encoder.out_dim, num_classes).to(device)
        train_mode = "deepset"

    elif args.encoder == "dwatt":
        hidden_str = f"hidden={args.dwatt_hidden_dim}" if args.dwatt_hidden_dim else "paper-faithful"
        print(f"\nBuilding DWAtt model ({hidden_str}, bottleneck={args.dwatt_bottleneck_ratio}, d_pos={args.dwatt_pos_embed_dim})...")
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_labels.tolist())
        val_dataset = LayerwiseTensorDataset(val_layerwise, val_labels.tolist())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        dwatt_encoder = DWAttEncoder(
            num_layers=L,
            layer_dim=D,
            hidden_dim=args.dwatt_hidden_dim,
            bottleneck_ratio=args.dwatt_bottleneck_ratio,
            pos_embed_dim=args.dwatt_pos_embed_dim,
            dropout=args.dropout
        )
        model = SingleClassifierLinearHead(dwatt_encoder, dwatt_encoder.out_dim, num_classes).to(device)
        train_mode = "dwatt"

    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder}")

    # Print class distribution
    print("\n" + "-"*50)
    ys_train = torch.tensor(train_labels, dtype=torch.long)
    print(f"Class distribution (train): {torch.bincount(ys_train).tolist()}")
    print(f"Majority baseline (train): {torch.bincount(ys_train).max().item() / len(ys_train):.4f}")

    ys_val = torch.tensor(val_labels, dtype=torch.long)
    print(f"Class distribution (validation): {torch.bincount(ys_val).tolist()}")
    print(f"Majority baseline (validation): {torch.bincount(ys_val).max().item() / len(ys_val):.4f}")
    print("-"*50)

    # Count model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # History & training
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc, best_epoch, best_path = 0.0, 0, None
    epochs_since_improvement = 0
    patience_for_checkpoints = 5

    # Extract model info from embeddings metadata for naming
    model_family = train_meta.get('model_family', 'unknown')
    model_size = train_meta.get('model_size', 'unknown')

    # Track training time and memory
    import time
    train_start_time = time.time()
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"\nStarting training for {args.epochs} epochs...\n" + "-"*60)
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, train_mode)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device, train_mode)
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)

        # Update peak memory
        if torch.cuda.is_available():
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / 1024**2)

        print(f"Epoch {epoch+1:2d}: Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.4f} | LR {lr_now:.2e}")

        improved = False
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch+1
            epochs_since_improvement = 0
            improved = True

            # Determine method type and config suffix (for correct filename)
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
            elif args.encoder == "dwatt":
                method = "dwatt"  # DWAtt has no linear variant
                hidden_suffix = f"_hidden{args.dwatt_hidden_dim}" if args.dwatt_hidden_dim else ""
                config_suffix = f"paper{hidden_suffix}"
            else:
                method = f"{linear_prefix}{args.encoder}"
                config_suffix = "unknown"

            # Save best model
            best_path = save_dir / f"{method}_{args.task}_{model_family}_{model_size}_{config_suffix}.pt"

            # Calculate training time so far
            train_time_so_far = time.time() - train_start_time

            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": tr_acc, "val_acc": va_acc,
                "train_loss": tr_loss, "val_loss": va_loss,
                "args": vars(args),
                "embeddings_metadata": train_meta,
                "param_count": param_count,
                "training_time_sec": train_time_so_far,
                "peak_memory_mb": peak_memory_mb,
            }, best_path)
            print(f"★ New best model saved! Val Acc: {va_acc:.4f}")
        else:
            epochs_since_improvement += 1

        # Save epoch checkpoint if within patience window
        if epochs_since_improvement < patience_for_checkpoints:
            checkpoint_path = checkpoints_dir / f"{args.task}_{model_family}_{model_size}_{args.encoder}_epoch{epoch+1}.pt"
            checkpoint_time = time.time() - train_start_time
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": tr_acc, "val_acc": va_acc,
                "train_loss": tr_loss, "val_loss": va_loss,
                "args": vars(args),
                "embeddings_metadata": train_meta,
                "param_count": param_count,
                "training_time_sec": checkpoint_time,
                "peak_memory_mb": peak_memory_mb,
            }, checkpoint_path)
            if improved:
                print(f"  → Checkpoint saved: epoch{epoch+1}.pt")
            else:
                print(f"  → Checkpoint saved: epoch{epoch+1}.pt ({epochs_since_improvement}/{patience_for_checkpoints} epochs since improvement)")
        elif epochs_since_improvement == patience_for_checkpoints:
            print(f"  ⚠ Stopped saving checkpoints (no improvement for {patience_for_checkpoints} epochs)")

    # Calculate final training time
    total_training_time = time.time() - train_start_time

    print("-"*60)
    print(f"Done. Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    if best_path:
        print(f"Saved: {best_path}")

    # Print training metrics summary
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(f"Model Parameters:     {param_count:,}")
    print(f"Training Time:        {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Peak GPU Memory:      {peak_memory_mb:.2f} MB")
    print(f"Best Epoch:           {best_epoch}/{args.epochs}")
    print(f"Best Val Accuracy:    {best_val_acc:.4f}")
    print("="*60)

    # Determine method type and config suffix (for output filenames)
    if args.encoder == "gin":
        method = "gcn" if args.gin_mlp_layers == 0 else "gin"
        config_suffix = args.graph_type
    elif args.encoder == "mlp":
        method = "mlp"
        config_suffix = f"{args.mlp_input}_layers{args.mlp_layers}"
    elif args.encoder == "weighted":
        method = "weighted"
        config_suffix = "softmax"
    elif args.encoder == "deepset":
        method = "deepset"
        config_suffix = f"{args.deepset_pooling_type}_pre{args.deepset_pre_pooling_layers}_post{args.deepset_post_pooling_layers}"
    elif args.encoder == "dwatt":
        method = "dwatt"
        hidden_suffix = f"_hidden{args.dwatt_hidden_dim}" if args.dwatt_hidden_dim else ""
        config_suffix = f"paper{hidden_suffix}"
    else:
        method = args.encoder
        config_suffix = "unknown"

    # Save history
    import csv
    hist_path = save_dir / f"{method}_{args.task}_{model_family}_{model_size}_{config_suffix}_history.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(history.keys())
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"Saved history CSV: {hist_path}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch — {args.task} ({method.upper()} {config_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_png = save_dir / f"{method}_{args.task}_{model_family}_{model_size}_{config_suffix}_loss.png"
    plt.savefig(loss_png, dpi=160)
    plt.close()
    print(f"Saved loss plot: {loss_png}")

    # Acc plot
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epoch — {args.task} ({method.upper()} {config_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_png = save_dir / f"{method}_{args.task}_{model_family}_{model_size}_{config_suffix}_acc.png"
    plt.savefig(acc_png, dpi=160)
    plt.close()
    print(f"Saved acc plot: {acc_png}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
