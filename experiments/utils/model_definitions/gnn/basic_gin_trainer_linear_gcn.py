#!/usr/bin/env python3
"""
Linear GCN training using precomputed embeddings.
Specifically for fair comparison with linear probing baselines.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

from ...precompute.h5_utils import load_embeddings_from_h5, get_h5_metadata
from .gnn_datasets import SingleGraphDataset
from .gnn_models import LayerGINEncoder, SingleClassifier


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch)
        labels = batch.y
        
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        total += labels.size(0)
        total_loss += loss.item()
        correct += (preds.argmax(dim=1) == labels).sum().item()
    
    return total_loss / max(1, len(train_loader)), correct / max(1, total)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)
            labels = batch.y
            
            total += labels.size(0)
            total_loss += criterion(preds, labels).item()
            correct += (preds.argmax(dim=1) == labels).sum().item()
    
    return total_loss / max(1, len(val_loader)), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Linear GCN Training (no ReLU) using precomputed embeddings")
    
    # Task and embeddings
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings")
    
    # Linear GCN hyperparams (fixed to 1 layer, no ReLU)
    parser.add_argument("--variant", type=int, default=2, choices=[1, 2],
                        help="Variant 1: output_dim=num_classes (no classifier head), Variant 2: output_dim=hidden_dim (with classifier head)")
    parser.add_argument("--gin_hidden_dim", type=int, default=256,
                        help="Hidden dimension (used for Variant 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--node_to_choose", type=str, default="mean",
                        choices=["mean"],
                        help="Pooling method: mean of all nodes")
    parser.add_argument("--graph_type", type=str, required=True,
                        choices=["linear", "virtual_node", "cayley", "cayley"],
                        help="Graph construction type")
    
    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    # Other
    parser.add_argument("--save_dir", type=str, default="./linear_gcn_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", default=None, help="Optuna trial object (if running from Optuna)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"Linear GCN Training (1 layer, no ReLU)")
    print(f"Task: {args.task}")
    print(f"Graph type: {args.graph_type}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Load precomputed embeddings from separate split files
    task_dir = os.path.join(args.embeddings_dir, args.task)
    train_h5 = os.path.join(task_dir, "train.h5")
    val_h5 = os.path.join(task_dir, "validation.h5")
    
    if not os.path.exists(train_h5):
        raise FileNotFoundError(f"Train embeddings not found: {train_h5}")
    if not os.path.exists(val_h5):
        raise FileNotFoundError(f"Validation embeddings not found: {val_h5}")
    
    print(f"Loading precomputed embeddings from: {task_dir}")
    train_emb, train_labels = load_embeddings_from_h5(train_h5, "train")
    val_emb, val_labels = load_embeddings_from_h5(val_h5, "validation")
    
    # Get dimensions
    N_train, L, D = train_emb.shape
    num_classes = len(np.unique(train_labels))
    
    print(f"Train: {N_train} samples, {L} layers, {D} dims per layer")
    print(f"Validation: {len(val_labels)} samples")
    print(f"Number of classes: {num_classes}")
    
    # Create graph datasets
    train_dataset = SingleGraphDataset(
        train_emb,
        train_labels.tolist(),
        graph_type=args.graph_type,
        add_virtual_node=(args.graph_type == "virtual_node")
    )
    
    val_dataset = SingleGraphDataset(
        val_emb,
        val_labels.tolist(),
        graph_type=args.graph_type,
        add_virtual_node=(args.graph_type == "virtual_node")
    )
    
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create Linear GCN encoder
    use_variant1 = (args.variant == 1)
    print(f"\nCreating Linear GCN encoder (Variant {args.variant}, 1 layer, no ReLU)...")
    print(f"  Variant: {args.variant} ({'direct num_classes output' if use_variant1 else 'hidden_dim + classifier head'})")
    print(f"  {'Output dim' if use_variant1 else 'Hidden dim'}: {num_classes if use_variant1 else args.gin_hidden_dim}")
    print(f"  Pooling: {args.node_to_choose}")
    print(f"  Dropout: {args.dropout}")
    
    gin_encoder = LayerGINEncoder(
        in_dim=D,
        hidden_dim=args.gin_hidden_dim,
        num_layers=1,  # Fixed to 1 layer for linear GCN
        dropout=args.dropout,
        gin_mlp_layers=0,  # Fixed to 0 for GCN mode
        node_to_choose=args.node_to_choose,
        graph_type=args.graph_type,
        use_linear=True,  # Remove ReLU (linear mode)
        output_dim=num_classes if use_variant1 else None  # Variant 1: direct output to num_classes
    )
    
    model = SingleClassifier(
        gin_encoder, 
        args.gin_hidden_dim, 
        num_classes, 
        use_variant1=use_variant1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": []
    }
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 80)
    start_time = time.time()
    
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        
        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)
        
        print(f"Epoch {epoch+1:2d}: Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.4f} | LR {lr_now:.2e}")
        
        # Report to Optuna if running from Optuna
        if args.trial is not None:
            args.trial.report(va_acc, epoch)
            # Early stopping check
            if args.trial.should_prune():
                print(f"Trial pruned at epoch {epoch+1}")
                raise Exception("Pruned by Optuna")
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch + 1
            
            # Save best model
            method = "linear_gcn"
            config_suffix = f"{args.graph_type}_{args.node_to_choose}_v{args.variant}"
            best_path = save_dir / f"{method}_{args.task}_{config_suffix}.pt"
            
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_acc": tr_acc,
                "val_acc": va_acc,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "args": vars(args),
            }, best_path)
            print(f"★ New best model saved! Val Acc: {va_acc:.4f}")
    
    training_time = time.time() - start_time
    
    print("-" * 80)
    print(f"Done. Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
    
    # Save history
    import csv
    method = "linear_gcn"
    config_suffix = f"{args.graph_type}_{args.node_to_choose}"
    hist_path = save_dir / f"{method}_{args.task}_{config_suffix}_history.csv"
    
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(history.keys())
        for i in range(len(history["epoch"])):
            w.writerow([history[k][i] for k in history.keys()])
    print(f"Saved history: {hist_path}")
    
    # Return for Optuna
    return best_val_acc


if __name__ == "__main__":
    main()

