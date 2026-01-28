import os
import argparse
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import time
from sklearn.model_selection import train_test_split

from ..text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from .gnn_datasets import load_task_data, compute_layerwise, SingleGraphDataset, SimpleTensorDataset, LayerwiseTensorDataset
from .gnn_models import LayerGINEncoder, SingleClassifier, MLPEncoder, LearnedWeightingEncoder, SingleClassifierLinearHead
from .basic_gin_trainer import train_epoch, validate_epoch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_and_eval_model(args: argparse.Namespace) -> float:
    print("Starting train_and_eval_model")
    print_gpu_memory()

    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = load_task_data(args.task, "train")

    # Try to load validation data, if not available, split from train
    val_split_ratio = getattr(args, "val_split_ratio", 0.15)  # Default 15% for validation
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
            random_state=seed,
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

    print("After loading data:")
    print_gpu_memory()


    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)

    print("After loading model:")
    print_gpu_memory()


    train_layerwise = compute_layerwise(wrapper, train_data["text"], batch_size=args.batch_size, token_pooling_method="mean")
    val_layerwise   = compute_layerwise(wrapper, val_data["text"],   batch_size=args.batch_size, token_pooling_method="mean")

    print("After computing layer-wise embeddings:")
    print_gpu_memory()

    num_classes = train_data["num_classes"]

    if args.encoder == "gin":
        train_dataset = SingleGraphDataset(train_layerwise, train_data["labels"], args.graph_type)
        val_dataset   = SingleGraphDataset(val_layerwise,   val_data["labels"],   args.graph_type)

        train_loader  = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader    = PyGDataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        L, D = train_layerwise[0].shape
        gin_encoder = LayerGINEncoder(in_dim=D, hidden_dim=args.gin_hidden_dim,
                                      num_layers=args.gin_layers, dropout=args.dropout,
                                      gin_mlp_layers=args.gin_mlp_layers, node_to_choose=args.node_to_choose,
                                      graph_type=args.graph_type)
        model = SingleClassifier(gin_encoder, args.gin_hidden_dim, num_classes).to(device)
        train_mode = "gin"

    elif args.encoder == "mlp":
        L, D = train_layerwise[0].shape
        in_dim = D if args.mlp_input in ("last", "mean") else L * D
        train_dataset = SimpleTensorDataset(train_layerwise, train_data["labels"], mode=args.mlp_input)
        val_dataset   = SimpleTensorDataset(val_layerwise,   val_data["labels"],   mode=args.mlp_input)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        mlp_encoder = MLPEncoder(in_dim=in_dim, hidden_dim=args.mlp_hidden_dim,
                                 num_layers=args.mlp_layers, dropout=args.dropout)
        model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":
        L, D = train_layerwise[0].shape
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_data["labels"])
        val_dataset   = LayerwiseTensorDataset(val_layerwise,   val_data["labels"])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = SingleClassifierLinearHead(weighted_encoder, weighted_encoder.out_dim, num_classes).to(device)
        train_mode = "weighted"

    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder}")

    model_param_count = sum(p.numel() for p in model.parameters())


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_val_acc = 0.0
    best_epoch = -1
    epoch_logs = []

    # Early stopping configuration (state-of-the-art: patience + min_delta)
    early_stopping_patience = getattr(args, "early_stopping_patience", 15)
    min_delta = getattr(args, "min_delta", 0.001)  # Minimum improvement threshold
    no_improvement_count = 0

    # Reset GPU memory stats to track peak usage during training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, train_mode)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, train_mode)

        # Optuna intermediate reporting (for pruning bad trials)
        if hasattr(args, "trial"):
            args.trial.report(val_acc, step=epoch)
            # Check if trial should be pruned (stop unpromising trials early)
            if args.trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        scheduler.step(val_acc)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "train_loss": tr_loss,
            "val_loss": val_loss
        })

        # Early stopping: check for significant improvement
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Stop if no improvement for 'patience' epochs
        if no_improvement_count >= early_stopping_patience:
            # Note: We don't restore model in Optuna since we don't save checkpoints
            # This is intentional to save disk space during hyperparameter search
            break

    duration = time.time() - start_time

    # Get peak GPU memory usage
    peak_memory_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 ** 2)  # Convert to MB

    res = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "param_count": model_param_count,
        "train_time_sec": duration,
        "peak_memory_mb": peak_memory_mb,
        "epoch_logs": epoch_logs
        }
    return res
