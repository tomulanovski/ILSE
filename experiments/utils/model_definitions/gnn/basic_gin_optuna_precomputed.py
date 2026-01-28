#!/usr/bin/env python3
"""
Optuna training using precomputed embeddings (no LLM loading).
Fast hyperparameter search!
"""
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import time
from pathlib import Path

from ...precompute.h5_utils import load_embeddings_from_h5
from .gnn_datasets import SingleGraphDataset, SimpleTensorDataset, LayerwiseTensorDataset
from .gnn_models import LayerGINEncoder, SingleClassifier, MLPEncoder, SingleClassifierLinearHead, LearnedWeightingEncoder, DeepSetEncoder, DWAttEncoder
from .basic_gin_trainer import train_epoch, validate_epoch


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_and_eval_model_precomputed(args: argparse.Namespace) -> dict:
    """
    Train and evaluate model using precomputed embeddings.

    Args:
        args: Arguments with:
            - embeddings_dir: Path to precomputed embeddings
            - task: Task name
            - encoder: 'gin' or 'mlp'
            - gin_*: GIN hyperparameters (if encoder=gin)
            - mlp_*: MLP hyperparameters (if encoder=mlp)
            - epochs, batch_size, lr, weight_decay, seed, trial (optional)

    Returns:
        Dict with best_val_acc, best_epoch, param_count, train_time_sec, epoch_logs
    """
    print("Starting train_and_eval_model_precomputed")
    print_gpu_memory()

    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed embeddings
    embeddings_base = Path(args.embeddings_dir)
    task_dir = embeddings_base / args.task

    train_h5 = task_dir / "train.h5"
    val_h5 = task_dir / "validation.h5"

    if not train_h5.exists():
        raise FileNotFoundError(f"Train embeddings not found: {train_h5}")
    if not val_h5.exists():
        raise FileNotFoundError(f"Validation embeddings not found: {val_h5}")

    print(f"Loading precomputed embeddings from {task_dir}...")
    train_emb, train_labels, train_meta = load_embeddings_from_h5(train_h5, load_metadata=True)
    val_emb, val_labels, val_meta = load_embeddings_from_h5(val_h5, load_metadata=True)

    print(f"  Train: {train_emb.shape}")
    print(f"  Validation: {val_emb.shape}")

    print("After loading embeddings:")
    print_gpu_memory()

    # Get dimensions and convert to list format
    N_train, L, D = train_emb.shape
    num_classes = train_meta.get('num_classes', len(np.unique(train_labels)))

    # Create proper numpy array copies (not views) to avoid torch.from_numpy() issues
    train_layerwise = [np.array(train_emb[i], copy=True) for i in range(len(train_emb))]
    val_layerwise = [np.array(val_emb[i], copy=True) for i in range(len(val_emb))]

    # Build model based on encoder type
    if args.encoder == "gin":
        train_dataset = SingleGraphDataset(train_layerwise, train_labels.tolist(), args.graph_type)
        val_dataset = SingleGraphDataset(val_layerwise, val_labels.tolist(), args.graph_type)

        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        gin_encoder = LayerGINEncoder(
            in_dim=D,
            hidden_dim=args.gin_hidden_dim,
            num_layers=args.gin_layers,
            dropout=args.dropout,
            gin_mlp_layers=args.gin_mlp_layers,
            node_to_choose=args.node_to_choose,
            graph_type=args.graph_type
        )
        model = SingleClassifier(gin_encoder, args.gin_hidden_dim, num_classes).to(device)
        train_mode = "gin"

    elif args.encoder == "mlp":
        # Determine input dimension based on mlp_input mode
        if args.mlp_input in ("last", "mean", "layer"):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        # Get layer_idx if mode is 'layer'
        layer_idx = getattr(args, 'mlp_layer_idx', None)

        train_dataset = SimpleTensorDataset(train_layerwise, train_labels.tolist(), mode=args.mlp_input, layer_idx=layer_idx)
        val_dataset = SimpleTensorDataset(val_layerwise, val_labels.tolist(), mode=args.mlp_input, layer_idx=layer_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        mlp_encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=args.mlp_hidden_dim,
            num_layers=args.mlp_layers,
            dropout=args.dropout
        )
        model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":
        # Weighted: learns scalar weights per layer (ELMo-style)
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_labels.tolist())
        val_dataset = LayerwiseTensorDataset(val_layerwise, val_labels.tolist())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = SingleClassifierLinearHead(weighted_encoder, D, num_classes).to(device)
        train_mode = "weighted"

    elif args.encoder == "deepset":
        # DeepSet: φ(pre-pooling) → POOL → ρ(post-pooling)
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
            dropout=args.dropout
        )
        model = SingleClassifierLinearHead(deepset_encoder, deepset_encoder.out_dim, num_classes).to(device)
        train_mode = "deepset"

    elif args.encoder == "dwatt":
        # DWAtt: Depth-Wise Attention (ElNokrashy et al. 2024)
        # Paper-faithful architecture with optional hidden_dim for fair comparison with GIN
        train_dataset = LayerwiseTensorDataset(train_layerwise, train_labels.tolist())
        val_dataset = LayerwiseTensorDataset(val_layerwise, val_labels.tolist())

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        dwatt_hidden_dim = getattr(args, 'dwatt_hidden_dim', None)
        dwatt_bottleneck_ratio = getattr(args, 'dwatt_bottleneck_ratio', 0.5)
        dwatt_pos_embed_dim = getattr(args, 'dwatt_pos_embed_dim', 24)

        dwatt_encoder = DWAttEncoder(
            num_layers=L,
            layer_dim=D,
            hidden_dim=dwatt_hidden_dim,  # None = paper-faithful, 256 = fair comparison
            bottleneck_ratio=dwatt_bottleneck_ratio,
            pos_embed_dim=dwatt_pos_embed_dim,
            dropout=args.dropout
        )
        model = SingleClassifierLinearHead(dwatt_encoder, dwatt_encoder.out_dim, num_classes).to(device)
        train_mode = "dwatt"

    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder}")

    model_param_count = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {model_param_count:,}")
    print("After building model:")
    print_gpu_memory()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0.0
    best_epoch = -1
    epoch_logs = []

    start_time = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, train_mode)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, train_mode)

        # Report to Optuna if trial is provided
        if hasattr(args, "trial"):
            args.trial.report(val_acc, step=epoch)

        scheduler.step(val_acc)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "train_loss": tr_loss,
            "val_loss": val_loss
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

    duration = time.time() - start_time

    res = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "param_count": model_param_count,
        "train_time_sec": duration,
        "epoch_logs": epoch_logs
    }

    return res
