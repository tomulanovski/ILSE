from math import e
import os
import argparse
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import time

from ..text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from .gnn_datasets import load_task_data, compute_layerwise, PairGraphRegressionDataset, PairTensorRegressionDataset, PairLayerwiseRegressionDataset
from .gnn_models import LayerGINEncoder, PairCosineSimScore, MLPEncoder, SingleClassifierLinearHead
from .sts_gin_trainer import train_epoch, validate_epoch


def train_and_eval_model(args: argparse.Namespace) -> float:
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_data = load_task_data(args.task, "train")
    try:
        val_data = load_task_data(args.task, "validation")
    except Exception:
        raise KeyError(f"No validation data for task: {args.task}")

    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)

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
                                      graph_type=args.graph_type)
        model = PairCosineSimScore(gin_encoder).to(device)
        train_mode = "gin"        

    elif args.encoder == "mlp":
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
                                 num_layers=args.mlp_layers, dropout=args.dropout)
        model = PairCosineSimScore(mlp_encoder).to(device)
        train_mode = "mlp"

    elif args.encoder == "weighted":
        # Import weighted encoder
        from .gnn_models import LearnedWeightingEncoder

        # Get dimensions
        L, D = train_lw_a[0].shape

        # Create datasets that keep full layerwise info
        train_dataset = PairLayerwiseRegressionDataset(
            train_lw_a, train_lw_b, train_data["original_scores"]
        )
        val_dataset = PairLayerwiseRegressionDataset(
            val_lw_a, val_lw_b, val_data["original_scores"]
        )

        # Standard DataLoader (not PyG)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Create weighted encoder
        weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
        model = PairCosineSimScore(weighted_encoder).to(device)
        train_mode = "weighted"

    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder}")

    model_param_count = sum(p.numel() for p in model.parameters())

    # Get score range from training data (pre-detected by load_task_data)
    min_score = train_data["min_score"]
    max_score = train_data["max_score"]
    print(f"Score range: [{min_score}, {max_score}]")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Track GPU memory and training time
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    best_val_spearman = -1.0
    best_epoch = -1
    epoch_logs = []

    start_time = time.time()
    for epoch in range(args.epochs):
        tr_loss, tr_spearman = train_epoch(model, train_loader, optimizer, criterion, device, train_mode, min_score, max_score)
        val_loss, val_spearman = validate_epoch(model, val_loader, criterion, device, train_mode, min_score, max_score)

        if hasattr(args, "trial"):
            args.trial.report(val_spearman, step=epoch)
        
        scheduler.step(val_loss)

        epoch_logs.append({
            "epoch": epoch + 1,
            "train_spearman": tr_spearman,
            "val_spearman": val_spearman,
            "train_loss": tr_loss,
            "val_loss": val_loss
        })

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch + 1

    duration = time.time() - start_time

    # Get peak GPU memory
    if torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory_mb = 0.0

    res = {
        "best_val_spearman": best_val_spearman,
        "best_epoch": best_epoch,
        "param_count": model_param_count,
        "train_time_sec": duration,
        "peak_memory_mb": peak_memory_mb,
        "epoch_logs": epoch_logs
        }
    return res
