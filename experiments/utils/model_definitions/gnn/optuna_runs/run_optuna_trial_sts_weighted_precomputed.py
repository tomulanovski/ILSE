#!/usr/bin/env python3
"""
Optuna hyperparameter search for Weighted encoder on STS tasks with precomputed embeddings.
"""
import optuna
from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
from scipy.stats import spearmanr

from experiments.utils.model_definitions.gnn.gnn_datasets import PrecomputedSTSLayerwiseDataset
from experiments.utils.model_definitions.gnn.gnn_models import LearnedWeightingEncoder, PairCosineSimScore
from experiments.utils.gpu_tracking import GPUTracker


def _cos_to_01(cos: torch.Tensor) -> torch.Tensor:
    """DEPRECATED: Map cosine similarity [-1, 1] to [0, 1]. Use _cos_to_score instead."""
    return (cos + 1.0) * 0.5


def _cos_to_score(cos: torch.Tensor, min_score: float, max_score: float) -> torch.Tensor:
    """Map cosine similarity [-1, 1] to score range [min_score, max_score]."""
    normalized = (cos + 1.0) * 0.5
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


def train_and_eval_model(args):
    """Train and evaluate Weighted model with precomputed embeddings."""
    seed = getattr(args, "seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed embeddings
    h5_path = Path(args.embeddings_dir) / f"{args.task}.h5"

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
    L, D = train_dataset.num_layers, train_dataset.layer_dim

    # Build model
    weighted_encoder = LearnedWeightingEncoder(num_layers=L, layer_dim=D)
    model = PairCosineSimScore(weighted_encoder).to(device)

    model_param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=False)

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
    print(f"Score range for {args.task}: [{min_score}, {max_score}]")

    best_val_spearman = -1.0
    best_epoch = -1
    epoch_logs = []

    # Initialize robust GPU tracker
    gpu_tracker = GPUTracker(device, require_gpu=False)
    gpu_tracker.start()

    for epoch in range(args.epochs):
        tr_loss, tr_spearman = train_epoch(model, train_loader, optimizer, criterion, device, min_score, max_score)
        val_loss, val_spearman = validate_epoch(model, val_loader, criterion, device, min_score, max_score)

        if hasattr(args, "trial"):
            args.trial.report(val_spearman, step=epoch)
            if args.trial.should_prune():
                raise optuna.TrialPruned()

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

    # Get runtime and GPU memory metrics
    metrics = gpu_tracker.stop()

    res = {
        "best_val_spearman": best_val_spearman,
        "best_epoch": best_epoch,
        "param_count": model_param_count,
        "train_time_sec": metrics["runtime_sec"],
        "peak_memory_mb": metrics["peak_memory_mb"],
        "peak_memory_gb": metrics["peak_memory_gb"],
        "epoch_logs": epoch_logs
    }
    return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--storage_url", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=1)
    cmd_args = parser.parse_args()

    def objective_with_args(trial):
        args = Namespace(
            task=cmd_args.task,
            model_family=cmd_args.model_family,
            model_size=cmd_args.model_size,
            embeddings_dir=cmd_args.embeddings_dir,
            encoder="weighted",
            epochs=25,
            batch_size=trial.suggest_categorical("batch_size", [64]),
            lr=trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
            weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3]),
            seed=trial.number,
            trial=trial
        )
        result = train_and_eval_model(args)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("param_count", result["param_count"])
        trial.set_user_attr("train_time_sec", result["train_time_sec"])
        trial.set_user_attr("peak_memory_mb", result["peak_memory_mb"])
        trial.set_user_attr("epoch_logs", result["epoch_logs"])
        return result["best_val_spearman"]

    study = optuna.create_study(
        direction="maximize",
        study_name=cmd_args.study_name,
        storage=cmd_args.storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    study.optimize(objective_with_args, n_trials=cmd_args.n_trials)
