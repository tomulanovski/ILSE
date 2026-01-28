#!/usr/bin/env python3
"""
Optuna hyperparameter search for DWAtt (Depth-Wise Attention) encoder on classification tasks
with precomputed embeddings.

DWAtt is from ElNokrashy et al. 2024:
"Depth-Wise Attention (DWAtt): A Layer Fusion Method for Data-Efficient Classification"
https://arxiv.org/abs/2209.15168

DWAtt architecture is FIXED per paper (Table 4):
- d_pos = 24 (positional embedding dimension)
- gamma = 0.5 (bottleneck ratio)
- hidden_dim = None (paper-faithful, operates in layer_dim)

Only training hyperparameters are searched:
- dropout, lr, weight_decay
"""
import optuna
from optuna.trial import TrialState
import hashlib
import json
import time
from argparse import Namespace
from experiments.utils.model_definitions.gnn.basic_gin_optuna_precomputed import train_and_eval_model_precomputed


def get_param_hash(params_dict):
    """Create a hash of parameters for comparison."""
    serializable_params = {}
    for k, v in params_dict.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable_params[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, ValueError):
                continue
        else:
            try:
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, ValueError):
                continue

    param_str = json.dumps(serializable_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def objective(trial, embeddings_dir, task):
    # Common parameters
    common_args = {
        "embeddings_dir": embeddings_dir,
        "task": task,
        "encoder": "dwatt",
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        "epochs": 50,
        "batch_size": trial.suggest_categorical("batch_size", [64]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        "seed": trial.number,
        "trial": trial
    }

    # DWAtt-specific parameters (FIXED per paper Table 4)
    encoder_args = {
        "dwatt_hidden_dim": None,           # Paper-faithful (no input projection)
        "dwatt_bottleneck_ratio": 0.5,      # Paper: gamma = 0.5
        "dwatt_pos_embed_dim": 24,          # Paper: d_pos = 24
        # Dummy GIN params (not used)
        "gin_hidden_dim": 256,
        "gin_layers": 2,
        "gin_mlp_layers": 1,
        "node_to_choose": "last",
        "graph_type": "linear",
        # Dummy MLP params (not used)
        "mlp_input": "last",
        "mlp_hidden_dim": 256,
        "mlp_layers": 2,
    }

    # Small delay to let database sync
    time.sleep(2)

    # Build current_params for duplicate detection
    current_params = {**common_args, **encoder_args}
    params_for_hash = {k: v for k, v in current_params.items() if k != 'trial'}

    # Check for duplicates
    current_hash = get_param_hash(params_for_hash)

    study = trial.study
    for other_trial in study.trials:
        if other_trial.number == trial.number:
            continue

        # Check COMPLETE, RUNNING, and PRUNED (but NOT FAILED - allow retries after code fixes)
        if other_trial.state in [TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED]:
            other_hash = get_param_hash(other_trial.params)

            if current_hash == other_hash:
                print(f"Warning: Trial {trial.number} is DUPLICATE of trial {other_trial.number} (state: {other_trial.state})")
                print(f"    Params: {current_params}")
                print(f"    Pruning this duplicate trial")
                raise optuna.exceptions.TrialPruned("Duplicate parameters")

    # Unique config - proceed with training
    print(f"Trial {trial.number} has unique config, starting training...")
    args = Namespace(**common_args, **encoder_args)
    result = train_and_eval_model_precomputed(args)

    # Log custom metrics
    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])
    trial.set_user_attr("epoch_logs", result["epoch_logs"])

    return result["best_val_acc"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for DWAtt with precomputed embeddings")
    parser.add_argument("--study_name", type=str, required=True, help="Name for the Optuna study")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings (e.g., ./precomputed_embeddings/TinyLlama_1.1B_mean_pooling)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., EmotionClassification)")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., TinyLlama, Pythia)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (e.g., 1.1B, 410m)")
    parser.add_argument("--storage_url", type=str, required=True, help="PostgreSQL storage URL for Optuna")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of Optuna trials to run")
    args = parser.parse_args()

    print(f"Starting Optuna study: {args.study_name}")
    print(f"Task: {args.task}")
    print(f"Base Model: {args.model_family}-{args.model_size}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage URL: {args.storage_url}")
    print()
    print("DWAtt architecture (paper-faithful, Table 4):")
    print("  - d_pos = 24 (positional embedding dimension)")
    print("  - gamma = 0.5 (bottleneck ratio)")
    print("  - hidden_dim = None (operates in layer_dim)")
    print()

    # Create pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # Don't prune first 5 trials (need baseline)
        n_warmup_steps=10,     # Wait 10 epochs before considering pruning
        interval_steps=5       # Check every 5 epochs
    )

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
        pruner=pruner
    )

    study.optimize(
        lambda trial: objective(trial, args.embeddings_dir, args.task),
        n_trials=args.n_trials,
        callbacks=[],
    )

    print(f"\nCompleted {args.n_trials} trials for study: {args.study_name}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
