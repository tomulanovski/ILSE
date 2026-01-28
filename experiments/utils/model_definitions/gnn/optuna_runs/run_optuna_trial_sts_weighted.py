#!/usr/bin/env python3
"""Optuna hyperparameter search for Weighted encoder on STS tasks."""
import optuna
from argparse import Namespace
from experiments.utils.model_definitions.gnn.sts_gin_optuna import train_and_eval_model


def objective(trial, task, model_family, model_size):
    args = Namespace(
        task=task,
        model_family=model_family,
        model_size=model_size,
        encoder="weighted",

        # Dummy params (not used for weighted)
        gin_hidden_dim=256,
        gin_layers=2,
        dropout=0.0,
        gin_mlp_layers=0,
        node_to_choose="last",
        graph_type="linear",
        mlp_input="last",
        mlp_hidden_dim=256,
        mlp_layers=0,

        # Actual hyperparameters to tune (minimal for weighted)
        epochs=25,
        batch_size=trial.suggest_categorical("batch_size", [64]),
        lr=trial.suggest_categorical("lr", [1e-4, 1e-3, 1e-2]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3]),

        save_dir="./weighted_sts_optuna",
        seed=trial.number,
        trial=trial
    )

    result = train_and_eval_model(args)

    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])

    return result["best_val_spearman"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--storage_url", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=1)
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    from functools import partial
    obj_fn = partial(objective,
                     task=args.task,
                     model_family=args.model_family,
                     model_size=args.model_size)

    study.optimize(obj_fn, n_trials=args.n_trials)

    print(f"\nBest trial:")
    print(f"  Value (Spearman): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
