#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for Linear GCN (1 layer, no ReLU).
For fair comparison with linear probing baselines.
"""
import optuna
from argparse import Namespace
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.utils.model_definitions.gnn.basic_gin_trainer_linear_gcn import main as train_linear_gcn


def objective(trial, task, embeddings_dir, filter_variant=None, filter_graph_type=None):
    """
    Optuna objective function for Linear GCN.
    
    Searches over:
    - variant: 1 (direct num_classes), 2 (hidden_dim + classifier head)
    - graph_type: virtual_node, cayley, linear, cayley
    - hidden_dim: 128, 256, 512
    - dropout: 0.0 to 0.3
    - node_to_choose: mean
    - lr: 1e-4, 1e-3
    - weight_decay: 1e-4, 1e-3
    
    Args:
        trial: Optuna trial
        task: Task name
        embeddings_dir: Path to embeddings
        filter_variant: If specified (1 or 2), only test that variant
        filter_graph_type: If specified, only test that graph_type (e.g., "cayley")
    """
    
    # Hyperparameters to search
    if filter_variant is not None:
        variant = filter_variant  # Fixed variant
    else:
        variant = trial.suggest_categorical("variant", [1, 2])  # Search over both
    
    # If filter_graph_type is set, only suggest that type (more efficient)
    # Otherwise, suggest all types for comprehensive search
    if filter_graph_type is not None:
        graph_type = trial.suggest_categorical("graph_type", [filter_graph_type])
    else:
        # Always use full list of choices to maintain consistency with existing studies
        # This prevents "CategoricalDistribution does not support dynamic value space" errors
        all_graph_types = ["fully_connected", "cayley"]
        graph_type = trial.suggest_categorical("graph_type", all_graph_types)
    hidden_dim = trial.suggest_categorical("gin_hidden_dim", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    node_to_choose = trial.suggest_categorical("node_to_choose", ["mean"])
    lr = trial.suggest_categorical("lr", [1e-4, 1e-3])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    
    # Create args namespace
    args = Namespace(
        task=task,
        embeddings_dir=embeddings_dir,
        variant=variant,
        gin_hidden_dim=hidden_dim,
        dropout=dropout,
        node_to_choose=node_to_choose,
        graph_type=graph_type,
        epochs=50,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        save_dir="./linear_gcn_optuna",
        seed=trial.number,  # Use trial number as seed
        trial=trial  # Pass trial for reporting
    )
    
    # Convert to argument list for argparse
    sys.argv = [
        'run_optuna_trial_linear_gcn.py',
        '--task', args.task,
        '--embeddings_dir', args.embeddings_dir,
        '--variant', str(args.variant),
        '--gin_hidden_dim', str(args.gin_hidden_dim),
        '--dropout', str(args.dropout),
        '--node_to_choose', args.node_to_choose,
        '--graph_type', args.graph_type,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--save_dir', args.save_dir,
        '--seed', str(args.seed),
    ]
    
    try:
        # Run training
        best_val_acc = train_linear_gcn()
        return best_val_acc
    except Exception as e:
        if "Pruned by Optuna" in str(e):
            # Re-raise pruning exception
            raise optuna.TrialPruned()
        else:
            # Other exceptions
            print(f"Trial failed with error: {e}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for Linear GCN")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings")
    parser.add_argument("--study_name", type=str, required=True,
                        help="Optuna study name")
    parser.add_argument("--storage_url", type=str, required=True,
                        help="PostgreSQL storage URL for Optuna")
    parser.add_argument("--n_trials", type=int, default=1,
                        help="Number of trials to run")
    parser.add_argument("--filter_variant", type=int, choices=[1, 2],
                        help="Only test this variant (1 or 2). If not specified, tests both.")
    parser.add_argument("--filter_graph_type", type=str,
                        choices=["virtual_node", "cayley", "linear", "cayley"],
                        help="Only test this graph_type. If not specified, tests all.")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Linear GCN Optuna Optimization")
    print(f"Task: {args.task}")
    print(f"Study: {args.study_name}")
    print(f"Trials: {args.n_trials}")
    if args.filter_variant:
        print(f"Variant Filter: {args.filter_variant} only")
    if args.filter_graph_type:
        print(f"Graph Type Filter: {args.filter_graph_type} only")
    print(f"{'='*80}\n")
    
    # Create or load study
    study = optuna.create_study(
        direction="maximize",  # Maximize validation accuracy
        study_name=args.study_name,
        storage=args.storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args.task, args.embeddings_dir, 
                               filter_variant=args.filter_variant,
                               filter_graph_type=args.filter_graph_type),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("Optimization Complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")

