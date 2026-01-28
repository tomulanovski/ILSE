import optuna
from optuna.trial import TrialState
import hashlib
import json
import time
from argparse import Namespace
from experiments.utils.model_definitions.gnn.basic_gin_optuna_precomputed import train_and_eval_model_precomputed


def get_param_hash(params_dict):
    """Create a hash of parameters for comparison."""
    # Filter out non-serializable values (like Trial objects)
    serializable_params = {}
    for k, v in params_dict.items():
        # Skip non-serializable objects (like Trial)
        if isinstance(v, (str, int, float, bool, type(None))):
            serializable_params[k] = v
        elif isinstance(v, (list, dict)):
            # Try to serialize to check if it's serializable
            try:
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
        else:
            # For other types, try to convert to string if possible
            try:
                json.dumps(v)
                serializable_params[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable values
                continue
    
    param_str = json.dumps(serializable_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()


def objective(trial, embeddings_dir, task, encoder_type, mlp_layer_idx, gin_layers_range, gin_mlp_layers_range, gin_hidden_dims, filter_graph_type=None, fixed_node_to_choose=None):
    # Common parameters
    common_args = {
        "embeddings_dir": embeddings_dir,
        "task": task,
        "encoder": encoder_type,
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        "epochs": 50,
        "batch_size": trial.suggest_categorical("batch_size", [64]),
        "lr": trial.suggest_categorical("lr", [1e-4, 1e-3]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        "seed": trial.number,
        "trial": trial
    }

    # Encoder-specific parameters
    if encoder_type == "gin":
        # If filter_graph_type is set, only suggest that type (more efficient)
        # Otherwise, suggest all types for comprehensive search
        if filter_graph_type is not None:
            graph_type = trial.suggest_categorical("graph_type", [filter_graph_type])
        else:
            # Always use full list of choices to maintain consistency with existing studies
            # This prevents "CategoricalDistribution does not support dynamic value space" errors
            all_graph_types = ["fully_connected", "cayley"]
            graph_type = trial.suggest_categorical("graph_type", all_graph_types)
        
        # Determine node_to_choose: if fixed value provided, still use suggest_categorical
        # with single option so it's recorded in trial.params
        if fixed_node_to_choose is not None:
            node_to_choose = trial.suggest_categorical("node_to_choose", [fixed_node_to_choose])
        else:
            # Pooling options: mean and sum
            pooling_options = ["mean", "sum"]
            node_to_choose = trial.suggest_categorical("node_to_choose", pooling_options)

        encoder_args = {
            "gin_hidden_dim": trial.suggest_categorical("gin_hidden_dim", gin_hidden_dims),
            "gin_layers": trial.suggest_int("gin_layers", gin_layers_range[0], gin_layers_range[1]),
            "gin_mlp_layers": trial.suggest_int("gin_mlp_layers", gin_mlp_layers_range[0], gin_mlp_layers_range[1]),
            "node_to_choose": node_to_choose,
            "graph_type": graph_type,
            # Dummy MLP params (not used)
            "mlp_input": "last",
            "mlp_hidden_dim": 256,
            "mlp_layers": 2,
        }
    elif encoder_type == "mlp":
        # Determine mlp_input mode based on layer_idx
        if mlp_layer_idx is not None:
            mlp_input = "layer"
        else:
            mlp_input = trial.suggest_categorical("mlp_input", ["last", "mean"])

        encoder_args = {
            "mlp_input": mlp_input,
            "mlp_layer_idx": mlp_layer_idx,
            "mlp_hidden_dim": trial.suggest_categorical("mlp_hidden_dim", [256]),
            "mlp_layers": trial.suggest_int("mlp_layers", 1, 3),
            # Dummy GIN params (not used)
            "gin_hidden_dim": 256,
            "gin_layers": 2,
            "gin_mlp_layers": 1,
            "node_to_choose": "last",
            "graph_type": "linear",
        }

    else:  # weighted
        encoder_args = {
            # Weighted has minimal hyperparameters (just layer weights)
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
    # Exclude 'trial' object as it's not JSON serializable and not needed for duplicate detection
    current_params = {**common_args, **encoder_args}
    params_for_hash = {k: v for k, v in current_params.items() if k != 'trial'}
    
    # Check for duplicates
    current_hash = get_param_hash(params_for_hash)

    study = trial.study
    for other_trial in study.trials:
        if other_trial.number == trial.number:
            continue

        # Check COMPLETE, RUNNING, and PRUNED (but NOT FAILED - allow retries after code fixes)
        # This prevents re-running:
        # - Successful trials (COMPLETE)
        # - Currently running trials (RUNNING) 
        # - Intentionally pruned trials (PRUNED)
        # But allows retrying failed trials (FAILED) - useful when fixing code bugs
        if other_trial.state in [TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED]:
            other_hash = get_param_hash(other_trial.params)

            if current_hash == other_hash:
                print(f"⚠️  Trial {trial.number} is DUPLICATE of trial {other_trial.number} (state: {other_trial.state})")
                print(f"    Params: {current_params}")
                print(f"    Pruning this duplicate trial")
                raise optuna.exceptions.TrialPruned("Duplicate parameters")

    # Unique config - proceed with training
    print(f"✓ Trial {trial.number} has unique config, starting training...")
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
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for GIN with precomputed embeddings")
    parser.add_argument("--study_name", type=str, required=True, help="Name for the Optuna study")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings (e.g., ./precomputed_embeddings/TinyLlama_1.1B_mean_pooling)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., EmotionClassification)")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (e.g., TinyLlama, Pythia)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (e.g., 1.1B, 410m)")
    parser.add_argument("--storage_url", type=str, required=True, help="PostgreSQL storage URL for Optuna")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of Optuna trials to run")
    parser.add_argument("--encoder", type=str, default="gin", choices=["gin", "mlp", "weighted"],
                        help="Encoder type: gin, mlp, or weighted")
    parser.add_argument("--mlp_layer_idx", type=int, default=None,
                        help="Specific layer index to use for MLP (only when encoder=mlp). If not set, will search over 'last' and 'mean'")
    parser.add_argument("--gin_layers_min", type=int, default=1,
                        help="Min number of GIN layers to search (default: 1)")
    parser.add_argument("--gin_layers_max", type=int, default=2,
                        help="Max number of GIN layers to search (default: 2)")
    parser.add_argument("--gin_mlp_layers_min", type=int, default=0,
                        help="Min number of MLP layers in GIN (default: 0)")
    parser.add_argument("--gin_mlp_layers_max", type=int, default=2,
                        help="Max number of MLP layers in GIN (default: 2)")
    parser.add_argument("--gin_hidden_dims", type=int, nargs="+", default=[256],
                        help="List of hidden dimensions to search for GIN (default: 256). E.g., --gin_hidden_dims 128 256 512")
    parser.add_argument("--filter_graph_type", type=str,
                        choices=["fully_connected", "cayley"],
                        help="Only test this graph_type. If not specified, tests all.")
    parser.add_argument("--node_to_choose", type=str,
                        choices=["mean", "sum"],
                        help="Fix node_to_choose to a specific value. If not specified, searches over ['mean', 'sum'].")
    args = parser.parse_args()
    
    print(f"Starting Optuna study: {args.study_name}")
    print(f"Task: {args.task}")
    print(f"Base Model: {args.model_family}-{args.model_size}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage URL: {args.storage_url}")
    if args.filter_graph_type:
        print(f"Graph Type Filter: {args.filter_graph_type} only")
    if args.node_to_choose:
        print(f"Node to Choose (fixed): {args.node_to_choose}")
    
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
    
    print(f"Pruner enabled: MedianPruner (stops trials with val_acc below median)")
    
    gin_layers_range = (args.gin_layers_min, args.gin_layers_max)
    gin_mlp_layers_range = (args.gin_mlp_layers_min, args.gin_mlp_layers_max)
    
    study.optimize(
        lambda trial: objective(
            trial,
            args.embeddings_dir,
            args.task,
            args.encoder,
            args.mlp_layer_idx,
            gin_layers_range,
            gin_mlp_layers_range,
            args.gin_hidden_dims,
            filter_graph_type=args.filter_graph_type,
            fixed_node_to_choose=args.node_to_choose
        ),
        n_trials=args.n_trials
    )
