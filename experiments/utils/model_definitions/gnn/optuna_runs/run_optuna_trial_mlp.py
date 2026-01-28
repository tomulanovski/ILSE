import optuna
from argparse import Namespace
from experiments.utils.model_definitions.gnn.basic_gin_optuna import train_and_eval_model

# Classification tasks supported
CLASSIFICATION_TASKS = [
    "AmazonCounterfactualClassification",
    "Banking77Classification",
    "MTOPIntentClassification",
    "EmotionClassification",
    "MassiveIntentClassification",
    "MTOPDomainClassification",
    "MassiveScenarioClassification"
]

# Global variables set from command line
TASK_NAME = None
MODEL_FAMILY = None
MODEL_SIZE = None

def objective(trial):
    # Validate task is classification
    if TASK_NAME not in CLASSIFICATION_TASKS:
        raise ValueError(f"Task '{TASK_NAME}' is not a supported classification task. Supported: {CLASSIFICATION_TASKS}")

    args = Namespace(
        task=TASK_NAME,
        model_family=MODEL_FAMILY,
        model_size=MODEL_SIZE,
        encoder="mlp",
        # MLP-specific hyperparameters
        mlp_input=trial.suggest_categorical("mlp_input", ["last", "mean"]),
        mlp_hidden_dim=256,  # Fixed to match GIN
        mlp_layers=trial.suggest_int("mlp_layers", 0, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        # Training hyperparameters
        epochs=50,
        batch_size=trial.suggest_categorical("batch_size", [64]),
        lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
        weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        save_dir="./mlp_optuna",
        seed=trial.number,
        trial=trial
    )
    result = train_and_eval_model(args)

    # Log custom metrics
    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])
    trial.set_user_attr("peak_memory_mb", result["peak_memory_mb"])
    trial.set_user_attr("epoch_logs", result["epoch_logs"])

    return result["best_val_acc"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for MLP baseline on classification tasks")
    parser.add_argument("--study_name", type=str, required=True, help="Name for the Optuna study")
    parser.add_argument("--task", type=str, required=True, help="MTEB classification task name")
    parser.add_argument("--model_family", type=str, default="Pythia", help="Model family (e.g., Pythia, Llama3)")
    parser.add_argument("--model_size", type=str, default="410m", help="Model size (e.g., 410m, 2.8b, 8B)")
    parser.add_argument("--storage_url", type=str, required=True, help="PostgreSQL storage URL for Optuna")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of Optuna trials to run")
    args = parser.parse_args()

    # Set global variables for objective function
    TASK_NAME = args.task
    MODEL_FAMILY = args.model_family
    MODEL_SIZE = args.model_size

    print(f"Starting Optuna study: {args.study_name}")
    print(f"Task: {TASK_NAME}")
    print(f"Base Model: {MODEL_FAMILY}-{MODEL_SIZE}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Storage URL: {args.storage_url}")

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
    study.optimize(objective, n_trials=args.n_trials)
