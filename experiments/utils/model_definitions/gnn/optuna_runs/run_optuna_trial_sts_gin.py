import optuna
from argparse import Namespace
from experiments.utils.model_definitions.gnn.sts_gin_optuna import train_and_eval_model

def objective(trial):
    args = Namespace(
        task="STSBenchmark",
        model_family="Pythia",
        model_size="410m",
        encoder="gin",
        # gin_hidden_dim=trial.suggest_categorical("gin_hidden_dim", [64, 128, 256, 512]),
        gin_hidden_dim=trial.suggest_categorical("gin_hidden_dim", [256]),
        gin_layers=trial.suggest_int("gin_layers", 1, 2),
        dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
        gin_mlp_layers=trial.suggest_int("gin_mlp_layers", 0, 2),
        node_to_choose=trial.suggest_categorical("node_to_choose", ["mean", "sum"]),
        graph_type=trial.suggest_categorical("graph_type", ["fully_connected", "cayley"]),
        mlp_input="last",
        mlp_hidden_dim=256,
        mlp_layers=2,
        epochs=25,
        batch_size=trial.suggest_categorical("batch_size", [64]),
        # lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
        # weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
        save_dir="./gin_optuna",  # no actual saving here
        seed=trial.number,  # for reproducibility
        trial=trial
    )
    result = train_and_eval_model(args)

    # Log custom metrics
    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("param_count", result["param_count"])
    trial.set_user_attr("train_time_sec", result["train_time_sec"])
    trial.set_user_attr("epoch_logs", result["epoch_logs"])  # you can later inspect it programmatically

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
    cmd_args = parser.parse_args()

    # Update objective to use command-line arguments
    def objective_with_args(trial):
        args = Namespace(
            task=cmd_args.task,
            model_family=cmd_args.model_family,
            model_size=cmd_args.model_size,
            encoder="gin",
            gin_hidden_dim=trial.suggest_categorical("gin_hidden_dim", [256]),
            gin_layers=trial.suggest_int("gin_layers", 1, 2),
            dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
            gin_mlp_layers=trial.suggest_int("gin_mlp_layers", 0, 2),
            node_to_choose=trial.suggest_categorical("node_to_choose", ["mean", "sum"]),
            graph_type=trial.suggest_categorical("graph_type", ["fully_connected", "cayley"]),
            mlp_input="last",
            mlp_hidden_dim=256,
            mlp_layers=2,
            epochs=25,
            batch_size=trial.suggest_categorical("batch_size", [64]),
            lr=trial.suggest_categorical("lr", [1e-4, 1e-3]),
            weight_decay=trial.suggest_categorical("weight_decay", [1e-4, 1e-3]),
            save_dir="./gin_optuna",
            seed=trial.number,
            trial=trial
        )
        result = train_and_eval_model(args)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("param_count", result["param_count"])
        trial.set_user_attr("train_time_sec", result["train_time_sec"])
        trial.set_user_attr("epoch_logs", result["epoch_logs"])
        return result["best_val_spearman"]

    study = optuna.create_study(
        direction="maximize",
        study_name=cmd_args.study_name,
        storage=cmd_args.storage_url,
        load_if_exists=True,
    )

    study.optimize(objective_with_args, n_trials=cmd_args.n_trials)
