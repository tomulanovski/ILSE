import argparse
import optuna
import os

def main():
    parser = argparse.ArgumentParser(description="Create an Optuna study with a specified name.")
    parser.add_argument("--name", type=str, required=True, help="Name of the study (used for study name and DB filename)")
    args = parser.parse_args()

    study_name = args.name
    # Use environment variable for database path
    optuna_dir = os.getenv("GNN_OPTUNA_DB_DIR", os.path.join(os.getenv("GNN_REPO_DIR", os.getcwd()), "optuna_db"))
    os.makedirs(optuna_dir, exist_ok=True)
    db_path = os.path.join(optuna_dir, f"{study_name}.db")
    storage_url = f"sqlite:///{db_path}"

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    print(f"Study '{study.study_name}' created at: {db_path}")

if __name__ == "__main__":
    main()
