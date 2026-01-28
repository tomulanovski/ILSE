import argparse
import optuna

def main():
    parser = argparse.ArgumentParser(description="Create an Optuna study using PostgreSQL.")
    parser.add_argument("--name", type=str, required=True, help="Name of the study (used as the study_name)")
    parser.add_argument("--user", type=str, default="USERNAME", help="PostgreSQL username")
    parser.add_argument("--host", type=str, default="localhost", help="PostgreSQL server host")
    parser.add_argument("--port", type=int, default=5432, help="PostgreSQL server port")
    parser.add_argument("--dbname", type=str, default="optuna", help="Database name")

    args = parser.parse_args()

    storage_url = f"postgresql://{args.user}@{args.host}:{args.port}/{args.dbname}"

    # Create the study (load if exists)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.name,
        storage=storage_url,
        load_if_exists=True,
    )

    print(f"Study '{args.name}' is ready in PostgreSQL database '{args.dbname}' at {args.host}:{args.port}")

if __name__ == "__main__":
    main()
