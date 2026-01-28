#!/usr/bin/env python3
"""
Generate SLURM jobs for DeepSet Optuna hyperparameter search (using precomputed embeddings).
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load user configuration from .env
load_dotenv()

# Classification tasks
TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]

# Models to run with model-specific time allocations
MODELS = {
    "Pythia-410m": {"family": "Pythia", "size": "410m", "time": "04:00:00"},
    "TinyLlama-1.1B": {"family": "TinyLlama", "size": "1.1B", "time": "06:00:00"},
    "Pythia-2.8b": {"family": "Pythia", "size": "2.8b", "time": "08:00:00"},
    "Llama3-8B": {"family": "Llama3", "size": "8B", "time": "08:00:00"},
}

# DeepSet grid search:
# - pre_pooling_layers: {0, 1}
# - post_pooling_layers: {0, 1, 2}
# - pooling_type: {mean, sum}
# - lr, weight_decay, dropout (standard ranges)
# Total combinations: 2 * 3 * 2 = 12 configs * ~8 hyperparameter combinations = ~96 trials
# Round up to 100 for safety
N_TRIALS = 100

# SLURM configuration (base config, time is model-specific)
SLURM_CONFIG = {
    "mem": "32G",
    "cpus": 4,
    "partition": "your-lab-partition",
    "qos": "owner",
    "gpu": "gpu:1",
}

# Paths (loaded from .env file)
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")
STORAGE_URL = os.getenv("GNN_OPTUNA_STORAGE")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")
if not STORAGE_URL:
    raise ValueError("GNN_OPTUNA_STORAGE not set in .env file. Copy .env.example to .env and configure it.")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres={gpu}
#SBATCH --output={log_dir}/optuna_trial_%a.out
#SBATCH --error={log_dir}/optuna_trial_%a.err
#SBATCH --array=1-{n_trials}
#SBATCH --job-name=deepset_{task_short}_{model_short}

CONDA_PATH={conda_path}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to repo directory
cd "{base_dir}"

# Add project root to Python path
export PYTHONPATH="{base_dir}:$PYTHONPATH"

# Run DeepSet Optuna trial
python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_deepset_precomputed \\
    --study_name "deepset_{task}_{model_family}_{model_size}_precomputed" \\
    --embeddings_dir "{embeddings_dir}" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --storage_url "{storage_url}" \\
    --n_trials 1 \\
    --deepset_hidden_dim 256

echo "DeepSet Optuna trial completed"
"""


def main():
    parser = argparse.ArgumentParser(description="Generate DeepSet Optuna SLURM jobs")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()),
                        help="Which models to generate jobs for (default: all)")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS,
                        help="Which tasks to generate jobs for (default: all)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                        help=f"Number of trials per study (default: {N_TRIALS})")
    args_cmd = parser.parse_args()

    jobs_dir = Path(BASE_DIR) / "scripts_and_jobs" / "slurm_jobs" / "generated_jobs" / "deepset_optuna"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    jobs_created = []

    for model_name in args_cmd.models:
        model_info = MODELS[model_name]
        model_family = model_info["family"]
        model_size = model_info["size"]
        model_time = model_info["time"]  # Model-specific time limit

        for task in args_cmd.tasks:
            # Determine embeddings directory based on model
            embeddings_dir = f"{BASE_DIR}/precomputed_embeddings/{model_family}_{model_size}_mean_pooling"

            # Create log directory
            log_dir = Path(BASE_DIR) / "job_logs" / f"logs_deepset_{task}_{model_family}_{model_size}"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Task abbreviations for job names
            task_short = task.replace("Classification", "").replace("Intent", "I").replace("Domain", "D")[:10]
            model_short = model_name.replace("-", "").replace(".", "")[:8]

            # Generate job script with model-specific time
            job_content = SLURM_TEMPLATE.format(
                time=model_time,  # Use model-specific time
                mem=SLURM_CONFIG["mem"],
                cpus=SLURM_CONFIG["cpus"],
                partition=SLURM_CONFIG["partition"],
                qos=SLURM_CONFIG["qos"],
                gpu=SLURM_CONFIG["gpu"],
                log_dir=log_dir,
                n_trials=args_cmd.n_trials,
                task_short=task_short,
                model_short=model_short,
                conda_path=CONDA_PATH,
                conda_env=CONDA_ENV,
                base_dir=BASE_DIR,
                task=task,
                model_family=model_family,
                model_size=model_size,
                embeddings_dir=embeddings_dir,
                storage_url=STORAGE_URL,
            )

            # Write job file
            job_file = jobs_dir / f"deepset_{task}_{model_family}_{model_size}.sh"
            with open(job_file, 'w') as f:
                f.write(job_content)

            # Make executable
            os.chmod(job_file, 0o755)

            jobs_created.append(str(job_file))

    print(f"\n✓ Generated {len(jobs_created)} DeepSet Optuna job files:")
    print(f"  Directory: {jobs_dir}")
    print(f"  Models: {', '.join(args_cmd.models)}")
    print(f"  Tasks: {len(args_cmd.tasks)} tasks")
    print(f"  Trials per study: {args_cmd.n_trials}")
    print(f"\nTo submit all jobs:")
    print(f"  for job in {jobs_dir}/*.sh; do sbatch \"$job\"; done")


if __name__ == "__main__":
    main()
