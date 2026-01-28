#!/usr/bin/env python3
"""
Generate SLURM jobs for MLP baseline Optuna experiments.

Creates jobs for all combinations of:
- Models: Pythia-410m, Pythia-2.8b, Llama3-8B
- Tasks: 7 classification tasks
- Each job runs 100 Optuna trials (array=1-100, n_trials=1 per job)

Total: 3 models × 7 tasks = 21 job files (each spawns 100 array tasks)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load user configuration from .env
load_dotenv()

# Configuration
MODELS = [
    ("Pythia", "410m", "01:00:00", "48G"),
    ("Pythia", "2.8b", "02:00:00", "64G"),
    ("Llama3", "8B", "03:00:00", "80G"),
]

TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "AmazonCounterfactualClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]

# SLURM configuration (loaded from .env file)
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")
REPO_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
STORAGE_URL = os.getenv("GNN_OPTUNA_STORAGE")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")
if not STORAGE_URL:
    raise ValueError("GNN_OPTUNA_STORAGE not set in .env file. Copy .env.example to .env and configure it.")

SLURM_CONFIG = {
    "partition": "gpu-general-pool",
    "qos": "public",
    "cpus_per_task": 8,
    "gres": "gpu:A100:1",
}

# Output directory base
BASE_OUTPUT_DIR = Path("scripts_and_jobs/slurm_jobs/generated_jobs")


def get_model_dir_name(model_family, model_size):
    """Get directory name matching GIN naming convention."""
    # Convert to lowercase and format like pythia410, pythia2.8b, llama3
    family_lower = model_family.lower()
    if family_lower == "llama3":
        return f"{family_lower}_mlp_optuna"
    else:
        return f"{family_lower}{model_size}_mlp_optuna"


def create_mlp_job(model_family, model_size, task, time_limit, memory):
    """Create a single MLP Optuna SLURM job script (array job with 100 trials)."""

    job_name = f"mlp_{task}_{model_family}_{model_size}"
    study_name = f"mlp_{task}_{model_family}_{model_size}"
    log_dir = f"{REPO_DIR}/job_logs/logs_mlp_{task}_{model_family}_{model_size}"

    script = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --qos={SLURM_CONFIG['qos']}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --output={log_dir}/optuna_trial_%a.out
#SBATCH --error={log_dir}/optuna_trial_%a.err
#SBATCH --array=1-100
#SBATCH --job-name={job_name}

CONDA_PATH={CONDA_PATH}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate your specific environment
conda activate {CONDA_ENV}

# Verify the environment was activated
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to the repo directory
cd "{REPO_DIR}"

echo "Job started at: $(date)"
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Trial: $SLURM_ARRAY_TASK_ID / 100"

time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_mlp \\
    --study_name "{study_name}" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --storage_url "{STORAGE_URL}" \\
    --n_trials 1

echo "Job finished at: $(date)"
"""

    return script


def main():
    print("Generating MLP baseline Optuna SLURM jobs...\n")

    # Generate jobs
    job_count = 0
    for model_family, model_size, time_limit, memory in MODELS:
        # Create model-specific directory (matches GIN structure)
        model_dir_name = get_model_dir_name(model_family, model_size)
        model_dir = BASE_OUTPUT_DIR / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {model_dir}")

        for task in TASKS:
            job_script = create_mlp_job(
                model_family, model_size, task, time_limit, memory
            )

            # Write job file (match GIN naming: optuna_TaskName_modelsize.sh)
            job_filename = f"optuna_{task}_{model_family.lower()}{model_size}.sh"
            job_path = model_dir / job_filename

            with open(job_path, "w") as f:
                f.write(job_script)

            job_count += 1
            print(f"  [{job_count:2d}] Created: {job_path}")

    print(f"\n{'='*70}")
    print(f"Total MLP Optuna job files created: {job_count}")
    print(f"Each job spawns 100 array tasks (1 Optuna trial each)")
    print(f"Total trials: {job_count} × 100 = {job_count * 100}")
    print(f"{'='*70}\n")

    # Print submission commands
    print("To submit all MLP jobs, run:")
    for model_family, model_size, _, _ in MODELS:
        model_dir_name = get_model_dir_name(model_family, model_size)
        model_dir = BASE_OUTPUT_DIR / model_dir_name
        print(f"  # {model_family}-{model_size}")
        print(f"  for job in {model_dir}/*.sh; do sbatch $job; done")
    print()
    print("To monitor progress:")
    print(f"  squeue -u $USER | grep tom_mlp")
    print()


if __name__ == "__main__":
    main()
