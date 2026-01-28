#!/usr/bin/env python3
"""
Generate SLURM jobs for Weighted baseline STS Optuna experiments.

Creates jobs for all combinations of:
- Models: Pythia-410m, Pythia-2.8b, Llama3-8B
- Tasks: 3 STS tasks (SICK-R, STSBenchmark, BIOSSES)
- Each job runs 50 Optuna trials (array=1-50, n_trials=1 per job)
  (Fewer trials than MLP because smaller hyperparameter space: just lr and weight_decay)

Total: 3 models × 3 tasks = 9 job files (each spawns 50 array tasks)
"""

from pathlib import Path

# Configuration
MODELS = [
    ("Pythia", "410m", "00:45:00", "48G"),   # Faster than MLP (fewer params to train)
    ("Pythia", "2.8b", "01:30:00", "64G"),
    ("Llama3", "8B", "02:30:00", "80G"),
]

STS_TASKS = [
    "SICK-R",
    "STSBenchmark",
    "BIOSSES",
]

# Load from environment variables (set in .env file)
import os
from dotenv import load_dotenv

load_dotenv()

CONDA_PATH = os.getenv("GNN_CONDA_PATH", "")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")
REPO_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
STORAGE_URL = os.getenv("GNN_OPTUNA_STORAGE", "")

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
    """Get directory name for STS weighted jobs."""
    family_lower = model_family.lower()
    if family_lower == "llama3":
        return f"{family_lower}_sts_weighted_optuna"
    else:
        return f"{family_lower}{model_size}_sts_weighted_optuna"


def create_weighted_sts_job(model_family, model_size, task, time_limit, memory):
    """Create a single Weighted STS Optuna SLURM job script (array job with 50 trials)."""

    job_name = f"sts_weighted_{task}_{model_family}_{model_size}"
    study_name = f"sts_weighted_{task}_{model_family}_{model_size}"
    log_dir = f"{REPO_DIR}/job_logs/logs_sts_weighted_{task}_{model_family}_{model_size}"

    script = f"""#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --qos={SLURM_CONFIG['qos']}
#SBATCH --gres={SLURM_CONFIG['gres']}
#SBATCH --output={log_dir}/optuna_trial_%a.out
#SBATCH --error={log_dir}/optuna_trial_%a.err
#SBATCH --array=1-50
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
echo "Trial: $SLURM_ARRAY_TASK_ID / 50"

time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_sts_weighted \\
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
    print("Generating Weighted baseline STS Optuna SLURM jobs...\n")

    # Generate jobs
    job_count = 0
    for model_family, model_size, time_limit, memory in MODELS:
        # Create model-specific directory
        model_dir_name = get_model_dir_name(model_family, model_size)
        model_dir = BASE_OUTPUT_DIR / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {model_dir}")

        for task in STS_TASKS:
            job_script = create_weighted_sts_job(
                model_family, model_size, task, time_limit, memory
            )

            # Write job file
            job_filename = f"optuna_{task}_{model_family.lower()}{model_size}.sh"
            job_path = model_dir / job_filename

            with open(job_path, "w") as f:
                f.write(job_script)

            job_count += 1
            print(f"  [{job_count:2d}] Created: {job_path}")

    print(f"\n{'='*70}")
    print(f"Total Weighted STS Optuna job files created: {job_count}")
    print(f"Each job spawns 50 array tasks (1 Optuna trial each)")
    print(f"Total trials: {job_count} × 50 = {job_count * 50}")
    print(f"{'='*70}\n")

    # Print submission commands
    print("To submit all Weighted STS jobs, run:")
    for model_family, model_size, _, _ in MODELS:
        model_dir_name = get_model_dir_name(model_family, model_size)
        model_dir = BASE_OUTPUT_DIR / model_dir_name
        print(f"  # {model_family}-{model_size}")
        print(f"  for job in {model_dir}/*.sh; do sbatch \"$job\"; done")
    print()
    print("To monitor progress:")
    print(f"  squeue -u $USER | grep tom_sts_weighted")
    print()
    print("\nNote: Weighted baseline has only ~24-32 learnable parameters")
    print("(one weight per layer), making it the minimal learnable baseline for STS.")


if __name__ == "__main__":
    main()
