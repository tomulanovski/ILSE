#!/usr/bin/env python3
"""
Generate SLURM jobs for STS precompute embeddings.

Usage:
    python3 generate_sts_precompute_jobs.py --models Pythia-410m --all-tasks
    python3 generate_sts_precompute_jobs.py --all-models --recommended
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config, get_cluster_type

load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "02:00:00", "48G"),
    "Pythia-410m": ("Pythia", "410m", "02:00:00", "32G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "03:00:00", "64G"),
    "Llama3-8B": ("Llama3", "8B", "04:00:00", "80G"),
}

# STS tasks - will attempt all, those without train splits will fail gracefully
ALL_STS_TASKS = [
    "STSBenchmark",  # Confirmed: has train split ✓
    "STS12",         # Confirmed: has train split ✓
    "STS13",         # Confirmed: evaluation-only (test only)
    "STS14",         # Unknown - will test
    "STS15",         # Unknown - will test
    "STS16",         # Unknown - will test
    "STS17",         # Unknown - will test
    "STS22",         # Unknown - will test
    "BIOSSES",       # Confirmed: evaluation-only (test only)
    "SICK-R",        # Confirmed: evaluation-only (test only)
]

RECOMMENDED_STS_TASKS = ["STSBenchmark", "STS12"]  # Confirmed trainable - for quick tests

# Paths
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")

if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
{cluster_sbatch_lines}
#SBATCH --output={log_dir}/precompute.out
#SBATCH --error={log_dir}/precompute.err
#SBATCH --job-name=sts_precomp_{task_short}

CONDA_PATH={conda_path}

source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate {conda_env}

echo "Current conda environment: $CONDA_PREFIX"
nvidia-smi

cd "{base_dir}"
export PYTHONPATH="{base_dir}:$PYTHONPATH"

echo "=========================================="
echo "STS Precompute Embeddings"
echo "=========================================="
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Job started at: $(date)"

python3 -m experiments.utils.precompute.precompute_sts \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --output_dir "./precomputed_embeddings_sts" \\
    --batch_size 256

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ STS precompute for {task} completed successfully"
else
    echo "✗ STS precompute for {task} failed"
fi

exit $EXIT_CODE
"""


def create_sts_precompute_job(model_name, task, partition_type='priority'):
    """Create a single STS precompute SLURM job script."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    model_family, model_size, time_limit, memory = MODEL_CONFIGS[model_name]

    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    task_short = task.lower().replace("benchmark", "").replace("multilingual", "multi")[:15]
    log_dir = f"{BASE_DIR}/job_logs/sts_precompute_{task}_{model_family}_{model_size}"

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        cluster_sbatch_lines=cluster_config.to_sbatch_lines(),
        log_dir=log_dir,
        task=task,
        task_short=task_short,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
    )

    return script


def main():
    parser = argparse.ArgumentParser(description="Generate STS precompute jobs")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=ALL_STS_TASKS)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--recommended", action="store_true")
    parser.add_argument("--partition", choices=['priority', 'general'], default='priority')

    args = parser.parse_args()

    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    elif args.models:
        models = args.models
    else:
        print("ERROR: Must specify --models or --all-models")
        return

    if args.all_tasks:
        tasks = ALL_STS_TASKS
    elif args.recommended:
        tasks = RECOMMENDED_STS_TASKS
    elif args.tasks:
        tasks = args.tasks
    else:
        print("ERROR: Must specify --tasks, --all-tasks, or --recommended")
        return

    cluster = get_cluster_type()

    print(f"\n{'='*70}")
    print(f"Generating STS Precompute jobs")
    print(f"{'='*70}")
    print(f"Cluster: {cluster.upper()}")
    print(f"Models: {', '.join(models)}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Total jobs: {len(models) * len(tasks)}")
    print(f"{'='*70}\n")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        if "-" in model_name:
            family, size = model_name.split("-", 1)
            model_str = f"{family.lower()}{size.lower()}"
        else:
            model_str = model_name.lower()

        job_dir_name = f"{model_str}_sts_precompute"
        job_dir = base_output_dir / job_dir_name
        job_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{model_name} (STS Precompute):")
        print(f"  Output dir: {job_dir}")

        for task in tasks:
            job_script = create_sts_precompute_job(model_name, task, partition_type=args.partition)

            job_filename = f"sts_precompute_{task}_{model_name.replace('-', '').lower()}.sh"
            job_path = job_dir / job_filename

            with open(job_path, "w", newline="\n") as f:
                f.write(job_script)

            job_count += 1
            print(f"    [{job_count:3d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} STS precompute job files")
    print(f"{'='*70}\n")

    print("To submit STS precompute jobs:")
    for model_name in models:
        if "-" in model_name:
            family, size = model_name.split("-", 1)
            model_str = f"{family.lower()}{size.lower()}"
        else:
            model_str = model_name.lower()
        job_dir_name = f"{model_str}_sts_precompute"
        job_dir = base_output_dir / job_dir_name
        print(f"  # {model_name}")
        print(f"  for job in {job_dir}/sts_precompute_*.sh; do sbatch \"$job\"; done")

    print()


if __name__ == "__main__":
    main()
