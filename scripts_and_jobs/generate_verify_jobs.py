#!/usr/bin/env python3
"""
Generate SLURM jobs for verifying precomputed embeddings.

Usage:
    # Generate verification jobs for STS
    python3 generate_verify_jobs.py --model Pythia-410m --task STSBenchmark --task_type sts

    # Generate verification jobs for classification
    python3 generate_verify_jobs.py --model TinyLlama-1.1B --task EmotionClassification --task_type classification

    # Generate for all STS tasks
    python3 generate_verify_jobs.py --model Pythia-410m --all-sts-tasks

    # Generate for all classification tasks
    python3 generate_verify_jobs.py --model Pythia-410m --all-classification-tasks
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file")

STS_TASKS = ["STSBenchmark", "STS12"]
CLASSIFICATION_TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification"
]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=job_logs/verify_precomputed/{job_name}_%j.out
#SBATCH --error=job_logs/verify_precomputed/{job_name}_%j.err
#SBATCH --job-name={job_name}

# Get cluster configuration
CONDA_PATH=${{GNN_CONDA_PATH:-"$HOME/miniconda3"}}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to repo directory
cd "${{GNN_REPO_DIR:-$HOME/gnn_lbl}}"

# Add project root to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "=========================================="
echo "Verifying Precomputed Embeddings"
echo "=========================================="
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Type: {task_type}"
echo "Samples: {n_samples}"
echo "Job started at: $(date)"

python3 scripts_and_jobs/scripts/verify_precomputed_equivalence.py \\
    --task {task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --task_type {task_type} \\
    --embeddings_dir {embeddings_dir} \\
    --n_samples {n_samples}

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Verification completed successfully"
else
    echo "✗ Verification failed"
fi

exit $EXIT_CODE
"""


def parse_model_name(model_name):
    """Parse model name into family and size."""
    if "-" in model_name:
        family, size = model_name.split("-", 1)
        return family, size
    else:
        raise ValueError(f"Invalid model name: {model_name}. Expected format: Family-Size (e.g., Pythia-410m)")


def get_model_resources(model_name):
    """Get time and memory resources based on model size."""
    if "410m" in model_name:
        return "00:20:00", "16G"
    elif "1.1B" in model_name:
        return "00:30:00", "24G"
    elif "2.8b" in model_name:
        return "00:45:00", "32G"
    elif "8B" in model_name:
        return "01:00:00", "48G"
    else:
        return "00:30:00", "32G"


def create_verify_job(model_name, task, task_type, n_samples=100):
    """Create a verification job script."""
    model_family, model_size = parse_model_name(model_name)
    time_limit, memory = get_model_resources(model_name)

    # Determine embeddings directory
    if task_type == "sts":
        embeddings_dir = "precomputed_embeddings_sts"
    else:
        embeddings_dir = "precomputed_embeddings"

    # Create job name
    model_str = f"{model_family.lower()}{model_size.lower()}"
    task_str = task.lower().replace("classification", "")
    job_name = f"verify_{task_type}_{task_str}_{model_str}"

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        job_name=job_name,
        conda_env=CONDA_ENV,
        task=task,
        model_family=model_family,
        model_size=model_size,
        task_type=task_type,
        embeddings_dir=embeddings_dir,
        n_samples=n_samples
    )

    return script, job_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate verification jobs for precomputed embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., Pythia-410m, TinyLlama-1.1B)")
    parser.add_argument("--task", type=str,
                        help="Single task to verify")
    parser.add_argument("--task_type", type=str, choices=["sts", "classification"],
                        help="Task type (required if --task specified)")
    parser.add_argument("--all-sts-tasks", action="store_true",
                        help="Generate jobs for all STS tasks")
    parser.add_argument("--all-classification-tasks", action="store_true",
                        help="Generate jobs for all classification tasks")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples to verify (default: 100)")

    args = parser.parse_args()

    # Determine which tasks to generate
    tasks_to_generate = []

    if args.task:
        if not args.task_type:
            print("ERROR: --task_type required when using --task")
            parser.print_help()
            return
        tasks_to_generate.append((args.task, args.task_type))

    if args.all_sts_tasks:
        for task in STS_TASKS:
            tasks_to_generate.append((task, "sts"))

    if args.all_classification_tasks:
        for task in CLASSIFICATION_TASKS:
            tasks_to_generate.append((task, "classification"))

    if not tasks_to_generate:
        print("ERROR: Must specify --task or --all-sts-tasks or --all-classification-tasks")
        parser.print_help()
        return

    # Create output directory
    output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs/verify_precomputed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = Path("job_logs/verify_precomputed")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating Verification Jobs")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks_to_generate)}")
    print(f"Samples per task: {args.n_samples}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}\n")

    job_count = 0
    job_files = []

    for task, task_type in tasks_to_generate:
        script, job_name = create_verify_job(args.model, task, task_type, args.n_samples)

        job_filename = f"{job_name}.sh"
        job_path = output_dir / job_filename

        with open(job_path, "w", newline="\n") as f:
            f.write(script)

        job_count += 1
        job_files.append(str(job_path))
        print(f"  [{job_count:2d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} verification job(s)")
    print(f"{'='*70}\n")

    # Print submission commands
    print("To submit jobs:")
    for job_file in job_files:
        print(f"  sbatch {job_file}")

    print("\nOr submit all at once:")
    print(f"  for job in {output_dir}/*.sh; do sbatch \"$job\"; done")

    print("\nTo monitor:")
    print(f"  squeue -u $USER | grep verify")
    print(f"  tail -f job_logs/verify_precomputed/*.out")
    print()


if __name__ == "__main__":
    main()
