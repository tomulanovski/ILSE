#!/usr/bin/env python3
"""
Unified SLURM job generator for precomputing embeddings.

Supports all models: TinyLlama-1.1B, Pythia-410m, Pythia-2.8b, Llama3-8B
Can be called directly or via pipeline.py
Works with both Bio and CS clusters (configured via .env)

Usage:
    # Generate for specific model(s) and task(s)
    python3 generate_precompute_jobs.py --models Pythia-410m --tasks EmotionClassification

    # Generate for all tasks on one model
    python3 generate_precompute_jobs.py --models TinyLlama-1.1B --all-tasks

    # Generate for specific partition (priority or general)
    python3 generate_precompute_jobs.py --models Pythia-410m --all-tasks --partition priority
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config, get_cluster_type

# Load user configuration from .env
load_dotenv()

# Model configurations: (family, size, time_limit, memory)
# Cluster-specific settings (partition, account, qos, gpu) come from cluster_config
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "01:30:00", "64G"),
    "Pythia-410m": ("Pythia", "410m", "01:30:00", "64G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "02:30:00", "80G"),
    "Llama3-8B": ("Llama3", "8B", "04:00:00", "96G"),
}

# All classification tasks (7 tasks - Amazon removed)
ALL_TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]

# Paths (loaded from .env file)
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
{cluster_sbatch_lines}
#SBATCH --output={log_dir}/precompute_{task}.out
#SBATCH --error={log_dir}/precompute_{task}.err
#SBATCH --job-name={job_name}

CONDA_PATH={conda_path}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Help with PyTorch memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to repo directory
cd "{base_dir}"

# Add project root to Python path
export PYTHONPATH="{base_dir}:$PYTHONPATH"

echo "=========================================="
echo "Precomputing Embeddings"
echo "=========================================="
echo "Model: {model_family}-{model_size}"
echo "Task: {task}"
echo "Job started at: $(date)"

time python3 -m experiments.utils.precompute.precompute_pipeline \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --tasks {task} \\
    --output_dir ./precomputed_embeddings \\
    --pooling_method mean \\
    --batch_size 32 \\
    --chunk_size 5000

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Embedding precomputation SUCCESSFUL"
    echo "  Output: precomputed_embeddings/{model_family}_{model_size}_mean_pooling/{task}/"
else
    echo "✗ Embedding precomputation FAILED"
fi

exit $EXIT_CODE
"""


def get_model_dir_name(model_name):
    """Get directory name for job files."""
    # TinyLlama-1.1B -> tinyllama1.1b_precompute
    # Pythia-410m -> pythia410m_precompute
    # Pythia-2.8b -> pythia2.8b_precompute
    # Keep dots in version number, remove only dashes
    return model_name.replace("-", "").lower() + "_precompute"


def create_precompute_job(model_name, task, partition_type='priority'):
    """Create a single precompute SLURM job script."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    model_family, model_size, time_limit, memory = MODEL_CONFIGS[model_name]

    # Get cluster-specific SLURM configuration
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Shorten task name for job name (SLURM has limits)
    task_short = task.replace("Classification", "").lower()[:10]
    job_name = f"precomp_{model_family.lower()}_{task_short}"

    log_dir = f"{BASE_DIR}/job_logs/{get_model_dir_name(model_name)}"

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        cluster_sbatch_lines=cluster_config.to_sbatch_lines(),
        log_dir=log_dir,
        task=task,
        job_name=job_name,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
    )

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for precomputing embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model, single task
    python3 generate_precompute_jobs.py --models Pythia-410m --tasks EmotionClassification

    # Single model, all tasks
    python3 generate_precompute_jobs.py --models TinyLlama-1.1B --all-tasks

    # Multiple models, all tasks
    python3 generate_precompute_jobs.py --models Pythia-410m Pythia-2.8b --all-tasks

    # All models, specific tasks
    python3 generate_precompute_jobs.py --all-models --tasks EmotionClassification Banking77Classification
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to generate jobs for"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Generate for all supported models"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_TASKS,
        help="Tasks to generate jobs for"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Generate for all classification tasks"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition, 'general' for shared (default: priority)"
    )

    args = parser.parse_args()

    # Determine models and tasks
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    elif args.models:
        models = args.models
    else:
        print("ERROR: Must specify --models or --all-models")
        parser.print_help()
        return

    if args.all_tasks:
        tasks = ALL_TASKS
    elif args.tasks:
        tasks = args.tasks
    else:
        print("ERROR: Must specify --tasks or --all-tasks")
        parser.print_help()
        return

    # Get cluster info
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=args.partition)

    print(f"\n{'='*70}")
    print(f"Generating precompute jobs")
    print(f"{'='*70}")
    print(f"Cluster: {cluster.upper()}")
    print(f"Partition type: {args.partition}")
    print(f"  Partition: {cluster_config.partition}")
    if cluster_config.account:
        print(f"  Account: {cluster_config.account}")
    if cluster_config.qos:
        print(f"  QOS: {cluster_config.qos}")
    print(f"Models: {', '.join(models)}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Total jobs: {len(models) * len(tasks)}")
    print(f"{'='*70}\n")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        model_dir_name = get_model_dir_name(model_name)
        model_dir = base_output_dir / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{model_name}:")
        print(f"  Output dir: {model_dir}")

        for task in tasks:
            job_script = create_precompute_job(model_name, task, partition_type=args.partition)

            # Write job file (keep dots in version number, remove only dashes)
            job_filename = f"precompute_{task}_{model_name.replace('-', '').lower()}.sh"
            job_path = model_dir / job_filename

            with open(job_path, "w", newline="\n") as f:
                f.write(job_script)

            job_count += 1
            print(f"    [{job_count:2d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} precompute job files")
    print(f"{'='*70}\n")

    # Print instructions
    print("NEXT STEPS:\n")
    print("1. Create log directories:")
    for model_name in models:
        log_dir = f"{BASE_DIR}/job_logs/{get_model_dir_name(model_name)}"
        print(f"   mkdir -p {log_dir}")

    print("\n2. Submit jobs:")
    for model_name in models:
        model_dir_name = get_model_dir_name(model_name)
        model_dir = base_output_dir / model_dir_name
        print(f"   # {model_name}")
        print(f"   for job in {model_dir}/precompute_*.sh; do sbatch \"$job\"; done")

    print("\n3. Monitor progress:")
    print(f"   squeue -u $USER | grep precomp")

    print("\n4. After precompute completes, check outputs:")
    for model_name in models:
        model_family, model_size, *_ = MODEL_CONFIGS[model_name]
        print(f"   ls precomputed_embeddings/{model_family}_{model_size}_mean_pooling/")

    print()


if __name__ == "__main__":
    main()
