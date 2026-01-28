#!/usr/bin/env python3
"""
Generate SLURM jobs for evaluating trained STS models.

Usage:
    # Generate evaluation jobs for all trained models
    python3 scripts_and_jobs/generate_sts_eval_jobs.py --model Pythia-410m

    # Generate for specific encoder
    python3 scripts_and_jobs/generate_sts_eval_jobs.py --model Pythia-410m --encoder gin

    # Generate for specific task
    python3 scripts_and_jobs/generate_sts_eval_jobs.py --model Pythia-410m --task STSBenchmark
"""
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config

# Load environment
load_dotenv()

BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file")

# STS tasks that have train splits (can be used for training)
STS_TASKS = ["STSBenchmark", "STS12"]

# All STS tasks for evaluation (including test-only tasks)
ALL_STS_EVAL_TASKS = [
    "STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16", 
    "BIOSSES", "SICK-R"
]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
{cluster_sbatch_lines}
#SBATCH --output=job_logs/sts_eval/{job_name}.out
#SBATCH --error=job_logs/sts_eval/{job_name}.err
#SBATCH --job-name={job_name}

# Source conda (path from .env file)
source "{conda_path}/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to repo directory
cd "{base_dir}"

# Add project root to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "=========================================="
echo "Evaluating STS Model"
echo "=========================================="
echo "Model: {model_family}-{model_size}"
echo "Encoder: {encoder}"
echo "Tasks: {tasks}"
echo "Model path: {model_path}"
echo "Job started at: $(date)"

python3 scripts_and_jobs/scripts/eval/evaluate_sts_model.py \\
    --model_path {model_path} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --encoder {encoder} \\
    --config {config} \\
    --tasks {tasks} \\
    --output_dir results/sts_results \\
    --batch_size 64

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully"
else
    echo "✗ Evaluation failed"
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


def find_trained_models(model_family, model_size, encoder=None, task=None, v2=False):
    """Find trained STS model checkpoints.

    Args:
        model_family: Model family (e.g., "Pythia", "TinyLlama")
        model_size: Model size (e.g., "410m", "1.1B")
        encoder: Optional encoder filter (e.g., "gin", "mlp")
        task: Optional task filter (e.g., "STSBenchmark")
        v2: If True, only find models with _cayley suffix. If False, exclude _cayley models.
    """
    saved_models_dir = Path("saved_models")

    # Model naming pattern: {method}_{task}_{model_family}_{model_size}_{config}.pt
    # cayley pattern: {method}_{task}_{model_family}_{model_size}_{config}_cayley.pt
    # Example: gin_STSBenchmark_Pythia_410m_cayley.pt
    # cayley example: gin_STSBenchmark_Pythia_410m_cayley_mean_cayley.pt
    # Linear example: linear_gcn_STSBenchmark_Pythia_410m_cayley_mean.pt
    # Note: method can be "gin", "gcn", "mlp", "weighted", "deepset"
    #       or "linear_gin", "linear_gcn", "linear_mlp", "linear_deepset"

    models = []
    model_str = f"{model_family}_{model_size}"

    for model_file in saved_models_dir.glob("*.pt"):
        filename = model_file.stem

        # Skip if not an STS model (STS models have task names in them)
        if not any(sts_task in filename for sts_task in STS_TASKS):
            continue

        # Skip if not matching the model
        if model_str not in filename:
            continue

        # Filter by cayley suffix
        is_v2_model = filename.endswith("_cayley")
        if v2 and not is_v2_model:
            # cayley mode: skip non-cayley models
            continue
        elif not v2 and is_v2_model:
            # baseline mode: skip cayley models
            continue

        # Parse the filename
        # Expected: {method}_{task}_{model_family}_{model_size}_{config}
        # Or for linear: linear_{method}_{task}_{model_family}_{model_size}_{config}
        parts = filename.split("_")

        if len(parts) < 5:  # Need at least: method, task, family, size, config
            continue

        # Check if this is a linear model
        if parts[0] == "linear":
            # Linear model: linear_gcn_STSBenchmark_Pythia_410m_cayley_mean
            # cayley: linear_gcn_STSBenchmark_Pythia_410m_cayley_mean_cayley
            if len(parts) < 6:  # Need extra part for linear prefix
                continue
            file_method = f"{parts[0]}_{parts[1]}"  # e.g., "linear_gcn"
            file_task = parts[2]     # STSBenchmark, STS12, etc.
            # parts[3] = model_family (e.g., "Pythia")
            # parts[4] = model_size (e.g., "410m")
            file_config = "_".join(parts[5:])  # Everything after model_size (KEEP _cayley suffix!)
        else:
            # Non-linear model: gcn_STSBenchmark_Pythia_410m_cayley
            # cayley: gcn_STSBenchmark_Pythia_410m_cayley_mean_cayley
            file_method = parts[0]  # gin, gcn, mlp, weighted, deepset
            file_task = parts[1]     # STSBenchmark, STS12, etc.
            # parts[2] = model_family (e.g., "Pythia")
            # parts[3] = model_size (e.g., "410m")
            file_config = "_".join(parts[4:])  # Everything after model_size (KEEP _cayley suffix!)

        # Normalize encoder filter (gin includes both gin and gcn)
        if encoder:
            if encoder == "gin":
                # Accept both "gin" and "gcn" when filtering by "gin"
                # Also accept linear versions
                if file_method not in ["gin", "gcn", "linear_gin", "linear_gcn"]:
                    continue
            elif encoder == "mlp":
                if file_method not in ["mlp", "linear_mlp"]:
                    continue
            elif encoder == "deepset":
                if file_method not in ["deepset", "linear_deepset"]:
                    continue
            elif encoder == "weighted":
                # Weighted is always linear, no linear_ prefix
                if file_method != "weighted":
                    continue
            elif encoder == "dwatt":
                # DWAtt is paper-faithful, no linear_ prefix
                if file_method != "dwatt":
                    continue
            else:
                # Exact match or linear version
                if file_method != encoder and file_method != f"linear_{encoder}":
                    continue

        # Filter by task if specified
        if task and file_task != task:
            continue

        models.append({
            "path": model_file,
            "encoder": file_method,  # Store actual method (gin/gcn/mlp/weighted/linear_gin/etc.)
            "task": file_task,
            "config": file_config,
            "is_v2": is_v2_model  # Track if this is a cayley model
        })

    return models


def create_eval_job(model_family, model_size, model_info, eval_tasks, partition='priority'):
    """Create an evaluation job script."""
    encoder = model_info["encoder"]
    task = model_info["task"]
    config = model_info["config"]
    model_path = model_info["path"]

    # Create job name
    model_str = f"{model_family.lower()}{model_size.lower().replace('.', '_')}"
    if len(eval_tasks) == 1:
        # Single task - include task name in job name
        job_name = f"eval_sts_{eval_tasks[0]}_{encoder}_{model_str}_{config}"
    else:
        # Multiple tasks - use generic name
        job_name = f"eval_sts_all_{encoder}_{model_str}_{config}"

    # Evaluate on specified tasks
    tasks = " ".join(eval_tasks)

    # Get cluster configuration
    cluster_config = get_cluster_config(partition_type=partition)
    cluster_sbatch_lines = cluster_config.to_sbatch_lines()

    script = SLURM_TEMPLATE.format(
        job_name=job_name,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
        encoder=encoder,
        config=config,
        tasks=tasks,
        model_path=model_path,
        cluster_sbatch_lines=cluster_sbatch_lines
    )

    return script, job_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate STS evaluation jobs for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g., Pythia-410m, TinyLlama-1.1B)")
    parser.add_argument("--encoder", type=str,
                        help="Filter by encoder type (gin/mlp/weighted/deepset/linear_gin/linear_gcn/linear_mlp/linear_deepset, default: all)")
    parser.add_argument("--task", type=str,
                        help="Filter by training task (default: all)")
    parser.add_argument("--eval_tasks", type=str, nargs="+",
                        help="Tasks to evaluate on (default: all STS tasks). Examples: --eval_tasks STS17, --eval_tasks STSBenchmark STS12 STS13")
    parser.add_argument("--partition", type=str, choices=['priority', 'general'], default='priority',
                        help="Partition type: 'priority' for lab partition (default), 'general' for shared pool")
    parser.add_argument("--v2", action="store_true",
                        help="Evaluate cayley models only (models with _cayley suffix, trained with pool_real_nodes_only=True)")

    args = parser.parse_args()

    # Determine which tasks to evaluate on
    if args.eval_tasks:
        eval_tasks = args.eval_tasks
    else:
        # Default: evaluate on all STS tasks
        eval_tasks = ALL_STS_EVAL_TASKS

    # Parse model name
    model_family, model_size = parse_model_name(args.model)

    # Find trained models
    print(f"\n{'='*70}")
    print(f"Searching for Trained STS Models")
    print(f"{'='*70}")
    print(f"Model: {model_family}-{model_size}")
    if args.encoder:
        print(f"Encoder filter: {args.encoder}")
    if args.task:
        print(f"Task filter: {args.task}")
    print(f"{'='*70}\n")

    models = find_trained_models(model_family, model_size, args.encoder, args.task, v2=args.v2)

    if not models:
        print(f"⚠️  No trained models found matching criteria!")
        print(f"\nExpected location: saved_models/")
        print(f"Expected pattern: {{method}}_{{task}}_{model_family}_{model_size}_{{config}}.pt")
        print(f"Example: gin_STSBenchmark_Pythia_410m_cayley.pt")
        print(f"\nMake sure you have trained models first:")
        print(f"  python3 pipeline.py sts-train --model {args.model} --submit")
        return

    print(f"Found {len(models)} trained model(s):\n")
    for i, model in enumerate(models, 1):
        print(f"  [{i}] {model['encoder']:8s} | {model['task']:15s} | {model['config']}")

    # Create output directory
    output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs/sts_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = Path("job_logs/sts_eval")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Generating Evaluation Jobs")
    print(f"{'='*70}")
    print(f"Output dir: {output_dir}")
    print(f"Eval tasks: {', '.join(eval_tasks)}")
    print(f"{'='*70}\n")

    job_count = 0
    job_files = []

    for model_info in models:
        script, job_name = create_eval_job(model_family, model_size, model_info, eval_tasks, partition=args.partition)

        job_filename = f"{job_name}.sh"
        job_path = output_dir / job_filename

        with open(job_path, "w", newline="\n") as f:
            f.write(script)

        job_count += 1
        job_files.append(str(job_path))
        print(f"  [{job_count:2d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} evaluation job(s)")
    print(f"{'='*70}\n")

    # Print submission commands
    print("To submit jobs:")
    for job_file in job_files:
        print(f"  sbatch {job_file}")

    print("\nOr submit all at once:")
    print(f"  for job in {output_dir}/*.sh; do sbatch \"$job\"; done")

    print("\nTo monitor:")
    print(f"  squeue -u $USER | grep eval_sts")
    print(f"  tail -f job_logs/sts_eval/*.out")

    print("\nResults will be saved to:")
    print(f"  results/sts_results/{model_family}/{model_size}/main/mteb/{{encoder}}/{{config}}/")
    print(f"\nEach model will be evaluated on:")
    print(f"  {', '.join(eval_tasks)}")
    print(f"\nEach task will have its own JSON file (e.g., STSBenchmark.json, STS12.json)")
    print()


if __name__ == "__main__":
    main()
