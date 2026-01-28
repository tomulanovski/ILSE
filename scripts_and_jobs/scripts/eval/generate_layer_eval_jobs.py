#!/usr/bin/env python3
"""
Generate and submit SLURM jobs to evaluate each layer of a model on multiple tasks.
"""
import argparse
import subprocess
from pathlib import Path
from typing import List
import time


# Model specifications (num_layers per model)
MODEL_SPECS = {
    'Pythia_14m': 7,
    'Pythia_70m': 7,
    'Pythia_160m': 13,    
    'Pythia_410m': 25,
    'Pythia_1b': 17,
    'Pythia_1.4b': 25,
    'Pythia_2.8b': 33,
    'Llama3_8B': 33,
}


def generate_layer_eval_job(
    model_name: str,
    task: str,
    layer_idx: int,
    embeddings_dir: str,
    output_base_dir: str,
    job_logs_dir: str,
    batch_size: int = 32,
    time_limit: str = "01:00:00",
    mem: str = "16G",
    cpus: int = 4
) -> str:
    """
    Generate SLURM job script content for evaluating a specific layer.

    Returns:
        Job script content as string
    """

    output_dir = Path(output_base_dir) / model_name / task
    output_file = output_dir / f"layer_{layer_idx}.json"

    job_name = f"eval_{model_name}_{task}_L{layer_idx}"
    log_file = Path(job_logs_dir) / f"{job_name}.out"
    err_file = Path(job_logs_dir) / f"{job_name}.err"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition=gpu-general-pool
#SBATCH --account=your-account
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=public
#SBATCH --output={log_file}
#SBATCH --error={err_file}

CONDA_PATH=${GNN_REPO_DIR}/miniconda3

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate lbl || {{
    echo "ERROR: Could not activate lbl env"
    exit 1
}}

# Change to repo directory
cd "${GNN_REPO_DIR}/llm_gnn_proj/gnn_lbl"

echo "Job started at: $(date)"
echo "Evaluating {model_name} - {task} - Layer {layer_idx}"

# Run evaluation
python3 -m scripts_and_jobs.scripts.eval.evaluate_embeddings_on_test \\
    --embeddings_dir {embeddings_dir} \\
    --task {task} \\
    --embedding_type layer \\
    --layer_idx {layer_idx} \\
    --batch_size {batch_size} \\
    --output_results {output_file}

echo "Job finished at: $(date)"
"""

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate and submit layer evaluation jobs"
    )

    parser.add_argument("--model_name", type=str, required=True,
                        choices=list(MODEL_SPECS.keys()),
                        help="Model name (e.g., Pythia_410m)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of tasks to evaluate")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings")
    parser.add_argument("--output_base_dir", type=str, default="./layer_eval_res",
                        help="Base directory for results")
    parser.add_argument("--job_logs_dir", type=str, default="./job_logs/layer_eval",
                        help="Directory for job logs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--time_limit", type=str, default="01:00:00",
                        help="SLURM time limit")
    parser.add_argument("--mem", type=str, default="16G",
                        help="Memory per job")
    parser.add_argument("--cpus", type=int, default=4,
                        help="CPUs per job")
    parser.add_argument("--dry_run", action="store_true",
                        help="Generate scripts but don't submit jobs")
    parser.add_argument("--scripts_dir", type=str, default="./temp_job_scripts",
                        help="Directory to save generated job scripts")

    args = parser.parse_args()

    # Get number of layers for model
    num_layers = MODEL_SPECS[args.model_name]

    # Create output directories
    output_base = Path(args.output_base_dir)
    job_logs = Path(args.job_logs_dir)
    scripts_dir = Path(args.scripts_dir)

    job_logs.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LAYER EVALUATION JOB GENERATION")
    print("=" * 70)
    print(f"Model: {args.model_name} ({num_layers} layers)")
    print(f"Tasks: {args.tasks}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Output dir: {args.output_base_dir}")
    print(f"Total jobs: {num_layers * len(args.tasks)}")
    print("=" * 70)

    submitted_jobs = []

    # Generate and submit jobs
    for task in args.tasks:
        print(f"\nProcessing task: {task}")

        # Create task output directory
        task_output_dir = output_base / args.model_name / task
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in range(num_layers):
            # Generate job script
            job_content = generate_layer_eval_job(
                model_name=args.model_name,
                task=task,
                layer_idx=layer_idx,
                embeddings_dir=args.embeddings_dir,
                output_base_dir=args.output_base_dir,
                job_logs_dir=args.job_logs_dir,
                batch_size=args.batch_size,
                time_limit=args.time_limit,
                mem=args.mem,
                cpus=args.cpus
            )

            # Save script to file
            script_file = scripts_dir / f"{args.model_name}_{task}_layer{layer_idx}.sbatch"
            with open(script_file, 'w') as f:
                f.write(job_content)

            if not args.dry_run:
                # Submit job
                try:
                    result = subprocess.run(
                        ['sbatch', str(script_file)],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    job_id = result.stdout.strip().split()[-1]
                    submitted_jobs.append((task, layer_idx, job_id))
                    print(f"  Layer {layer_idx:2d}: Submitted job {job_id}")
                except subprocess.CalledProcessError as e:
                    print(f"  Layer {layer_idx:2d}: Failed to submit - {e}")
            else:
                print(f"  Layer {layer_idx:2d}: Script generated at {script_file}")

            time.sleep(2)

    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN - Scripts generated but not submitted")
        print(f"Scripts saved in: {scripts_dir}")
    else:
        print(f"SUBMITTED {len(submitted_jobs)} JOBS")
        print(f"Results will be saved in: {output_base / args.model_name}")
        print(f"Job logs in: {job_logs}")
    print("=" * 70)

    # Save job list
    if not args.dry_run:
        job_list_file = output_base / args.model_name / "submitted_jobs.txt"
        with open(job_list_file, 'w') as f:
            f.write("task,layer,job_id\n")
            for task, layer, job_id in submitted_jobs:
                f.write(f"{task},{layer},{job_id}\n")
        print(f"\nJob list saved to: {job_list_file}")


if __name__ == "__main__":
    main()
