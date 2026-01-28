#!/usr/bin/env python3
"""
Generate SLURM job files for Optuna trials on precomputed embeddings.
One job per task.
"""
import argparse
from pathlib import Path
from typing import List


def extract_model_name(embeddings_dir: str) -> str:
    """
    Extract model name from embeddings directory path.
    E.g., "./precomputed_embeddings/Llama3_8B_mean_pooling" -> "Llama_8B"
    """
    path = Path(embeddings_dir)
    dir_name = path.name  # "Llama3_8B_mean_pooling"

    # Remove common suffixes
    for suffix in ["_mean_pooling", "_last_pooling", "_first_pooling"]:
        if dir_name.endswith(suffix):
            dir_name = dir_name[:-len(suffix)]
            break

    return dir_name


def generate_job_content(
    task: str,
    encoder: str,
    study_name: str,
    embeddings_dir: str,
    compute_server: str,
    job_name: str
) -> str:
    """
    Generate SLURM job script content for a single task.
    """

    script = f"""#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu-general-pool
#SBATCH --account=your-account
#SBATCH --gres=gpu:A100:1
#SBATCH --qos=public
#SBATCH --output=job_logs/logs_optuna_gin_precomputed/{job_name}_%A_%a.out
#SBATCH --error=job_logs/logs_optuna_gin_precomputed/{job_name}_%A_%a.err
#SBATCH --array=1-100


CONDA_PATH=${GNN_REPO_DIR}/miniconda3

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Check if conda is properly sourced
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda command not found. Check your conda installation path."
    exit 1
fi

# Activate your specific environment
conda activate lbl || {{
    echo "ERROR: Could not activate lbl env"
    exit 1
}}

# Verify the environment was activated
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to the repo directory (adjust as needed)
cd "${GNN_REPO_DIR}/llm_gnn_proj/gnn_lbl"

# Random sleep to stagger job starts (reduces duplicate sampling)
SLEEP_TIME=$((RANDOM % 30 + 1))
echo "Sleeping for ${{SLEEP_TIME}} seconds to stagger job start..."
sleep ${{SLEEP_TIME}}

# ============================================================================
# CONFIGURATION
# ============================================================================
STUDY_NAME="{study_name}"
EMBEDDINGS_DIR="{embeddings_dir}"
TASK="{task}"
COMPUTE_SERVER="{compute_server}"
ENCODER="{encoder}"
MLP_LAYER_IDX=""  # Empty for searching over 'last' and 'mean'

# GIN hyperparameter ranges (only used when ENCODER="gin")
GIN_LAYERS_MIN=1
GIN_LAYERS_MAX=2
GIN_MLP_LAYERS_MIN=0
GIN_MLP_LAYERS_MAX=2
GIN_HIDDEN_DIMS="256"

# ============================================================================
# RUN
# ============================================================================

echo "Job started at: $(date)"
echo "Study: ${{STUDY_NAME}}"
echo "Embeddings: ${{EMBEDDINGS_DIR}}"
echo "Task: ${{TASK}}"
echo "Compute server: ${{COMPUTE_SERVER}}"
echo "Encoder: ${{ENCODER}}"
echo "MLP layer idx: ${{MLP_LAYER_IDX}}"

# Create log directory if it doesn't exist
mkdir -p job_logs/logs_optuna_gin_precomputed

# Build command with conditional parameters
CMD="python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_gin_precomputed \\
    --study_name ${{STUDY_NAME}} \\
    --embeddings_dir ${{EMBEDDINGS_DIR}} \\
    --task ${{TASK}} \\
    --compute_server ${{COMPUTE_SERVER}} \\
    --encoder ${{ENCODER}} \\
    --gin_layers_min ${{GIN_LAYERS_MIN}} \\
    --gin_layers_max ${{GIN_LAYERS_MAX}} \\
    --gin_mlp_layers_min ${{GIN_MLP_LAYERS_MIN}} \\
    --gin_mlp_layers_max ${{GIN_MLP_LAYERS_MAX}} \\
    --gin_hidden_dims ${{GIN_HIDDEN_DIMS}}"

if [ -n "$MLP_LAYER_IDX" ]; then
    CMD="${{CMD}} --mlp_layer_idx ${{MLP_LAYER_IDX}}"
fi

echo "Running: ${{CMD}}"
time eval ${{CMD}}

echo "Job finished at: $(date)"
"""

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job files for Optuna trials on precomputed embeddings"
    )

    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings (e.g., ./precomputed_embeddings/Llama3_8B_mean_pooling)")
    parser.add_argument("--compute_server", type=str, required=True,
                        help="Compute server name (e.g., your-compute-node)")
    parser.add_argument("--encoder", type=str, required=True, choices=["gin", "mlp"],
                        help="Encoder type: gin or mlp")
    parser.add_argument("--base_study_name", type=str, required=True,
                        help="Base study name (e.g., gin_Llama_precomputed)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of tasks")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for job files (default: auto-generated from embeddings_dir)")

    args = parser.parse_args()

    # Extract model name and create output directory
    model_name = extract_model_name(args.embeddings_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"{model_name}_optuna_jobs")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GENERATING OPTUNA JOB FILES")
    print("="*70)
    print(f"Model name: {model_name}")
    print(f"Embeddings dir: {args.embeddings_dir}")
    print(f"Compute server: {args.compute_server}")
    print(f"Encoder: {args.encoder}")
    print(f"Base study name: {args.base_study_name}")
    print(f"Tasks: {args.tasks}")
    print(f"Output directory: {output_dir}")
    print("="*70)

    # Generate job file for each task
    generated_files = []

    for task in args.tasks:
        # Create study name for this task
        study_name = f"{args.base_study_name}_{task}"

        # Create job name
        job_name = f"optuna_precomputed_{args.encoder}_{task}"

        # Generate job content
        job_content = generate_job_content(
            task=task,
            encoder=args.encoder,
            study_name=study_name,
            embeddings_dir=args.embeddings_dir,
            compute_server=args.compute_server,
            job_name=job_name
        )

        # Write to file
        job_file = output_dir / f"{job_name}.sbatch"
        with open(job_file, 'w') as f:
            f.write(job_content)

        generated_files.append(job_file)
        print(f"✓ Generated: {job_file}")

    print("\n" + "="*70)
    print(f"GENERATED {len(generated_files)} JOB FILES")
    print("="*70)
    print(f"Location: {output_dir.absolute()}")
    print("\nTo submit all jobs, run:")
    print(f"  cd {output_dir}")
    print(f"  for job in *.sbatch; do sbatch $job; done")
    print("="*70)


if __name__ == "__main__":
    main()
