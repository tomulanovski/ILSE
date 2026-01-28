#!/usr/bin/env python3
"""
Generate SLURM job files to evaluate trained models using precomputed embeddings.

This script creates jobs that use flexible_embedding_evaluator.py for MTEB-style
evaluation without loading LLMs (faster evaluation using cached embeddings).

Usage:
    python3 generate_precomputed_eval_jobs.py --dry_run  # Preview
    python3 generate_precomputed_eval_jobs.py            # Generate jobs
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re

# Add parent directory to path for cluster_config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cluster_config import get_cluster_config

# Get cluster-specific paths from environment
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
REPO_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())

if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")

# Default classification tasks (exclude Massive tasks which are slow/problematic)
DEFAULT_CLASSIFICATION_TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]


def parse_model_filename(filename: str) -> Optional[Dict]:
    """
    Parse model filename to extract metadata.

    Expected formats:
        GIN/GCN: {method}_{task}_{model_family}_{model_size}_{graph_type}.pt
        MLP: {method}_{task}_{model_family}_{model_size}_{mlp_input}_layers{N}.pt
        Weighted: {method}_{task}_{model_family}_{model_size}_softmax.pt

    Examples:
        gin_EmotionClassification_Pythia_410m_cayley.pt
        gcn_Banking77Classification_Pythia_2.8b_linear.pt
        mlp_EmotionClassification_Pythia_410m_last_layers1.pt
        weighted_EmotionClassification_Pythia_410m_softmax.pt

    Returns:
        Dict with keys: method, task, model_family, model_size, graph_type/config
    """
    if not filename.endswith('.pt'):
        return None

    # Remove .pt extension
    base = filename[:-3]

    # First, extract method (first part before first underscore)
    parts = base.split('_')
    method = parts[0]
    if method not in ['gin', 'gcn', 'mlp', 'lora', 'weighted']:
        print(f"  ⚠️ Skipping {filename} - unknown method '{method}'")
        return None

    # Remove method from the string
    remaining = base[len(method)+1:]  # +1 for the underscore

    # Known graph types (including multi-part ones)
    known_graph_types = ['virtual_node', 'fully_connected', 'linear', 'cayley', 'cayley']
    known_mlp_inputs = ['last', 'mean', 'flatten']
    known_weighted_types = ['softmax']

    # For MLP models, check for _layers{N} suffix first
    mlp_layers = None
    if method == 'mlp':
        layers_match = re.search(r'_layers(\d+)$', remaining)
        if layers_match:
            mlp_layers = layers_match.group(1)
            remaining = remaining[:layers_match.start()]

    # Try to match config from the end
    config = None
    for pattern in known_graph_types + known_mlp_inputs + known_weighted_types:
        if remaining.endswith('_' + pattern):
            config = pattern
            remaining = remaining[:-(len(pattern)+1)]
            break

    if config is None:
        print(f"  ⚠️ Skipping {filename} - couldn't identify config")
        return None

    # Now remaining should be: {task}_{model_family}_{model_size}
    known_families = ['Pythia', 'Llama3', 'TinyLlama', 'BERT', 'Llama']

    parts = remaining.split('_')

    # Find model_family
    model_family = None
    model_family_idx = None
    for i, part in enumerate(parts):
        if part in known_families:
            model_family = part
            model_family_idx = i
            break

    if model_family is None or model_family_idx >= len(parts) - 1:
        print(f"  ⚠️ Skipping {filename} - couldn't identify model_family")
        return None

    # model_size is right after model_family
    model_size = parts[model_family_idx + 1]

    # task is everything before model_family
    task = '_'.join(parts[:model_family_idx])

    if not task:
        print(f"  ⚠️ Skipping {filename} - empty task name")
        return None

    result = {
        'method': method,
        'task': task,
        'model_family': model_family,
        'model_size': model_size,
        'config': config,
        'filename': filename
    }

    if mlp_layers is not None:
        result['mlp_layers'] = mlp_layers

    return result


def find_all_models(models_dir: str) -> List[Dict]:
    """
    Scan models directory and parse all model files.

    Args:
        models_dir: Path to saved_models directory

    Returns:
        List of model metadata dicts
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"Error: Model directory not found: {models_dir}")
        return []

    model_files = list(models_path.glob('*.pt'))
    print(f"Found {len(model_files)} .pt files in {models_dir}")

    models = []
    for model_file in model_files:
        metadata = parse_model_filename(model_file.name)
        if metadata:
            metadata['full_path'] = str(model_file.absolute())
            models.append(metadata)

    print(f"Parsed {len(models)} valid model files")
    return models


def generate_eval_job(
    model_info: Dict,
    embeddings_base_dir: str,
    output_dir: str,
    repo_root: str,
    partition_type: str = 'priority'
) -> str:
    """
    Generate a SLURM job file for evaluating a model using precomputed embeddings.

    Args:
        model_info: Model metadata dict
        embeddings_base_dir: Base directory for precomputed embeddings
        output_dir: Directory to save job files
        repo_root: Root directory of the repository
        partition_type: 'priority' or 'general'

    Returns:
        Path to generated job file
    """
    # Get cluster-specific configuration
    cluster_config = get_cluster_config(partition_type=partition_type)
    slurm_params = cluster_config.to_slurm_dict()
    method = model_info['method']
    task = model_info['task']
    model_family = model_info['model_family']
    model_size = model_info['model_size']
    config = model_info['config']
    model_path = model_info['full_path']
    mlp_layers = model_info.get('mlp_layers', None)

    # Determine embedding type based on method
    if method in ('gin', 'gcn'):
        embedding_type = 'gin'
    elif method == 'mlp':
        embedding_type = 'mlp'
    elif method == 'weighted':
        embedding_type = 'weighted'
    else:
        print(f"  ⚠️ Skipping {method} - unsupported method for precomputed evaluation")
        return None

    # Embeddings directory path
    # Format: precomputed_embeddings/{ModelFamily}_{ModelSize}_mean_pooling/{TaskName}/
    # Note: precompute step creates directories with _mean_pooling suffix
    embeddings_dir = f"{embeddings_base_dir}/{model_family}_{model_size}_mean_pooling"

    # Job filename
    if mlp_layers:
        job_filename = f"eval_precomp_{method}_{task}_{model_family}_{model_size}_{config}_layers{mlp_layers}.sh"
        config_suffix = f"{config}_layers{mlp_layers}"
    else:
        job_filename = f"eval_precomp_{method}_{task}_{model_family}_{model_size}_{config}.sh"
        config_suffix = config

    job_path = os.path.join(output_dir, job_filename)

    # Results directory
    results_dir = f"mteb_results_precomputed/{method}/{model_family}_{model_size}/{config_suffix}"
    results_file = f"{results_dir}/{task}.json"

    # Build SLURM directives
    slurm_directives = f"""#!/bin/bash
#SBATCH --job-name=eval_pc_{method}_{task[:10]}
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition={slurm_params['partition']}
#SBATCH --gres={slurm_params['gres']}"""

    # Add optional SLURM parameters if present
    if 'account' in slurm_params:
        slurm_directives += f"\n#SBATCH --account={slurm_params['account']}"
    if 'qos' in slurm_params:
        slurm_directives += f"\n#SBATCH --qos={slurm_params['qos']}"

    slurm_directives += f"""
#SBATCH --output=job_logs/eval_precomputed/{method}_{task}_{model_family}_{model_size}_{config_suffix}.out
#SBATCH --error=job_logs/eval_precomputed/{method}_{task}_{model_family}_{model_size}_{config_suffix}.err
"""

    # SLURM script content
    script = slurm_directives + f"""
# Set environment paths
CONDA_PATH={CONDA_PATH}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate final_project || {{
    echo "ERROR: Could not activate final_project env"
    exit 1
}}

echo "Current conda environment: $CONDA_PREFIX"

# Print GPU info
nvidia-smi

# Go to repo directory
cd "{repo_root}"

echo "=========================================="
echo "Evaluating Model with Precomputed Embeddings"
echo "=========================================="
echo "Method: {method.upper()}"
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Configuration: {config_suffix}"
echo "Model Path: {model_path}"
echo "Embeddings Dir: {embeddings_dir}"
echo "Started at: $(date)"
echo "=========================================="

# Create results directory
mkdir -p {results_dir}

# Run evaluation with flexible_embedding_evaluator.py
python3 -m scripts_and_jobs.scripts.eval.flexible_embedding_evaluator \\
    --embeddings_dir "{embeddings_dir}" \\
    --task "{task}" \\
    --embedding_type "{embedding_type}" \\
    --model_path "{model_path}" \\
    --batch_size 32 \\
    --output_results "{results_file}"

EXIT_CODE=$?

echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: {results_file}"
echo "=========================================="

exit $EXIT_CODE
"""

    with open(job_path, 'w') as f:
        f.write(script)

    # Make executable
    os.chmod(job_path, 0o755)

    return job_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs to evaluate trained models using precomputed embeddings"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="saved_models",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--embeddings_base_dir",
        type=str,
        default="precomputed_embeddings",
        help="Base directory for precomputed embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts_and_jobs/slurm_jobs/generated_jobs/eval_precomputed",
        help="Directory to save generated job files"
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=REPO_DIR,
        help="Root directory of the repository (for cluster)"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition (default), 'general' for shared pool"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without generating files"
    )
    parser.add_argument(
        "--filter_task",
        type=str,
        default=None,
        help="Only generate jobs for specific task"
    )
    parser.add_argument(
        "--filter_method",
        type=str,
        default=None,
        help="Only generate jobs for specific method (gin/gcn/mlp/weighted/deepset)"
    )
    parser.add_argument(
        "--filter_model",
        type=str,
        default=None,
        help="Only generate jobs for specific model (e.g., Pythia_410m)"
    )
    parser.add_argument(
        "--filter_graph_type",
        type=str,
        default=None,
        help="Only generate jobs for specific graph type (e.g., cayley, virtual_node)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to evaluate (default: 5 classification tasks excluding Massive)"
    )

    args = parser.parse_args()

    print("="*70)
    print("Generate SLURM Jobs for Precomputed Embeddings Evaluation")
    print("="*70)

    # Find all models
    models = find_all_models(args.models_dir)

    if not models:
        print("\nNo models found!")
        return

    # Filter to only supported methods
    models = [m for m in models if m['method'] in ('gin', 'gcn', 'mlp', 'weighted', 'deepset')]

    # Apply filters
    if args.filter_task:
        models = [m for m in models if m['task'] == args.filter_task]
        print(f"\nFiltered to task: {args.filter_task}")

    if args.filter_method:
        models = [m for m in models if m['method'] == args.filter_method]
        print(f"\nFiltered to method: {args.filter_method}")

    if args.filter_model:
        model_family, model_size = args.filter_model.split("_")
        models = [m for m in models if m['model_family'] == model_family and m['model_size'] == model_size]
        print(f"\nFiltered to model: {args.filter_model}")

    if args.filter_graph_type:
        # For GIN/GCN, filter by config (graph_type)
        # For MLP/Weighted, skip this filter (they don't have graph types)
        models = [m for m in models if m['method'] in ('mlp', 'weighted') or m['config'] == args.filter_graph_type]
        print(f"\nFiltered to graph_type: {args.filter_graph_type}")

    # Filter by tasks (default: only 5 classification tasks, exclude Massive)
    if args.tasks:
        # User specified tasks explicitly
        models = [m for m in models if m['task'] in args.tasks]
        print(f"\nFiltered to tasks: {', '.join(args.tasks)}")
    else:
        # Default: only include the 5 standard classification tasks (exclude Massive)
        models = [m for m in models if m['task'] in DEFAULT_CLASSIFICATION_TASKS]
        print(f"\nFiltered to default tasks (5 classification tasks, excluding Massive)")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Found {len(models)} models to evaluate")
    print(f"{'='*70}")

    # Group by task for better display
    tasks = {}
    for model in models:
        task = model['task']
        if task not in tasks:
            tasks[task] = []
        tasks[task].append(model)

    for task, task_models in sorted(tasks.items()):
        print(f"\n{task}: ({len(task_models)} models)")
        for m in task_models:
            config = m['config']
            if 'mlp_layers' in m:
                config += f"_layers{m['mlp_layers']}"
            print(f"  {m['method']:3s} | {m['model_family']}-{m['model_size']:5s} | {config}")

    # Generate job files or just print
    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - No files generated")
        print(f"{'='*70}")
        print(f"\nWould generate {len(models)} job files in:")
        print(f"  {args.output_dir}")
        print(f"\nEmbeddings would be loaded from:")
        print(f"  {args.embeddings_base_dir}")
    else:
        print(f"\n{'='*70}")
        print("Generating job files...")
        print(f"{'='*70}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Create log directory
        os.makedirs("job_logs/eval_precomputed", exist_ok=True)

        # Generate each job file
        generated = 0
        for model in models:
            job_file = generate_eval_job(
                model,
                args.embeddings_base_dir,
                args.output_dir,
                args.repo_root,
                partition_type=args.partition
            )
            if job_file:
                print(f"  ✓ {os.path.basename(job_file)}")
                generated += 1

        print(f"\n{'='*70}")
        print(f"Generated {generated} job files successfully!")
        print(f"{'='*70}")
        print(f"\nTo submit all jobs:")
        print(f"  cd {args.repo_root}")
        print(f"  for job in {args.output_dir}/*.sh; do sbatch $job; done")
        print(f"\nTo submit selectively:")
        print(f"  sbatch {args.output_dir}/eval_precomp_gin_*.sh")
        print(f"  sbatch {args.output_dir}/eval_precomp_*_Pythia_410m_*.sh")
        print(f"\nTo check status:")
        print(f"  squeue -u $USER | grep eval_pc")
        print(f"\nResults will be saved to:")
        print(f"  mteb_results_precomputed/{{method}}/{{model}}/{{config}}/{{task}}.json")


if __name__ == "__main__":
    main()
