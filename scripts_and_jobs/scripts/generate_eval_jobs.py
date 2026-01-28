#!/usr/bin/env python3
"""
Generate SLURM job files to evaluate all trained models on MTEB test sets.

Scans the saved_models directory and creates evaluation jobs for each model.

Usage:
    python3 generate_eval_jobs.py --dry_run  # Preview
    python3 generate_eval_jobs.py            # Generate jobs
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import cluster_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from cluster_config import get_cluster_config, get_cluster_type


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
        mlp_Banking77Classification_Pythia_410m_mean_layers2.pt
        weighted_EmotionClassification_Pythia_410m_softmax.pt
        gin_EmotionClassification_Pythia_410m_virtual_node.pt  # Note: virtual_node has underscore!

    Returns:
        Dict with keys: method, task, model_family, model_size, graph_type, mlp_layers (optional)
    """
    if not filename.endswith('.pt'):
        return None

    # Remove .pt extension
    base = filename[:-3]

    # First, extract method (first part or first two parts if linear_)
    parts = base.split('_')

    # Check if this is a linear model (starts with "linear_")
    if parts[0] == 'linear' and len(parts) > 1:
        method = f"{parts[0]}_{parts[1]}"  # e.g., "linear_gcn" or "linear_mlp"
        if parts[1] not in ['gin', 'gcn', 'mlp', 'deepset', 'dwatt']:
            print(f"  ⚠️ Skipping {filename} - unknown linear method '{parts[1]}'")
            return None
    else:
        method = parts[0]
        if method not in ['gin', 'gcn', 'mlp', 'lora', 'weighted', 'deepset', 'dwatt']:
            print(f"  ⚠️ Skipping {filename} - unknown method '{method}'")
            return None

    # Remove method from the string
    remaining = base[len(method)+1:]  # +1 for the underscore

    # Known graph types (including multi-part ones)
    # Note: Order matters - longer matches first (cayley_attention before cayley)
    # Linear models include pooling method in config (e.g., cayley_mean, cayley_attention)
    known_graph_types = ['virtual_node', 'fully_connected', 'cayley_attention', 'cayley_mean', 'cayley', 'linear', 'cayley']
    known_mlp_inputs = ['last', 'mean', 'flatten']
    known_weighted_types = ['softmax']  # For weighted baseline
    known_dwatt_types = ['paper']  # For DWAtt baseline (paper-faithful, no hidden_dim projection)

    # For MLP models, check for _layers{N} suffix first
    mlp_layers = None
    if method == 'mlp':
        # Check for _layers{N} pattern at the end
        import re
        layers_match = re.search(r'_layers(\d+)$', remaining)
        if layers_match:
            mlp_layers = layers_match.group(1)
            # Remove the _layers{N} suffix
            remaining = remaining[:layers_match.start()]

    # For DeepSet models, check for _<pooling>_pre<N>_post<M> pattern
    deepset_config = None
    if method == 'deepset':
        import re
        deepset_match = re.search(r'_(mean|sum)_pre(\d+)_post(\d+)$', remaining)
        if deepset_match:
            pooling_type, pre_layers, post_layers = deepset_match.groups()
            deepset_config = f"{pooling_type}_pre{pre_layers}_post{post_layers}"
            graph_type_or_input = deepset_config  # Store in graph_type_or_input for compatibility
            remaining = remaining[:deepset_match.start()]
        else:
            print(f"  ⚠️ Skipping {filename} - couldn't parse DeepSet config")
            return None
    elif method == 'dwatt':
        import re
        # DWAtt can have _paper (paper-faithful) or _hidden{N} suffix
        dwatt_match = re.search(r'_(paper|hidden\d+)$', remaining)
        if dwatt_match:
            dwatt_config = dwatt_match.group(1)
            graph_type_or_input = dwatt_config
            remaining = remaining[:dwatt_match.start()]
        else:
            print(f"  ⚠️ Skipping {filename} - couldn't parse DWAtt config")
            return None
    else:
        # Try to match graph_type from the end (for GIN/MLP/Weighted)
        graph_type_or_input = None
        for gt in known_graph_types + known_mlp_inputs + known_weighted_types + known_dwatt_types:
            if remaining.endswith('_' + gt):
                graph_type_or_input = gt
                remaining = remaining[:-(len(gt)+1)]  # Remove graph_type and underscore
                break

        if graph_type_or_input is None:
            print(f"  ⚠️ Skipping {filename} - couldn't identify graph_type/mlp_input")
            return None

    # Now remaining should be: {task}_{model_family}_{model_size}
    # We know model_family is one of: Pythia, Llama3, BERT
    # And model_size follows it
    known_families = ['Pythia', 'Llama3', 'TinyLlama', 'BERT', 'Llama']

    parts = remaining.split('_')

    # Find model_family by looking for known families
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
        'graph_type': graph_type_or_input,
        'filename': filename
    }

    # Add mlp_layers if present
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


def generate_eval_job(model_info: Dict, output_dir: str, repo_root: str, partition_type: str = 'priority') -> str:
    """
    Generate a SLURM job file for evaluating a single model.

    Args:
        model_info: Model metadata dict
        output_dir: Directory to save job files
        repo_root: Root directory of the repository
        partition_type: 'priority' or 'general' for SLURM partition selection

    Returns:
        Path to generated job file
    """
    method = model_info['method']
    task = model_info['task']
    model_family = model_info['model_family']
    model_size = model_info['model_size']
    graph_type = model_info['graph_type']
    model_path = model_info['full_path']
    mlp_layers = model_info.get('mlp_layers', None)

    # Get cluster configuration
    cluster_config = get_cluster_config(partition_type=partition_type)
    
    # Get paths from environment
    conda_path = os.getenv("GNN_CONDA_PATH")
    conda_env = os.getenv("GNN_CONDA_ENV", "final_project")
    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())

    # Job filename - include mlp_layers if present
    if mlp_layers:
        job_filename = f"eval_{method}_{task}_{model_family}_{model_size}_{graph_type}_layers{mlp_layers}.sh"
        config_suffix = f"{graph_type}_layers{mlp_layers}"
    else:
        job_filename = f"eval_{method}_{task}_{model_family}_{model_size}_{graph_type}.sh"
        config_suffix = graph_type
    job_path = os.path.join(output_dir, job_filename)

    # Results directory - organized by method, model, and configuration
    results_dir = f"mteb_results_best_models/{method}/{model_family}_{model_size}/{config_suffix}"

    # SLURM script content (cluster-aware)
    script = f"""#!/bin/bash
#SBATCH --job-name=eval_{method}_{task[:15]}
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
{cluster_config.to_sbatch_lines()}
#SBATCH --output={base_dir}/job_logs/eval_best/{method}_{task}_{model_family}_{model_size}_{config_suffix}.out
#SBATCH --error={base_dir}/job_logs/eval_best/{method}_{task}_{model_family}_{model_size}_{config_suffix}.err

# Source conda
source "{conda_path}/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env} || {{
    echo "ERROR: Could not activate {conda_env} env"
    exit 1
}}

echo "Current conda environment: $CONDA_PREFIX"

# Print GPU info
nvidia-smi

# Go to repo directory
cd "{repo_root}"

echo "=========================================="
echo "Evaluating Best Model on Test Set"
echo "=========================================="
echo "Method: {method.upper()}"
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Configuration: {config_suffix}"
echo "Model Path: {model_path}"
echo "Started at: $(date)"
echo "=========================================="

# Run evaluation
python3 -m scripts_and_jobs.scripts.eval.mteb_evaluator \\
    --model_path "{model_path}" \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --tasks {task} \\
    --batch_size 32 \\
    --output_dir {results_dir} \\
    --device_map auto

EXIT_CODE=$?

echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: {results_dir}/"
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
        description="Generate SLURM jobs to evaluate all trained models on MTEB test sets"
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="saved_models",
        help="Directory containing trained model checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts_and_jobs/slurm_jobs/generated_jobs/eval_best_models",
        help="Directory to save generated job files"
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=os.getenv("GNN_REPO_DIR", os.getcwd()),
        help="Root directory of the repository (for cluster)"
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
        help="Only generate jobs for specific method (gin/gcn/mlp/weighted/deepset/dwatt)"
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
        help="Only generate jobs for specific graph_type (e.g., cayley, linear, virtual_node)"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition, 'general' for shared (default: priority)"
    )

    args = parser.parse_args()
    
    # Get repo_root from environment if not provided
    if not args.repo_root:
        args.repo_root = os.getenv("GNN_REPO_DIR", os.getcwd())
        args.repo_root = os.getenv('GNN_REPO_DIR', os.getcwd())

    print("="*70)
    print("Generate SLURM Jobs for Model Evaluation on Test Sets")
    print("="*70)

    # Find all models
    models = find_all_models(args.models_dir)

    if not models:
        print("\nNo models found!")
        return

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
        models = [m for m in models if m.get('graph_type') == args.filter_graph_type]
        print(f"\nFiltered to graph_type: {args.filter_graph_type}")

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
            config = m['graph_type']
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
    else:
        print(f"\n{'='*70}")
        print("Generating job files...")
        print(f"{'='*70}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Create log directory
        os.makedirs("job_logs/eval_best", exist_ok=True)

        # Generate each job file
        for model in models:
            job_file = generate_eval_job(model, args.output_dir, args.repo_root, partition_type=args.partition)
            print(f"  ✓ {os.path.basename(job_file)}")

        print(f"\n{'='*70}")
        print("Job files generated successfully!")
        print(f"{'='*70}")
        print(f"\nTo submit all jobs:")
        print(f"  cd {args.repo_root}")
        print(f"  for job in {args.output_dir}/*.sh; do sbatch $job; done")
        print(f"\nTo submit selectively:")
        print(f"  sbatch {args.output_dir}/eval_gin_EmotionClassification_*.sh")
        print(f"  sbatch {args.output_dir}/eval_*_Pythia_410m_*.sh")
        print(f"\nTo check status:")
        print(f"  squeue -u $USER | grep eval")
        print(f"\nResults will be saved to:")
        print(f"  mteb_results_best_models/{{method}}/{{model}}/{{graph_type}}/{{task}}.json")


if __name__ == "__main__":
    main()
