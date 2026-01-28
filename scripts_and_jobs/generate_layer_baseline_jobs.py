#!/usr/bin/env python3
"""
Unified SLURM job generator for layer baseline evaluations.
Evaluates each layer separately to find best single-layer baseline.

Usage:
    # All layers, all tasks
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --all-layers --all-tasks

    # Specific layers
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --layers 0 5 10 22 --all-tasks

    # Specific tasks
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --all-layers --tasks EmotionClassification
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path for experiments imports
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Add scripts_and_jobs directory to path for cluster_config import
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_config import get_cluster_config, get_cluster_type
from transformers import AutoConfig
from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications

# Load user configuration from .env
load_dotenv()

# Model configurations: (family, size, time_limit, memory)
# Note: num_layers is now determined dynamically from model config
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "00:30:00", "32G"),
    "Pythia-410m": ("Pythia", "410m", "00:30:00", "32G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "00:45:00", "48G"),
    "Llama3-8B": ("Llama3", "8B", "01:00:00", "64G"),
}


def get_num_layers(model_name):
    """Dynamically get the number of layers (including embedding) from model config."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_family, model_size, _, _ = MODEL_CONFIGS[model_name]
    
    # Create model specs to get the model path
    model_specs = TextModelSpecifications(model_family, model_size, revision="main")
    model_path = model_specs.model_path_func(model_family, model_size)
    
    # Load config and get num_layers = num_hidden_layers + 1 (embedding layer)
    config = AutoConfig.from_pretrained(model_path, revision="main")
    num_layers = config.num_hidden_layers + 1  # +1 for embedding layer
    
    return num_layers

# All classification tasks
ALL_CLASSIFICATION_TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]

# All STS tasks for evaluation
ALL_STS_TASKS = [
    "STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16", 
    "STS17", "STS22", "BIOSSES", "SICK-R"
]

# Combined list of all tasks
ALL_TASKS = ALL_CLASSIFICATION_TASKS + ALL_STS_TASKS

# Paths (loaded from .env file)
# BASE_DIR already set above
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
{cluster_sbatch_lines}
#SBATCH --output={log_dir}/layer_{layer_idx}_{task}.out
#SBATCH --error={log_dir}/layer_{layer_idx}_{task}.err
#SBATCH --job-name=L{layer_idx}_{model_short}_{task_short}

CONDA_PATH={conda_path}

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate environment
conda activate {conda_env}

# Verify environment
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to repo directory
cd "{base_dir}"

# Add project root to Python path
export PYTHONPATH="{base_dir}:$PYTHONPATH"

echo "=========================================="
echo "Layer Baseline Evaluation"
echo "=========================================="
echo "Model: {model_family}-{model_size}"
echo "Layer: {layer_idx} / {num_layers}"
echo "Task: {task}"
echo "Job started at: $(date)"

# Run MTEB evaluation for this specific layer
time python3 MTEB-Harness.py \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --revision main \\
    --evaluation_layer {layer_idx} \\
    --base_results_path results/layer_baselines \\
    --filter_tasks {task} \\
    --purpose run_tasks

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Layer {layer_idx} evaluation on {task} completed successfully"
else
    echo "✗ Layer {layer_idx} evaluation on {task} failed"
fi

exit $EXIT_CODE
"""


def get_job_dir_name(model_name):
    """Get directory name for job files."""
    # TinyLlama-1.1B -> tinyllama1.1b_layer_baseline
    # Keep dots in version number
    return model_name.replace("-", "").lower() + "_layer_baseline"


def create_layer_baseline_job(model_name, layer_idx, task, partition_type='priority'):
    """Create a single layer baseline SLURM job script."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    model_family, model_size, time_limit, memory = MODEL_CONFIGS[model_name]
    
    # Dynamically get number of layers (including embedding)
    num_layers = get_num_layers(model_name)

    # Validate layer index
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(f"Invalid layer {layer_idx} for {model_name} (has {num_layers} layers, 0-{num_layers-1})")

    # Get cluster-specific SLURM configuration
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Shorten names for SLURM job name
    model_short = model_family.lower()[:4]  # tiny, pyth, llam
    # Handle both classification and STS tasks
    if "Classification" in task:
        task_short = task.replace("Classification", "")[:8]  # Emotion, Banking7, etc.
    elif task.startswith("STS"):
        task_short = task[:8]  # STSBench, STS12, etc.
    else:
        task_short = task[:8]  # BIOSSES, SICK-R, etc.

    log_dir = f"{BASE_DIR}/job_logs/layer_baseline_{model_family}_{model_size}"

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        cluster_sbatch_lines=cluster_config.to_sbatch_lines(),
        log_dir=log_dir,
        layer_idx=layer_idx,
        task=task,
        model_short=model_short,
        task_short=task_short,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
        num_layers=num_layers,
    )

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for layer baseline evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # All layers, all tasks for TinyLlama
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --all-layers --all-tasks

    # Specific layers, all tasks
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --layers 0 5 10 22 --all-tasks

    # All layers, specific task
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --all-layers --tasks EmotionClassification

    # STS tasks
    python3 generate_layer_baseline_jobs.py --models Pythia-410m --all-layers --tasks STSBenchmark STS12

    # Last layer only
    python3 generate_layer_baseline_jobs.py --models TinyLlama-1.1B --layers 22 --all-tasks
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
        "--layers",
        nargs="+",
        type=int,
        help="Specific layer indices to evaluate"
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Generate for all layers"
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
        help="Generate for all tasks (classification + STS)"
    )
    parser.add_argument(
        "--all-sts-tasks",
        action="store_true",
        help="Generate for all STS tasks only"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition (default), 'general' for shared pool"
    )

    args = parser.parse_args()

    # Determine models
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    elif args.models:
        models = args.models
    else:
        print("ERROR: Must specify --models or --all-models")
        parser.print_help()
        return

    # Determine tasks
    if args.all_sts_tasks:
        tasks = ALL_STS_TASKS
    elif args.all_tasks:
        tasks = ALL_TASKS
    elif args.tasks:
        tasks = args.tasks
    else:
        print("ERROR: Must specify --tasks, --all-tasks, or --all-sts-tasks")
        parser.print_help()
        return

    print(f"\n{'='*70}")
    print(f"Generating Layer Baseline jobs")
    print(f"{'='*70}")
    print(f"Models: {', '.join(models)}")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        model_family, model_size, *_ = MODEL_CONFIGS[model_name]
        
        # Dynamically get number of layers (including embedding)
        num_layers = get_num_layers(model_name)

        # Determine layers
        if args.all_layers:
            layers = list(range(num_layers))
        elif args.layers:
            layers = args.layers
        else:
            print(f"ERROR: Must specify --layers or --all-layers for {model_name}")
            continue

        job_dir_name = get_job_dir_name(model_name)
        job_dir = base_output_dir / job_dir_name
        job_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{model_name} ({num_layers} layers total):")
        print(f"  Layers: {layers if len(layers) <= 10 else f'{len(layers)} layers'}")
        print(f"  Tasks: {', '.join(tasks)}")
        print(f"  Output dir: {job_dir}")
        print(f"  Total jobs: {len(layers) * len(tasks)}")

        for layer_idx in layers:
            for task in tasks:
                job_script = create_layer_baseline_job(model_name, layer_idx, task, partition_type=args.partition)

                # Write job file
                job_filename = f"layer_{layer_idx:02d}_{task}_{model_name.replace('-', '').replace('.', '').lower()}.sh"
                job_path = job_dir / job_filename

                with open(job_path, "w", newline="\n") as f:
                    f.write(job_script)

                job_count += 1

        print(f"  ✓ Generated {len(layers) * len(tasks)} job files")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} layer baseline job files")
    print(f"{'='*70}\n")

    # Print instructions
    print("NEXT STEPS:\n")
    print("1. Create log directories:")
    for model_name in models:
        model_family, model_size, *_ = MODEL_CONFIGS[model_name]
        log_dir = f"{BASE_DIR}/job_logs/layer_baseline_{model_family}_{model_size}"
        print(f"   mkdir -p {log_dir}")

    print("\n2. Submit jobs:")
    for model_name in models:
        job_dir_name = get_job_dir_name(model_name)
        job_dir = base_output_dir / job_dir_name
        print(f"   # {model_name}")
        print(f"   for job in {job_dir}/layer_*.sh; do sbatch \"$job\"; done")

    print("\n3. Monitor progress:")
    print(f"   squeue -u $USER | grep -E 'L[0-9]+'")

    print("\n4. After completion, check results:")
    for model_name in models:
        model_family, model_size, *_ = MODEL_CONFIGS[model_name]
        print(f"   ls results/layer_baselines/{model_family}/{model_size}/")

    print()


if __name__ == "__main__":
    main()
