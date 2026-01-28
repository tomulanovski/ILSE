#!/usr/bin/env python3
"""
Unified SLURM job generator for Optuna hyperparameter search (with precomputed embeddings).

Supports:
- Models: TinyLlama-1.1B, Pythia-410m, Pythia-2.8b, Llama3-8B
- Methods: GIN, MLP, Weighted
- All classification tasks
- Both Bio and CS clusters (configured via .env)

Usage:
    # Single model, single method, single task
    python3 generate_optuna_jobs.py --models Pythia-410m --methods gin --tasks EmotionClassification

    # Single model, all methods, all tasks
    python3 generate_optuna_jobs.py --models TinyLlama-1.1B --all-methods --all-tasks

    # Multiple models with specific partition
    python3 generate_optuna_jobs.py --models Pythia-410m Pythia-2.8b --methods mlp --all-tasks --partition priority
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config, get_cluster_type

# Load user configuration from .env
load_dotenv()

# Model configurations: (family, size, time_limit_gin, time_limit_mlp, time_limit_weighted, time_limit_deepset, time_limit_dwatt, memory)
# Cluster-specific settings (partition, account, qos, gpu) come from cluster_config
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "04:00:00", "03:00:00", "02:00:00", "04:00:00", "04:00:00", "32G"),
    "Pythia-410m": ("Pythia", "410m", "04:00:00", "03:00:00", "02:00:00", "04:00:00", "04:00:00", "32G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "05:00:00", "04:00:00", "03:00:00", "05:00:00", "05:00:00", "48G"),
    "Llama3-8B": ("Llama3", "8B", "06:00:00", "05:00:00", "04:00:00", "06:00:00", "06:00:00", "64G"),
}

# Method configurations: (n_trials, description)
METHOD_CONFIGS = {
    "gin": (100, "Graph Isomorphism Network"),
    "mlp": (100, "Multi-Layer Perceptron baseline"),
    "weighted": (50, "Learned layer weighting (ELMo-style)"),
    "deepset": (100, "DeepSet (φ → pool → ρ)"),
    "dwatt": (50, "Depth-Wise Attention (ElNokrashy et al. 2024)"),
}

# All classification tasks (7 tasks)
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
STORAGE_URL = os.getenv("GNN_OPTUNA_STORAGE")

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")
if not STORAGE_URL:
    raise ValueError("GNN_OPTUNA_STORAGE not set in .env file. Copy .env.example to .env and configure it.")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
{cluster_sbatch_lines}
#SBATCH --output={log_dir}/optuna_trial_%a.out
#SBATCH --error={log_dir}/optuna_trial_%a.err
#SBATCH --array=1-{n_trials}
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

# Change to repo directory
cd "{base_dir}"

# Add project root to Python path
export PYTHONPATH="{base_dir}:$PYTHONPATH"

echo "=========================================="
echo "Optuna Hyperparameter Search (Precomputed)"
echo "=========================================="
echo "Method: {method_upper}"
echo "Model: {model_family}-{model_size}"
echo "Task: {task}"
echo "Trial: $SLURM_ARRAY_TASK_ID / {n_trials}"
echo "Job started at: $(date)"

# Path to precomputed embeddings
EMBEDDINGS_DIR="./precomputed_embeddings/{model_family}_{model_size}_mean_pooling"

# Check if embeddings exist
if [ ! -d "$EMBEDDINGS_DIR/{task}" ]; then
    echo "ERROR: Precomputed embeddings not found at: $EMBEDDINGS_DIR/{task}"
    echo ""
    echo "Please run precompute step first:"
    echo "  python3 pipeline.py precompute --model {model_family}-{model_size}"
    echo "Or:"
    echo "  python3 scripts_and_jobs/generate_precompute_jobs.py --models {model_family}-{model_size} --tasks {task}"
    echo ""
    exit 1
fi

echo "Using precomputed embeddings from: $EMBEDDINGS_DIR"
echo ""

{ssh_tunnel_setup}
# Auto-detect Optuna database URL (handles dynamic compute nodes)
STORAGE_URL=$(python3 scripts_and_jobs/scripts/optuna_storage.py)
echo "Optuna storage: $STORAGE_URL"
echo ""

{python_command}

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Optuna trial $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "✗ Optuna trial $SLURM_ARRAY_TASK_ID failed"
fi

{ssh_tunnel_cleanup}
exit $EXIT_CODE
"""


def get_job_dir_name(model_name, method):
    """Get directory name for job files."""
    # TinyLlama-1.1B + gin -> tinyllama_gin_optuna
    # Pythia-410m + mlp -> pythia410m_mlp_optuna
    # Pythia-2.8b + mlp -> pythia2.8b_mlp_optuna
    # Remove only the first dash (between family and size), keep dots
    if "-" in model_name:
        family, size = model_name.split("-", 1)
        model_str = f"{family.lower()}{size.lower()}"
    else:
        model_str = model_name.lower()
    return f"{model_str}_{method}_optuna"


def create_optuna_job(model_name, method, task, partition_type='priority', filter_graph_type=None):
    """Create a single Optuna SLURM job script."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method: {method}. Supported: {list(METHOD_CONFIGS.keys())}")

    model_family, model_size, time_gin, time_mlp, time_weighted, time_deepset, time_dwatt, memory = MODEL_CONFIGS[model_name]
    n_trials, method_desc = METHOD_CONFIGS[method]

    # Get cluster-specific SLURM configuration
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Select appropriate time limit based on method
    time_limits = {"gin": time_gin, "mlp": time_mlp, "weighted": time_weighted, "deepset": time_deepset, "dwatt": time_dwatt}
    time_limit = time_limits[method]

    # Build filter argument if filtering by graph_type
    # When filter is set, graph_type will be included in study name for clarity
    if filter_graph_type is not None:
        filter_graph_type_arg = f" \\\n    --filter_graph_type {filter_graph_type}"
        # Add to job name and log dir for clarity
        job_suffix = f"_{filter_graph_type}"
    else:
        filter_graph_type_arg = ""
        job_suffix = ""

    # Shorten task name for job name (SLURM has limits)
    task_short = task.replace("Classification", "").lower()[:10]
    job_name = f"{method}_{model_family.lower()}_{task_short}{job_suffix}"

    log_dir = f"{BASE_DIR}/job_logs/logs_{method}_{task}_{model_family}_{model_size}{job_suffix}"

    # SSH tunnel setup for CS cluster (compute nodes can't access login node's tunnel)
    if cluster == 'cs':
        ssh_tunnel_setup = """# === SSH Tunnel Setup for CS Cluster ===
# CS compute nodes cannot access the login node's tunnel, so we create our own
echo "Setting up SSH tunnel to Bio cluster PostgreSQL..."
BIO_LOGIN="${GNN_BIO_LOGIN:-USERNAME@your-cluster-login-node}"
SSH_KEY="${GNN_SSH_KEY:-/home/USERNAME/.ssh/id_rsa}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Get current PostgreSQL node from Bio cluster
DB_HOST=$(ssh $SSH_OPTS "$BIO_LOGIN" "cat ~/.optuna_db_host 2>/dev/null || echo 'your-compute-node'")
DB_HOST=$(echo "$DB_HOST" | cut -d. -f1)  # Strip domain suffix
echo "  PostgreSQL running on Bio node: $DB_HOST"

# Use unique port per array task to avoid conflicts when multiple jobs run on same node
# If SLURM_ARRAY_TASK_ID is not set, use process ID and random offset for uniqueness
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    # Fallback: use process ID + random offset to avoid conflicts
    BASE_PORT=$((5433 + $$ % 1000 + RANDOM % 100))
else
    BASE_PORT=$((5433 + SLURM_ARRAY_TASK_ID))
fi

# Find an available port starting from BASE_PORT
LOCAL_PORT=$BASE_PORT
REMOTE_PORT=5432
MAX_ATTEMPTS=50
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    # Check if port is available (not in use)
    if ! netstat -tuln 2>/dev/null | grep -q ":${LOCAL_PORT} " && ! ss -tuln 2>/dev/null | grep -q ":${LOCAL_PORT} "; then
        break
    fi
    LOCAL_PORT=$((LOCAL_PORT + 1))
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "  ✗ Could not find available port after $MAX_ATTEMPTS attempts"
    exit 1
fi

echo "  Using local port: $LOCAL_PORT (base: $BASE_PORT, attempt: $ATTEMPT)"

# Create SSH tunnel with auto-accept host key
ssh $SSH_OPTS -f -N -L ${LOCAL_PORT}:${DB_HOST}:${REMOTE_PORT} "$BIO_LOGIN"
sleep 3  # Wait for tunnel to establish

# Verify tunnel is running
if pgrep -f "ssh.*${LOCAL_PORT}.*${REMOTE_PORT}" > /dev/null; then
    TUNNEL_PID=$(pgrep -f "ssh.*${LOCAL_PORT}.*${REMOTE_PORT}")
    echo "  ✓ SSH tunnel established (PID: $TUNNEL_PID)"
else
    echo "  ✗ Failed to establish SSH tunnel"
    exit 1
fi

# Override storage URL to use the dynamic port
# Use credentials from environment or construct from GNN_OPTUNA_STORAGE
if [ -z "$GNN_OPTUNA_STORAGE" ]; then
    echo "  ⚠ WARNING: GNN_OPTUNA_STORAGE not set. Please configure in .env file"
    export GNN_OPTUNA_STORAGE="postgresql://USERNAME:PASSWORD@localhost:${LOCAL_PORT}/optuna"
else
    # Extract credentials from existing GNN_OPTUNA_STORAGE and update port
    CREDS=$(echo "$GNN_OPTUNA_STORAGE" | sed 's|postgresql://||' | sed 's|@.*||')
    export GNN_OPTUNA_STORAGE="postgresql://${CREDS}@localhost:${LOCAL_PORT}/optuna"
fi
echo "  ✓ Using tunnel: localhost:${LOCAL_PORT} -> ${DB_HOST}:${REMOTE_PORT}"
echo ""
# === End SSH Tunnel Setup ===

"""
        ssh_tunnel_cleanup = """
# Clean up SSH tunnel
if [ ! -z "$TUNNEL_PID" ]; then
    echo "Closing SSH tunnel (PID: $TUNNEL_PID)..."
    kill $TUNNEL_PID 2>/dev/null || true
fi
"""
    else:
        # Bio cluster - no tunnel needed
        ssh_tunnel_setup = ""
        ssh_tunnel_cleanup = ""

    # Build Python command based on method
    if method == "deepset":
        # DeepSets uses its own dedicated script
        python_command = f"""time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_deepset_precomputed \\
    --study_name "deepset_{task}_{model_family}_{model_size}_precomputed" \\
    --embeddings_dir "$EMBEDDINGS_DIR" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --storage_url "$STORAGE_URL" \\
    --n_trials 1 \\
    --deepset_hidden_dim 256"""
    elif method == "dwatt":
        # DWAtt uses its own dedicated script (paper-faithful architecture)
        python_command = f"""time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_dwatt_precomputed \\
    --study_name "dwatt_{task}_{model_family}_{model_size}_precomputed" \\
    --embeddings_dir "$EMBEDDINGS_DIR" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --storage_url "$STORAGE_URL" \\
    --n_trials 1"""
    else:
        # GIN/MLP/Weighted use the unified script
        # Include graph_type in study name if filter is set (format: method_graphType_task_model_size, like STS)
        if filter_graph_type is not None:
            study_name = f"{method}_{filter_graph_type}_{task}_{model_family}_{model_size}"
        else:
            study_name = f"{method}_{task}_{model_family}_{model_size}"
        
        python_command = f"""time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_gin_precomputed \\
    --study_name "{study_name}" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --embeddings_dir "$EMBEDDINGS_DIR" \\
    --storage_url "$STORAGE_URL" \\
    --encoder {method} \\
    --n_trials 1{filter_graph_type_arg}"""

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        cluster_sbatch_lines=cluster_config.to_sbatch_lines(),
        log_dir=log_dir,
        n_trials=n_trials,
        job_name=job_name,
        method=method,
        method_upper=method.upper(),
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
        task=task,
        storage_url=STORAGE_URL,
        ssh_tunnel_setup=ssh_tunnel_setup,
        ssh_tunnel_cleanup=ssh_tunnel_cleanup,
        filter_graph_type_arg=filter_graph_type_arg,
        python_command=python_command,
    )

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for Optuna hyperparameter search (with precomputed embeddings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model, single method, all tasks
    python3 generate_optuna_jobs.py --models Pythia-410m --methods gin --all-tasks

    # Single model, all methods, all tasks
    python3 generate_optuna_jobs.py --models TinyLlama-1.1B --all-methods --all-tasks

    # Multiple models, specific method, specific tasks
    python3 generate_optuna_jobs.py --models Pythia-410m Pythia-2.8b --methods mlp --tasks EmotionClassification

    # All combinations
    python3 generate_optuna_jobs.py --all-models --all-methods --all-tasks
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
        "--methods",
        nargs="+",
        choices=list(METHOD_CONFIGS.keys()),
        help="Methods to generate jobs for (gin, mlp, weighted)"
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Generate for all methods"
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
    parser.add_argument(
        "--graph_type",
        type=str,
        choices=["linear", "virtual_node", "cayley", "cayley"],
        help="Filter by graph_type (e.g., 'cayley'). If not specified, generates jobs for all graph types. Only applies to 'gin' method."
    )

    args = parser.parse_args()

    # Determine models, methods, and tasks
    if args.all_models:
        models = list(MODEL_CONFIGS.keys())
    elif args.models:
        models = args.models
    else:
        print("ERROR: Must specify --models or --all-models")
        parser.print_help()
        return

    if args.all_methods:
        methods = list(METHOD_CONFIGS.keys())
    elif args.methods:
        methods = args.methods
    else:
        print("ERROR: Must specify --methods or --all-methods")
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
    print(f"Generating Optuna jobs (with precomputed embeddings)")
    print(f"{'='*70}")
    print(f"Cluster: {cluster.upper()}")
    print(f"Partition type: {args.partition}")
    print(f"  Partition: {cluster_config.partition}")
    if cluster_config.account:
        print(f"  Account: {cluster_config.account}")
    if cluster_config.qos:
        print(f"  QOS: {cluster_config.qos}")
    print(f"Models: {', '.join(models)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Tasks: {', '.join(tasks)}")
    if args.graph_type:
        print(f"Graph Type Filter: {args.graph_type} only (applies to 'gin' method)")
    print(f"Total jobs: {len(models) * len(methods) * len(tasks)}")
    print(f"{'='*70}\n")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        for method in methods:
            job_dir_name = get_job_dir_name(model_name, method)
            job_dir = base_output_dir / job_dir_name
            job_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{model_name} + {method.upper()}:")
            print(f"  Output dir: {job_dir}")

            for task in tasks:
                # Only apply graph_type filter to 'gin' method
                filter_gt = args.graph_type if (args.graph_type and method == 'gin') else None
                job_script = create_optuna_job(model_name, method, task, partition_type=args.partition, filter_graph_type=filter_gt)

                # Write job file (keep dots in version number, remove only dashes)
                job_filename = f"optuna_{task}_{model_name.replace('-', '').lower()}.sh"
                job_path = job_dir / job_filename

                with open(job_path, "w", newline="\n") as f:
                    f.write(job_script)

                job_count += 1
                print(f"    [{job_count:3d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} Optuna job files")
    print(f"{'='*70}\n")

    # Print instructions
    print("IMPORTANT: Ensure precomputed embeddings exist before submitting!\n")
    print("Check precomputed embeddings:")
    for model_name in models:
        model_family, model_size, *_ = MODEL_CONFIGS[model_name]
        print(f"  ls precomputed_embeddings/{model_family}_{model_size}_mean_pooling/")

    print("\nIf embeddings are missing, run precompute first:")
    print(f"  python3 pipeline.py precompute --model <model-name>")
    print(f"  # Or:")
    print(f"  python3 scripts_and_jobs/generate_precompute_jobs.py --models <model> --all-tasks")

    print("\nTo submit Optuna jobs:")
    for model_name in models:
        for method in methods:
            job_dir_name = get_job_dir_name(model_name, method)
            job_dir = base_output_dir / job_dir_name
            print(f"  # {model_name} + {method.upper()}")
            print(f"  for job in {job_dir}/optuna_*.sh; do sbatch \"$job\"; done")

    print("\nTo monitor progress:")
    print(f"  squeue -u $USER | grep -E '{'|'.join(methods)}'")

    print()


if __name__ == "__main__":
    main()
