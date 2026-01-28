#!/usr/bin/env python3
"""
Unified SLURM job generator for STS (Semantic Textual Similarity) Optuna hyperparameter search.

Supports:
- Models: TinyLlama-1.1B, Pythia-410m, Pythia-2.8b, Llama3-8B
- Methods: GIN, MLP, Weighted
- All STS tasks from MTEB
- Both Bio and CS clusters (configured via .env)

Usage:
    # Single model, all methods, all STS tasks
    python3 generate_sts_jobs.py --models Pythia-410m --all-methods --all-tasks

    # Multiple models, specific method
    python3 generate_sts_jobs.py --models Pythia-410m TinyLlama-1.1B --methods gin --all-tasks
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config, get_cluster_type

# Load user configuration from .env
load_dotenv()

# Model configurations: (family, size, time_limit_gin, time_limit_mlp, time_limit_weighted, time_limit_deepset, time_limit_dwatt, memory)
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "08:00:00", "08:00:00", "08:00:00", "08:00:00", "08:00:00", "32G"),
    "Pythia-410m": ("Pythia", "410m", "03:00:00", "02:00:00", "01:30:00", "03:00:00", "03:00:00", "32G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "04:00:00", "03:00:00", "02:00:00", "04:00:00", "04:00:00", "48G"),
    "Llama3-8B": ("Llama3", "8B", "12:00:00", "12:00:00", "12:00:00", "12:00:00", "12:00:00", "64G"),
}

# Method configurations: (n_trials, description)
METHOD_CONFIGS = {
    "gin": (25, "Graph Isomorphism Network (per graph type)"),  # 100 trials per graph type
    "mlp": (25, "Multi-Layer Perceptron baseline (last layer only)"),
    "weighted": (10, "Learned layer weighting (ELMo-style)"),  # Only 10 trials - minimal params
    "deepset": (25, "DeepSet permutation-invariant encoder"),
    "dwatt": (25, "Depth-Wise Attention (ElNokrashy et al. 2024)"),  # DWAtt baseline
    # NEW LINEAR METHODS (no ReLU activations):
    "linear_gin": (25, "Linear GIN (cayley, 1 layer, no ReLU)"),
    "linear_mlp": (25, "Linear MLP (last layer, 1 hidden layer, no ReLU)"),
    "linear_deepset": (25, "Linear DeepSet (2 variants: pre1_post0 + pre0_post1, no ReLU)"),
}

# GIN graph types (separate study per graph type)
GIN_GRAPH_TYPES = ["linear", "cayley", "cayley", "virtual_node", "fully_connected"]

# STS tasks - will attempt all, precompute will filter to only those with train splits
ALL_STS_TASKS = [
    "STSBenchmark",     # STS Benchmark (0-5 scale) - MOST COMMON - Confirmed train split ✓
    "STS12",            # SemEval STS 2012 - Confirmed train split ✓
    "STS13",            # SemEval STS 2013 - Confirmed evaluation-only (test only)
    "STS14",            # SemEval STS 2014 - Unknown, will test
    "STS15",            # SemEval STS 2015 - Unknown, will test
    "STS16",            # SemEval STS 2016 - Unknown, will test
    "STS17",            # SemEval STS 2017 (cross-lingual) - Unknown, will test
    "STS22",            # SemEval STS 2022 (multilingual) - Unknown, will test
    "BIOSSES",          # Biomedical similarity - Confirmed evaluation-only (test only)
    "SICK-R",           # Compositional reasoning - Confirmed evaluation-only (test only)
]

# Recommended subset for quick experiments (confirmed trainable tasks)
RECOMMENDED_STS_TASKS = [
    "STSBenchmark",     # Most commonly used, well-studied
    "STS12",            # Good baseline task
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
#SBATCH --output={log_dir}/sts_optuna_trial_%a.out
#SBATCH --error={log_dir}/sts_optuna_trial_%a.err
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
echo "STS Optuna Hyperparameter Search"
echo "=========================================="
echo "Method: {method_upper}"
echo "Model: {model_family}-{model_size}"
echo "Task: {task}"
echo "Trial: $SLURM_ARRAY_TASK_ID / {n_trials}"
echo "Job started at: $(date)"

{ssh_tunnel_setup}
# Auto-detect Optuna database URL (handles dynamic compute nodes)
STORAGE_URL=$(python3 scripts_and_jobs/scripts/optuna_storage.py)
echo "Optuna storage: $STORAGE_URL"
echo ""

time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_sts_{method}_precomputed \\
    --study_name "{study_name}" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    {extra_args}\\
    --embeddings_dir "precomputed_embeddings_sts/{model_family}_{model_size}_mean_pooling" \\
    --storage_url "$STORAGE_URL" \\
    --n_trials 1

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ STS Optuna trial $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "✗ STS Optuna trial $SLURM_ARRAY_TASK_ID failed"
fi

{ssh_tunnel_cleanup}
exit $EXIT_CODE
"""


def get_job_dir_name(model_name, method):
    """Get directory name for STS job files."""
    # TinyLlama-1.1B + gin -> tinyllama_gin_sts
    if "-" in model_name:
        family, size = model_name.split("-", 1)
        model_str = f"{family.lower()}{size.lower()}"
    else:
        model_str = model_name.lower()
    return f"{model_str}_{method}_sts"


def create_sts_optuna_job(model_name, method, task, graph_type=None, partition_type='priority'):
    """Create a single STS Optuna SLURM job script.

    Args:
        model_name: Model name (e.g., "TinyLlama-1.1B")
        method: Method name (gin, mlp, weighted, deepset)
        task: STS task name
        graph_type: Graph type for GIN (required if method=="gin")
        partition_type: SLURM partition type
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method: {method}. Supported: {list(METHOD_CONFIGS.keys())}")
    if method == "gin" and graph_type is None:
        raise ValueError("graph_type is required for GIN method")

    model_family, model_size, time_gin, time_mlp, time_weighted, time_deepset, time_dwatt, memory = MODEL_CONFIGS[model_name]
    n_trials, method_desc = METHOD_CONFIGS[method]

    # Get cluster-specific SLURM configuration
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Select appropriate time limit based on method
    time_limits = {
        "gin": time_gin,
        "mlp": time_mlp,
        "weighted": time_weighted,
        "deepset": time_deepset,
        "dwatt": time_dwatt,
        # Linear methods use same time as their non-linear counterparts
        "linear_gin": time_gin,
        "linear_mlp": time_mlp,
        "linear_deepset": time_deepset,
    }
    time_limit = time_limits[method]

    # Study naming: include graph_type for GIN
    if method == "gin":
        study_name = f"gin_{graph_type}_sts_{task}_{model_family}_{model_size}_precomputed"
        extra_args = f"--graph_type \"{graph_type}\" "
        log_dir = f"{BASE_DIR}/job_logs/logs_sts_gin_{graph_type}_{task}_{model_family}_{model_size}"
    elif method == "linear_gin":
        # Linear GIN: fixed cayley graph, no graph_type argument (hardcoded in script)
        study_name = f"linear_gin_sts_{task}_{model_family}_{model_size}_precomputed"
        extra_args = ""
        log_dir = f"{BASE_DIR}/job_logs/logs_sts_linear_gin_{task}_{model_family}_{model_size}"
    elif method == "linear_mlp":
        # Linear MLP: fixed 1 layer, last input, no extra args
        study_name = f"linear_mlp_sts_{task}_{model_family}_{model_size}_precomputed"
        extra_args = ""
        log_dir = f"{BASE_DIR}/job_logs/logs_sts_linear_mlp_{task}_{model_family}_{model_size}"
    elif method == "linear_deepset":
        # Linear DeepSet: variant is a hyperparameter (not command-line arg)
        study_name = f"linear_deepset_sts_{task}_{model_family}_{model_size}_precomputed"
        extra_args = ""
        log_dir = f"{BASE_DIR}/job_logs/logs_sts_linear_deepset_{task}_{model_family}_{model_size}"
    else:
        study_name = f"{method}_sts_{task}_{model_family}_{model_size}_precomputed"
        extra_args = ""
        log_dir = f"{BASE_DIR}/job_logs/logs_sts_{method}_{task}_{model_family}_{model_size}"

    # Shorten task name for job name (SLURM has limits)
    task_short = task.lower().replace("benchmark", "").replace("multilingual", "multi")[:15]
    if method == "gin":
        job_name = f"sts_gin_{graph_type}_{model_family.lower()}_{task_short}"
    else:
        job_name = f"sts_{method}_{model_family.lower()}_{task_short}"

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
        study_name=study_name,
        extra_args=extra_args,
        storage_url=STORAGE_URL,
        ssh_tunnel_setup=ssh_tunnel_setup,
        ssh_tunnel_cleanup=ssh_tunnel_cleanup,
    )

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for STS Optuna hyperparameter search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model, all methods, all STS tasks
    python3 generate_sts_jobs.py --models Pythia-410m --all-methods --all-tasks

    # Recommended STS tasks only (STSBenchmark, STS12)
    python3 generate_sts_jobs.py --models Pythia-410m --all-methods --recommended

    # Specific task
    python3 generate_sts_jobs.py --models TinyLlama-1.1B --methods gin weighted --tasks STSBenchmark
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
        help="Methods to generate jobs for (gin, mlp, weighted, deepset)"
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Generate for all methods"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=ALL_STS_TASKS,
        help="STS tasks to generate jobs for"
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Generate for all STS tasks (10 tasks - precompute will filter to trainable ones)"
    )
    parser.add_argument(
        "--recommended",
        action="store_true",
        help="Generate for recommended STS tasks only (2 confirmed trainable: STSBenchmark, STS12)"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition, 'general' for shared (default: priority)"
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
        tasks = ALL_STS_TASKS
    elif args.recommended:
        tasks = RECOMMENDED_STS_TASKS
    elif args.tasks:
        tasks = args.tasks
    else:
        print("ERROR: Must specify --tasks, --all-tasks, or --recommended")
        parser.print_help()
        return

    # Get cluster info
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=args.partition)

    print(f"\n{'='*70}")
    print(f"Generating STS Optuna jobs")
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

    # Calculate total jobs (GIN creates 4 jobs per task, others create 1)
    total_jobs = 0
    for method in methods:
        if method == "gin":
            total_jobs += len(models) * len(tasks) * len(GIN_GRAPH_TYPES)
        else:
            total_jobs += len(models) * len(tasks)
    print(f"Total jobs: {total_jobs} (GIN: {len(GIN_GRAPH_TYPES)} per task, others: 1 per task)")
    print(f"{'='*70}\n")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        for method in methods:
            # GIN: create separate jobs for each graph type
            if method == "gin":
                for graph_type in GIN_GRAPH_TYPES:
                    # Create directory for this graph type
                    job_dir_name = f"{get_job_dir_name(model_name, method)}_{graph_type}"
                    job_dir = base_output_dir / job_dir_name
                    job_dir.mkdir(parents=True, exist_ok=True)

                    print(f"\n{model_name} + GIN-{graph_type} (STS):")
                    print(f"  Output dir: {job_dir}")

                    for task in tasks:
                        job_script = create_sts_optuna_job(model_name, method, task,
                                                           graph_type=graph_type,
                                                           partition_type=args.partition)

                        # Write job file
                        job_filename = f"sts_optuna_{graph_type}_{task}_{model_name.replace('-', '').lower()}.sh"
                        job_path = job_dir / job_filename

                        with open(job_path, "w", newline="\n") as f:
                            f.write(job_script)

                        job_count += 1
                        print(f"    [{job_count:3d}] {job_filename}")
            elif method in ["linear_gin", "linear_mlp", "linear_deepset"]:
                # NEW: Linear methods (no graph type loop, fixed architectures)
                job_dir_name = get_job_dir_name(model_name, method)
                job_dir = base_output_dir / job_dir_name
                job_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{model_name} + {method.upper()} (STS):")
                print(f"  Output dir: {job_dir}")

                for task in tasks:
                    job_script = create_sts_optuna_job(model_name, method, task, partition_type=args.partition)

                    # Write job file
                    job_filename = f"sts_optuna_{task}_{model_name.replace('-', '').lower()}.sh"
                    job_path = job_dir / job_filename

                    with open(job_path, "w", newline="\n") as f:
                        f.write(job_script)

                    job_count += 1
                    print(f"    [{job_count:3d}] {job_filename}")
            else:
                # MLP, Weighted, DeepSet: one job per task
                job_dir_name = get_job_dir_name(model_name, method)
                job_dir = base_output_dir / job_dir_name
                job_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n{model_name} + {method.upper()} (STS):")
                print(f"  Output dir: {job_dir}")

                for task in tasks:
                    job_script = create_sts_optuna_job(model_name, method, task, partition_type=args.partition)

                    # Write job file
                    job_filename = f"sts_optuna_{task}_{model_name.replace('-', '').lower()}.sh"
                    job_path = job_dir / job_filename

                    with open(job_path, "w", newline="\n") as f:
                        f.write(job_script)

                    job_count += 1
                    print(f"    [{job_count:3d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} STS Optuna job files")
    print(f"{'='*70}\n")

    # Print instructions
    print("To submit STS Optuna jobs:")
    for model_name in models:
        for method in methods:
            job_dir_name = get_job_dir_name(model_name, method)
            job_dir = base_output_dir / job_dir_name
            print(f"  # {model_name} + {method.upper()} (STS)")
            print(f"  for job in {job_dir}/sts_optuna_*.sh; do sbatch \"$job\"; done")

    print("\nTo monitor progress:")
    print(f"  squeue -u $USER | grep -E 'sts_{'|'.join(methods)}'")

    print()


if __name__ == "__main__":
    main()
