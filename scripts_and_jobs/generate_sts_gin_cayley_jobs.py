#!/usr/bin/env python3
"""
Generate SLURM jobs for STS GIN Cayley Optuna search.

Features:
- pool_real_nodes_only: exclude virtual nodes from pooling
- sum pooling option in search space
- train_eps: learnable epsilon in GIN aggregation
- STSBenchmark-only training workflow (train on STSBenchmark, evaluate on all STS tasks)

Usage:
    python3 generate_sts_gin_cayley_jobs.py --models Pythia-410m
    python3 generate_sts_gin_cayley_jobs.py --all-models
"""
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from cluster_config import get_cluster_config, get_cluster_type

# Load user configuration from .env
load_dotenv()

# Model configurations: (family, size, time_limit, memory)
MODEL_CONFIGS = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", "08:00:00", "32G"),
    "Pythia-410m": ("Pythia", "410m", "03:00:00", "32G"),
    "Pythia-2.8b": ("Pythia", "2.8b", "04:00:00", "48G"),
    "Llama3-8B": ("Llama3", "8B", "12:00:00", "64G"),
}

# GIN cayley graph types - focusing on cayley (where pool_real_nodes_only matters most)
# You can add other graph types if needed, but cayley benefits most from this feature
GIN_V2_GRAPH_TYPES = ["cayley"]  # Can add: "linear", "cayley", "virtual_node"

# DEFAULT TRAINING TASK: Train on STSBenchmark only (best splits, most studied)
# After Optuna, train best models and evaluate on ALL STS tasks to test generalization
DEFAULT_TRAIN_TASK = "STSBenchmark"

# Evaluation tasks (used after training best models from STSBenchmark Optuna)
EVAL_STS_TASKS = [
    "STSBenchmark",  # In-domain (same as training)
    "SICK-R",        # Compositional reasoning
    "BIOSSES",       # Biomedical domain transfer
    "STS12",         # Classic benchmark
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
]

# More trials for Cayley hyperparameter exploration
# pool_real_nodes_only (2) × train_eps (2) × node_to_choose (3) = 12 base combinations
N_TRIALS = 50  # Enough to explore the expanded search space

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
#SBATCH --output={log_dir}/sts_gin_cayley_optuna_trial_%a.out
#SBATCH --error={log_dir}/sts_gin_cayley_optuna_trial_%a.err
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
echo "STS GIN Cayley Optuna (Enhanced Pooling)"
echo "=========================================="
echo "Model: {model_family}-{model_size}"
echo "Graph Type: {graph_type}"
echo "Task: {task}"
echo "Trial: $SLURM_ARRAY_TASK_ID / {n_trials}"
echo "Cayley features: pool_real_nodes_only + train_eps + sum pooling"
echo "Job started at: $(date)"

{ssh_tunnel_setup}
# Auto-detect Optuna database URL (handles dynamic compute nodes)
STORAGE_URL=$(python3 scripts_and_jobs/scripts/optuna_storage.py)
echo "Optuna storage: $STORAGE_URL"
echo ""

time python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_sts_gin_precomputed_cayley \\
    --study_name "{study_name}" \\
    --task "{task}" \\
    --model_family "{model_family}" \\
    --model_size "{model_size}" \\
    --graph_type "{graph_type}" \\
    --embeddings_dir "precomputed_embeddings_sts/{model_family}_{model_size}_mean_pooling" \\
    --storage_url "$STORAGE_URL" \\
    --n_trials 1

EXIT_CODE=$?

echo "Job finished at: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ STS GIN Cayley Optuna trial $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "✗ STS GIN Cayley Optuna trial $SLURM_ARRAY_TASK_ID failed"
fi

{ssh_tunnel_cleanup}
exit $EXIT_CODE
"""


def create_sts_gin_cayley_job(model_name, task, graph_type, partition_type='priority'):
    """Create STS GIN Cayley Optuna SLURM job script.

    Args:
        model_name: Model name (e.g., "TinyLlama-1.1B")
        task: STS task name
        graph_type: Graph type for GIN (e.g., "cayley")
        partition_type: SLURM partition type
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_CONFIGS.keys())}")

    model_family, model_size, time_limit, memory = MODEL_CONFIGS[model_name]

    # Get cluster-specific SLURM configuration
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Study naming with cayley suffix for study identification
    study_name = f"gin_{graph_type}_sts_{task}_{model_family}_{model_size}_cayley_precomputed"
    log_dir = f"{BASE_DIR}/job_logs/logs_sts_gin_{graph_type}_{task}_{model_family}_{model_size}_cayley"

    # Shorten task name for job name (SLURM has limits)
    task_short = task.lower().replace("benchmark", "").replace("multilingual", "multi")[:15]
    job_name = f"sts_gin_cayley_{graph_type}_{model_family.lower()}_{task_short}"

    # SSH tunnel setup for CS cluster
    if cluster == 'cs':
        ssh_tunnel_setup = """# === SSH Tunnel Setup for CS Cluster ===
echo "Setting up SSH tunnel to Bio cluster PostgreSQL..."
BIO_LOGIN="${GNN_BIO_LOGIN:-USERNAME@your-cluster-login-node}"
SSH_KEY="${GNN_SSH_KEY:-/home/USERNAME/.ssh/id_rsa}"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Get current PostgreSQL node from Bio cluster
DB_HOST=$(ssh $SSH_OPTS "$BIO_LOGIN" "cat ~/.optuna_db_host 2>/dev/null || echo 'your-compute-node'")
DB_HOST=$(echo "$DB_HOST" | cut -d. -f1)

# Find available port
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    BASE_PORT=$((5433 + $$ % 1000 + RANDOM % 100))
else
    BASE_PORT=$((5433 + SLURM_ARRAY_TASK_ID))
fi

LOCAL_PORT=$BASE_PORT
REMOTE_PORT=5432
MAX_ATTEMPTS=50
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if ! netstat -tuln 2>/dev/null | grep -q ":${LOCAL_PORT} " && ! ss -tuln 2>/dev/null | grep -q ":${LOCAL_PORT} "; then
        break
    fi
    LOCAL_PORT=$((LOCAL_PORT + 1))
    ATTEMPT=$((ATTEMPT + 1))
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "  ✗ Could not find available port"
    exit 1
fi

ssh $SSH_OPTS -f -N -L ${LOCAL_PORT}:${DB_HOST}:${REMOTE_PORT} "$BIO_LOGIN"
sleep 3

if pgrep -f "ssh.*${LOCAL_PORT}.*${REMOTE_PORT}" > /dev/null; then
    TUNNEL_PID=$(pgrep -f "ssh.*${LOCAL_PORT}.*${REMOTE_PORT}")
    echo "  ✓ SSH tunnel established (PID: $TUNNEL_PID)"
else
    echo "  ✗ Failed to establish SSH tunnel"
    exit 1
fi

# Use credentials from environment or construct from GNN_OPTUNA_STORAGE
if [ -z "$GNN_OPTUNA_STORAGE" ]; then
    echo "  ⚠ WARNING: GNN_OPTUNA_STORAGE not set. Please configure in .env file"
    export GNN_OPTUNA_STORAGE="postgresql://USERNAME:PASSWORD@localhost:${LOCAL_PORT}/optuna"
else
    # Extract credentials from existing GNN_OPTUNA_STORAGE and update port
    CREDS=$(echo "$GNN_OPTUNA_STORAGE" | sed 's|postgresql://||' | sed 's|@.*||')
    export GNN_OPTUNA_STORAGE="postgresql://${CREDS}@localhost:${LOCAL_PORT}/optuna"
fi
"""
        ssh_tunnel_cleanup = """
# Clean up SSH tunnel
if [ ! -z "$TUNNEL_PID" ]; then
    kill $TUNNEL_PID 2>/dev/null || true
fi
"""
    else:
        ssh_tunnel_setup = ""
        ssh_tunnel_cleanup = ""

    script = SLURM_TEMPLATE.format(
        time=time_limit,
        mem=memory,
        cluster_sbatch_lines=cluster_config.to_sbatch_lines(),
        log_dir=log_dir,
        n_trials=N_TRIALS,
        job_name=job_name,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        base_dir=BASE_DIR,
        model_family=model_family,
        model_size=model_size,
        task=task,
        graph_type=graph_type,
        study_name=study_name,
        ssh_tunnel_setup=ssh_tunnel_setup,
        ssh_tunnel_cleanup=ssh_tunnel_cleanup,
    )

    return script


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs for STS GIN Cayley Optuna (enhanced pooling) - STSBenchmark training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model, train on STSBenchmark (recommended)
    python3 generate_sts_gin_cayley_jobs.py --models Pythia-410m

    # All models, train on STSBenchmark
    python3 generate_sts_gin_cayley_jobs.py --all-models

Workflow:
    1. Generate jobs (trains on STSBenchmark only)
    2. Submit jobs with sbatch
    3. After Optuna completes, train best models
    4. Evaluate best models on all STS tasks to test generalization
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
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type (default: priority)"
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

    # Train on STSBenchmark only (fixed)
    tasks = [DEFAULT_TRAIN_TASK]

    # Get cluster info
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=args.partition)

    print(f"\n{'='*70}")
    print(f"Generating STS GIN Cayley Optuna jobs (Enhanced Pooling)")
    print(f"{'='*70}")
    print(f"Cluster: {cluster.upper()}")
    print(f"Partition: {cluster_config.partition}")
    print(f"Models: {', '.join(models)}")
    print(f"Graph types: {', '.join(GIN_V2_GRAPH_TYPES)}")
    print(f"Training task: {DEFAULT_TRAIN_TASK} (fixed - trains on STSBenchmark only)")
    print(f"Trials per job: {N_TRIALS}")
    print(f"\nCayley features:")
    print(f"  ✓ pool_real_nodes_only hyperparameter (True/False)")
    print(f"  ✓ train_eps hyperparameter (learnable epsilon, True/False)")
    print(f"  ✓ 'sum' pooling added to search space (mean, sum, attention)")
    print(f"  ✓ STSBenchmark-only training (evaluate on all STS tasks later)")

    total_jobs = len(models) * len(GIN_V2_GRAPH_TYPES)  # Only 1 task now
    print(f"\nTotal jobs: {total_jobs} ({len(models)} models × {len(GIN_V2_GRAPH_TYPES)} graph types)")
    print(f"\nWorkflow:")
    print(f"  1. Optuna tunes on STSBenchmark (this script)")
    print(f"  2. Train best models from Optuna results")
    print(f"  3. Evaluate on: {', '.join(EVAL_STS_TASKS)}")
    print(f"{'='*70}\n")

    job_count = 0
    base_output_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs")

    for model_name in models:
        for graph_type in GIN_V2_GRAPH_TYPES:
            # Create directory for this graph type
            if "-" in model_name:
                family, size = model_name.split("-", 1)
                model_str = f"{family.lower()}{size.lower()}"
            else:
                model_str = model_name.lower()

            job_dir_name = f"{model_str}_gin_sts_cayley_{graph_type}"
            job_dir = base_output_dir / job_dir_name
            job_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{model_name} + GIN-{graph_type}-cayley (STS):")
            print(f"  Output dir: {job_dir}")

            for task in tasks:
                job_script = create_sts_gin_cayley_job(
                    model_name, task, graph_type,
                    partition_type=args.partition
                )

                # Write job file
                job_filename = f"sts_gin_cayley_{graph_type}_{task}_{model_name.replace('-', '').lower()}.sh"
                job_path = job_dir / job_filename

                with open(job_path, "w", newline="\n") as f:
                    f.write(job_script)

                job_count += 1
                print(f"    [{job_count:3d}] {job_filename}")

    print(f"\n{'='*70}")
    print(f"✓ Generated {job_count} STS GIN Cayley job files")
    print(f"{'='*70}\n")

    # Print submit instructions
    print("To submit jobs:")
    for model_name in models:
        if "-" in model_name:
            family, size = model_name.split("-", 1)
            model_str = f"{family.lower()}{size.lower()}"
        else:
            model_str = model_name.lower()

        for graph_type in GIN_V2_GRAPH_TYPES:
            job_dir_name = f"{model_str}_gin_sts_cayley_{graph_type}"
            job_dir = base_output_dir / job_dir_name
            print(f"  # {model_name} + GIN-{graph_type}-cayley (STS)")
            print(f"  for job in {job_dir}/sts_gin_cayley_*.sh; do sbatch \"$job\"; done")

    print("\nTo monitor progress:")
    print(f"  squeue -u $USER | grep 'sts_gin_cayley'")
    print()


if __name__ == "__main__":
    main()
