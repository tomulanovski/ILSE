#!/usr/bin/env python3
"""
Unified pipeline wrapper for ILSE workflows (ALL MODELS USE PRECOMPUTED EMBEDDINGS).

This script provides a unified interface to run the complete ML pipeline for both
Classification and STS (Semantic Textual Similarity) tasks.

Classification Pipeline:
0. postgres: Submit PostgreSQL Optuna database job (one-time setup)
1. precompute: Generate and submit jobs to precompute embeddings
2. optuna: Generate Optuna hyperparameter search jobs (GIN/MLP/Weighted)
3. train: Query Optuna DB and generate training jobs for best models
4. eval: Generate evaluation jobs for trained models
5. summarize: Aggregate results to CSV

STS Pipeline:
1. sts-precompute: Precompute embeddings for sentence pairs
2. sts-optuna: Hyperparameter search for STS tasks (V1 or V2 with --v2 flag)
3. sts-train: Train best models from Optuna results
4. sts-eval: Evaluate on test sets (wrappers available)
5. sts-summarize: Aggregate STS results (supports --linear flag for filtering)

STS V2 Features (use --v2 flag):
- pool_real_nodes_only: FIXED to True - excludes virtual nodes from pooling (V1 pooled all nodes)
- train_eps: Searchable [True, False] - learnable epsilon in GIN aggregation (1-2 params)
- sum pooling: Added to search space alongside mean and attention
- STSBenchmark-only training: Train on STSBenchmark, evaluate on all STS tasks for generalization

Usage Examples (Classification):
    # Start PostgreSQL database (one-time, or when restarting)
    python3 pipeline.py postgres

    # Generate jobs only (manual submission)
    python3 pipeline.py precompute --model Pythia-410m
    python3 pipeline.py optuna --model Pythia-410m --methods gin mlp weighted

    # Auto-submit jobs after generating
    python3 pipeline.py precompute --model Pythia-410m --submit
    python3 pipeline.py optuna --model Pythia-410m --methods gin --submit

    # Complete workflow
    python3 pipeline.py train --model Pythia-410m
    python3 pipeline.py eval --model Pythia-410m
    python3 pipeline.py summarize --output pythia410m_results.csv --filter-model Pythia_410m

Usage Examples (STS):
    # V1 workflow: Standard STS training
    python3 pipeline.py sts-precompute --model Pythia-410m --submit
    python3 pipeline.py sts-optuna --model Pythia-410m --methods gin mlp weighted --submit
    python3 pipeline.py sts-train --model Pythia-410m --submit  # Trains V1 models

    # V2 workflow: pool_real_nodes_only=True, train_eps searchable, sum pooling
    python3 pipeline.py sts-precompute --model Pythia-410m --tasks STSBenchmark --submit
    python3 pipeline.py sts-optuna --model Pythia-410m --methods gin --v2 --submit
    python3 pipeline.py sts-train --model Pythia-410m --v2 --submit  # Trains V2 models
    python3 pipeline.py sts-eval --model Pythia-410m --submit

    # Transfer learning (optional): Use Pythia-410m hyperparameters for larger models
    python3 pipeline.py sts-train --model TinyLlama-1.1B --transfer-from Pythia-410m --submit

Requirements:
    - .env file must be configured (copy from .env.example)
    - All settings loaded from environment variables
"""

import argparse
import subprocess
import sys
import os
import glob
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Add scripts to path for cluster_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts_and_jobs"))
from cluster_config import get_cluster_type

# Verify .env exists
if not os.path.exists(".env"):
    print("ERROR: .env file not found!")
    print("Please copy .env.example to .env and configure your settings:")
    print("  cp .env.example .env")
    print("  nano .env  # Edit with your paths and credentials")
    sys.exit(1)

# Model configurations
SUPPORTED_MODELS = ["TinyLlama-1.1B", "Pythia-410m", "Pythia-2.8b", "Llama3-8B"]
MODEL_FAMILY_MAP = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B"),
    "Pythia-410m": ("Pythia", "410m"),
    "Pythia-2.8b": ("Pythia", "2.8b"),
    "Llama3-8B": ("Llama3", "8B"),
}
# Note: Layer counts are now determined dynamically from model configs
# This dict is kept for backward compatibility but should not be used
# Layer counts are calculated as num_hidden_layers + 1 (including embedding layer)

# STS task configuration
# Note: Will attempt all tasks - those without train splits will fail gracefully
ALL_STS_TASKS = [
    "STSBenchmark",  # Confirmed: has train split ✓
    "STS12",         # Confirmed: has train split ✓
    "STS13",         # Confirmed: evaluation-only (test only)
    "STS14",         # Unknown - will test
    "STS15",         # Unknown - will test
    "STS16",         # Unknown - will test
    "STS17",         # Unknown - will test
    "STS22",         # Unknown - will test
    "BIOSSES",       # Confirmed: evaluation-only (test only)
    "SICK-R",        # Confirmed: evaluation-only (test only)
]
RECOMMENDED_STS_TASKS = ["STSBenchmark", "STS12"]  # Confirmed trainable - for quick tests

# Retrieval task configuration (BEIR datasets)
# Start with smaller datasets for validation, then scale to MS MARCO
ALL_RETRIEVAL_TASKS = [
    # Small datasets for initial validation
    "SciFact",       # Small: 5k documents, scientific claims
    "NFCorpus",      # Small: 3.6k documents, medical/nutrition
    "FiQA",          # Medium: 57k documents, financial QA
    # Medium datasets
    "TRECCOVID",     # Medium: 171k documents, COVID scientific
    "ArguAna",       # Medium: argument retrieval
    # Large datasets (use after validation)
    "MSMARCO",       # Large: 8.8M passages, general domain (Phase 2)
    "NQ",            # Medium: Natural Questions
    "HotpotQA",      # Medium: multi-hop QA
]
RECOMMENDED_RETRIEVAL_TASKS = ["SciFact", "NFCorpus"]  # Small tasks for initial validation

# Classification tasks - exclude massive tasks by default
ALL_CLASSIFICATION_TASKS = [
    "EmotionClassification",
    "Banking77Classification",
    "MTOPIntentClassification",
    "MTOPDomainClassification",
    "PoemSentimentClassification",
]
# Massive tasks (excluded from default task list)
MASSIVE_TASKS = [
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
]


def get_optuna_storage_url_with_tunnel():
    """
    Get Optuna storage URL, setting up SSH tunnel if needed (CS cluster).
    
    On CS cluster: Sets up SSH tunnel to Bio cluster PostgreSQL and returns localhost URL.
    On Bio cluster: Uses dynamic detection from optuna_storage.py.
    
    Returns:
        tuple: (storage_url, error_message) - error_message is None on success
    """
    cluster = get_cluster_type()
    
    if cluster == 'cs':
        # CS cluster: Need SSH tunnel to Bio cluster PostgreSQL
        print("Setting up SSH tunnel to Bio cluster PostgreSQL...")
        
        bio_login = os.getenv("GNN_BIO_LOGIN", "USERNAME@your-cluster-login-node")
        ssh_key = os.getenv("GNN_SSH_KEY", "/home/USERNAME/.ssh/id_rsa")
        local_port = 5433
        remote_port = 5432
        
        # Get PostgreSQL host from Bio cluster
        try:
            result = subprocess.run(
                ["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no", 
                 "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                 bio_login, "cat ~/.optuna_db_host 2>/dev/null || echo 'your-compute-node'"],
                capture_output=True, text=True, timeout=30
            )
            db_host = result.stdout.strip().split('.')[0]  # Strip domain suffix
            if not db_host:
                db_host = "your-compute-node"
            print(f"  PostgreSQL running on Bio node: {db_host}")
        except Exception as e:
            return None, f"Failed to get DB host from Bio cluster: {e}"
        
        # Check if tunnel already exists on the port
        try:
            check_result = subprocess.run(
                ["ss", "-tuln"], capture_output=True, text=True
            )
            tunnel_exists = f":{local_port} " in check_result.stdout
        except:
            # Fallback: try netstat
            try:
                check_result = subprocess.run(
                    ["netstat", "-tuln"], capture_output=True, text=True
                )
                tunnel_exists = f":{local_port} " in check_result.stdout
            except:
                tunnel_exists = False
        
        if not tunnel_exists:
            # Create SSH tunnel
            try:
                subprocess.run([
                    "ssh", "-i", ssh_key, "-f", "-N", 
                    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", 
                    "-o", "LogLevel=ERROR",
                    "-L", f"{local_port}:{db_host}:{remote_port}",
                    bio_login
                ], check=True)
                time.sleep(2)  # Wait for tunnel to establish
                print(f"  ✓ SSH tunnel established: localhost:{local_port} -> {db_host}:{remote_port}")
            except Exception as e:
                return None, f"Failed to establish SSH tunnel: {e}"
        else:
            print(f"  ✓ SSH tunnel already exists on port {local_port}")
        
        # Get credentials from .env
        storage_template = os.getenv("GNN_OPTUNA_STORAGE", "")
        if "@" in storage_template:
            credentials = storage_template.split("@")[0].replace("postgresql://", "")
            database = "optuna"
            if "/" in storage_template:
                database = storage_template.split("/")[-1]
            storage_url = f"postgresql://{credentials}@localhost:{local_port}/{database}"
        else:
            # Fallback: try to extract from GNN_OPTUNA_STORAGE or use generic placeholder
            storage_url = os.getenv("GNN_OPTUNA_STORAGE", f"postgresql://USERNAME:PASSWORD@localhost:{local_port}/optuna")
            if "USERNAME" in storage_url or "PASSWORD" in storage_url:
                print(f"  ⚠ WARNING: Using placeholder credentials. Please set GNN_OPTUNA_STORAGE in .env")
        
        # Mask password in output
        masked_url = storage_url
        if '@' in storage_url and ':' in storage_url.split('@')[0]:
            parts = storage_url.split('@')
            creds = parts[0].split(':')
            if len(creds) >= 2:
                masked_url = f"{creds[0]}:***@{parts[1]}"
        print(f"  ✓ Using: {masked_url}")
        return storage_url, None
    
    else:
        # Bio cluster: Use dynamic detection
        try:
            sys.path.insert(0, os.path.join(os.getcwd(), "scripts_and_jobs", "scripts"))
            from optuna_storage import get_optuna_storage_url
            storage_url = get_optuna_storage_url()
            if storage_url:
                # Mask password in output
                masked_url = storage_url
                if '@' in storage_url and ':' in storage_url.split('@')[0]:
                    parts = storage_url.split('@')
                    creds = parts[0].split(':')
                    if len(creds) >= 2:
                        masked_url = f"{creds[0]}:***@{parts[1]}"
                print(f"  ✓ Using: {masked_url}")
                return storage_url, None
            else:
                return None, "Could not determine storage URL from optuna_storage module"
        except ImportError:
            # Fallback to .env
            storage_url = os.getenv("GNN_OPTUNA_STORAGE")
            if storage_url:
                return storage_url, None
            return None, "GNN_OPTUNA_STORAGE not set in .env file"


def check_precomputed_embeddings(model, tasks=None):
    """Check if precomputed embeddings exist for the model and tasks."""
    if model not in MODEL_FAMILY_MAP:
        return False, f"Unknown model: {model}"

    model_family, model_size = MODEL_FAMILY_MAP[model]
    embeddings_base = Path("precomputed_embeddings") / f"{model_family}_{model_size}_mean_pooling"

    if not embeddings_base.exists():
        return False, f"Embeddings directory not found: {embeddings_base}"

    if tasks is None:
        # Check if any embeddings exist
        task_dirs = list(embeddings_base.glob("*Classification"))
        if not task_dirs:
            return False, f"No task embeddings found in: {embeddings_base}"
        return True, f"Found embeddings for {len(task_dirs)} tasks"

    # Check specific tasks
    missing_tasks = []
    for task in tasks:
        task_dir = embeddings_base / task
        if not task_dir.exists():
            missing_tasks.append(task)

    if missing_tasks:
        return False, f"Missing embeddings for tasks: {', '.join(missing_tasks)}"

    return True, f"All embeddings found"


def submit_jobs(job_pattern, log_dir=None, since_timestamp=None):
    """
    Submit SLURM jobs matching the pattern.

    Args:
        job_pattern: Glob pattern for job files
        log_dir: Optional log directory to create
        since_timestamp: Optional timestamp - only submit files modified after this time
    """
    # Create log directory if specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"✓ Created log directory: {log_dir}")

    # Find all job files matching pattern
    job_files = glob.glob(job_pattern)

    # Filter by modification time if timestamp provided
    if since_timestamp:
        original_count = len(job_files)
        job_files = [f for f in job_files if os.path.getmtime(f) > since_timestamp]
        filtered_count = original_count - len(job_files)
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} old job file(s)")

    if not job_files:
        if since_timestamp:
            print(f"⚠️  No NEW job files found (all files are older than generation timestamp)")
        else:
            print(f"WARNING: No job files found matching: {job_pattern}")
        return 0

    print(f"\nSubmitting {len(job_files)} job(s)...")
    submitted = 0
    failed = 0

    for job_file in job_files:
        job_name = os.path.basename(job_file)
        print(f"  Submitting: {job_name}...", end=" ")

        result = subprocess.run(
            ["sbatch", job_file],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            # Parse job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            print(f"✓ Job ID: {job_id}")
            submitted += 1
        else:
            print(f"✗ FAILED")
            print(f"    Error: {result.stderr.strip()}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Submission complete: {submitted} submitted, {failed} failed")
    print(f"{'='*70}")

    return submitted


def run_precompute(model, submit=False, tasks=None, partition='priority'):
    """Generate precompute embeddings jobs using unified generator."""
    print(f"\n{'='*70}")
    print(f"STEP 1: Generating precompute embeddings jobs for {model}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_precompute_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    cmd = ["python3", generator_script, "--models", model, "--partition", partition]
    if tasks:
        cmd.extend(["--tasks"] + tasks)
    else:
        # Default: use all tasks except massive ones
        cmd.extend(["--tasks"] + ALL_CLASSIFICATION_TASKS)

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        # Get job pattern (keep dots in version number)
        model_dir_name = model.replace("-", "").lower() + "_precompute"
        job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
        log_dir = f"{base_dir}/job_logs/{model_dir_name}"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit jobs manually:")
            print(f"  mkdir -p {log_dir}")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py precompute --model {model} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep precomp")
    else:
        print(f"\n✗ Job generation failed!")


def run_optuna(model, methods, submit=False, tasks=None, partition='priority', graph_type=None):
    """Generate Optuna hyperparameter search jobs using unified generator."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Generating Optuna jobs for {model}")
    print(f"Methods: {', '.join(methods)}")
    if graph_type:
        print(f"Graph Type Filter: {graph_type} only (applies to 'gin' method)")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    # Check if precomputed embeddings exist
    print("Checking for precomputed embeddings...")
    exists, message = check_precomputed_embeddings(model, tasks)
    print(f"  {message}")

    if not exists:
        print(f"\n❌ ERROR: Precomputed embeddings not found!")
        print(f"\nYou must run precompute step first:")
        print(f"  python3 pipeline.py precompute --model {model}")
        print(f"\nThis will generate embeddings that ALL methods (GIN, MLP, Weighted) can reuse.")
        return

    print("  ✓ Precomputed embeddings found!\n")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_optuna_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    cmd = ["python3", generator_script, "--models", model, "--methods"] + methods + ["--partition", partition]
    if tasks:
        cmd.extend(["--tasks"] + tasks)
    else:
        # Default: use all tasks except massive ones
        cmd.extend(["--tasks"] + ALL_CLASSIFICATION_TASKS)
    if graph_type:
        cmd.extend(["--graph_type", graph_type])

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            for method in methods:
                # Keep dots in version number
                model_dir_name = model.replace("-", "").lower() + f"_{method}_optuna"
                job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
                print(f"\n{method.upper()}:")
                submit_jobs(job_pattern, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all generated jobs manually:")
            for method in methods:
                # Keep dots in version number
                model_dir_name = model.replace("-", "").lower() + f"_{method}_optuna"
                job_dir = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}"
                print(f"  # {method.upper()}")
                print(f"  for job in {job_dir}/*.sh; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py optuna --model {model} --methods {' '.join(methods)} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep -E '{'|'.join(methods)}'")
    else:
        print(f"\n✗ Job generation failed!")


def run_train(model, submit=False, filter_study=None, filter_task=None, graph_type=None):
    """Query Optuna DB and generate training jobs for best models."""
    print(f"\n{'='*70}")
    print(f"STEP 3: Training best models from Optuna results")
    if graph_type:
        print(f"Graph Type Filter: {graph_type} (will filter by trial params, not study name)")
    print(f"{'='*70}\n")

    # Get storage URL with automatic SSH tunnel for CS cluster
    storage_url, error = get_optuna_storage_url_with_tunnel()
    if error:
        print(f"ERROR: {error}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())

    # Build command
    cmd = [
        "python3",
        "scripts_and_jobs/scripts/train_best_models_from_optuna.py",
        "--storage_url", storage_url,
    ]

    # Add filters
    if model:
        # Convert model format (e.g., TinyLlama-1.1B → TinyLlama_1.1B)
        model_filter = model.replace("-", "_")
        cmd.extend(["--filter_model", model_filter])

        # Transfer learning: Use Pythia-410m hyperparameters for larger models
        if model in ["TinyLlama-1.1B", "Llama3-8B"]:
            cmd.extend(["--source_model", "Pythia_410m"])
            print(f"\n🔄 Transfer Learning: Using Pythia-410m hyperparameters for {model}")

    # Add filter_study if provided (for filtering studies by name pattern)
    if filter_study:
        cmd.extend(["--filter_study", filter_study])

    # Add filter_graph_type if provided (filters by trial.params['graph_type'])
    # This works because train_best_models_from_optuna.py groups trials by graph_type
    if graph_type:
        cmd.extend(["--filter_graph_type", graph_type])
        print(f"Will filter configurations by graph_type='{graph_type}' in trial params")

    if filter_task:
        cmd.extend(["--filter_task", filter_task])

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Training job files generated!")

        job_pattern = "scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_*.sh"
        log_dir = f"{base_dir}/job_logs/train_best_models"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit jobs manually:")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nTo submit by method:")
            print(f"  for job in scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_gin_*.sh; do sbatch \"$job\"; done")
            print(f"  for job in scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_mlp_*.sh; do sbatch \"$job\"; done")
            print(f"  for job in scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_weighted_*.sh; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py train --model {model} --submit")
    else:
        print(f"\n✗ Training job generation failed!")


def run_eval(model, submit=False, filter_task=None, graph_type=None, use_precomputed=False, tasks=None, filter_method=None):
    """Generate evaluation jobs for trained models."""
    print(f"\n{'='*70}")
    print(f"STEP 4: Generating evaluation jobs")
    if use_precomputed:
        print(f"Mode: Precomputed embeddings (MTEB-style, faster)")
    else:
        print(f"Mode: Regular MTEB evaluation")
    if filter_method:
        print(f"Method Filter: {filter_method}")
    if graph_type:
        print(f"Graph Type Filter: {graph_type}")
    if tasks:
        print(f"Tasks: {', '.join(tasks)}")
    else:
        print(f"Tasks: Default (5 classification tasks, excluding Massive)")
    print(f"{'='*70}\n")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())

    if use_precomputed:
        # Use new precomputed embeddings evaluation
        cmd = ["python3", "scripts_and_jobs/scripts/generate_precomputed_eval_jobs.py"]
        job_dir = "eval_precomputed"
        job_pattern = "scripts_and_jobs/slurm_jobs/generated_jobs/eval_precomputed/eval_precomp_*.sh"
    else:
        # Use regular MTEB evaluation
        cmd = ["python3", "scripts_and_jobs/scripts/generate_eval_jobs.py"]
        job_dir = "eval_best_models"
        job_pattern = "scripts_and_jobs/slurm_jobs/generated_jobs/eval_best_models/eval_*.sh"

    if model:
        # Parse model family and size, combine for filter_model
        if model in MODEL_FAMILY_MAP:
            family, size = MODEL_FAMILY_MAP[model]
            filter_model = f"{family}_{size}"
            cmd.extend(["--filter_model", filter_model])

    if filter_method:
        cmd.extend(["--filter_method", filter_method])

    if filter_task:
        cmd.extend(["--filter_task", filter_task])

    if graph_type:
        # Both regular and precomputed eval support graph_type filtering
        cmd.extend(["--filter_graph_type", graph_type])

    if tasks:
        # Pass specific tasks (only supported in precomputed eval for now)
        if use_precomputed:
            cmd.extend(["--tasks"] + tasks)
        else:
            print("⚠️  Warning: --tasks filter only supported in precomputed eval mode")

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Evaluation job files generated!")

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            log_dir = f"{base_dir}/job_logs/{job_dir}"
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit jobs manually:")
            if use_precomputed:
                print(f"  for job in scripts_and_jobs/slurm_jobs/generated_jobs/eval_precomputed/eval_precomp_*.sh; do sbatch \"$job\"; done")
            else:
                print(f"  for job in scripts_and_jobs/slurm_jobs/generated_jobs/eval_best_models/eval_*.sh; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py eval --model {model} --submit")
    else:
        print(f"\n✗ Evaluation job generation failed!")


def run_layer_baseline(model, layers=None, tasks=None, submit=False, all_sts_tasks=False, partition='priority'):
    """Generate layer baseline evaluation jobs."""
    print(f"\n{'='*70}")
    print(f"STEP: Layer Baseline Evaluation for {model}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_layer_baseline_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    # Note: num_layers is determined dynamically by the generator script
    cmd = ["python3", generator_script, "--models", model, "--partition", partition]

    if layers:
        cmd.extend(["--layers"] + [str(l) for l in layers])
    else:
        cmd.append("--all-layers")

    if all_sts_tasks:
        cmd.append("--all-sts-tasks")
    elif tasks:
        cmd.extend(["--tasks"] + tasks)
    else:
        # Default: use all tasks except massive ones
        cmd.extend(["--tasks"] + ALL_CLASSIFICATION_TASKS)

    print(f"Model: {model}")
    print(f"Layers: {'all' if not layers else layers}")

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        model_dir_name = model.replace("-", "").lower() + "_layer_baseline"
        job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/layer_*.sh"
        model_family, model_size = MODEL_FAMILY_MAP[model]
        log_dir = f"{base_dir}/job_logs/layer_baseline_{model_family}_{model_size}"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit manually:")
            print(f"  mkdir -p {log_dir}")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py layer-baseline --model {model} --submit")

        print(f"\nResults will be in:")
        print(f"  results/layer_baselines/{model_family}/{model_size}/")
    else:
        print(f"\n✗ Job generation failed!")


def run_summarize(output_file, model=None, filter_model=None, filter_task=None, include_layers=True, use_precomputed=False):
    """Summarize evaluation results to CSV."""
    print(f"\n{'='*70}")
    print(f"STEP 5: Summarizing results to CSV")
    print(f"Output: {output_file}")
    if include_layers and model:
        print(f"Including layer baselines: YES")
    if use_precomputed:
        print(f"Using precomputed results: YES")
    print(f"{'='*70}\n")

    # Choose base directory based on evaluation mode
    base_dir = "mteb_results_precomputed" if use_precomputed else "mteb_results_best_models"

    # Choose summarizer based on whether we want layer baselines
    if include_layers and model:
        cmd = [
            "python3",
            "scripts_and_jobs/scripts/summarize_with_layers.py",
            "--model", model,
            "--base_dir", base_dir,
            "--output_dir", "./summary",
            "--output", output_file,
        ]
    else:
        cmd = [
            "python3",
            "scripts_and_jobs/scripts/summarize_to_csv.py",
            "--base_dir", base_dir,
            "--output_dir", "./summary",
            "--output_filename", output_file,
        ]

    if filter_model:
        cmd.extend(["--filter-model" if include_layers and model else "--filter_model", filter_model])

    if filter_task:
        cmd.extend(["--filter-task" if include_layers and model else "--filter_task", filter_task])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Results summarized successfully!")
        print(f"Output file: summary/{output_file}")
    else:
        print(f"\n✗ Summarization failed!")


def run_postgres():
    """Submit PostgreSQL database job."""
    print(f"\n{'='*70}")
    print(f"Starting PostgreSQL Optuna Database")
    print(f"{'='*70}\n")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    postgres_job = f"{base_dir}/scripts_and_jobs/slurm_jobs/generated_jobs/postgres.sh"

    if not os.path.exists(postgres_job):
        print(f"ERROR: PostgreSQL job script not found: {postgres_job}")
        print(f"Expected location: scripts_and_jobs/slurm_jobs/generated_jobs/postgres.sh")
        sys.exit(1)

    print(f"Submitting PostgreSQL job...")
    print(f"  Script: {postgres_job}")
    print(f"  Log: job_logs/postgres.out")
    print()

    result = subprocess.run(["sbatch", postgres_job], capture_output=True, text=True)

    if result.returncode == 0:
        # Extract job ID from sbatch output
        job_id = result.stdout.strip().split()[-1] if result.stdout else "unknown"
        print(f"✓ PostgreSQL job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print()
        print(f"Monitor with:")
        print(f"  squeue -u $USER | grep postgres")
        print(f"  tail -f job_logs/postgres.out")
        print()
        print(f"Once running, the database host will be auto-written to:")
        print(f"  ~/.optuna_db_host")
        print()
        print(f"Other jobs will automatically detect and connect to this database.")
    else:
        print(f"✗ Failed to submit PostgreSQL job!")
        print(f"Error: {result.stderr}")
        sys.exit(1)


def run_sts_precompute(model, submit=False, tasks=None, partition='priority'):
    """Generate STS precompute embeddings jobs."""
    print(f"\n{'='*70}")
    print(f"STS STEP 1: Generating precompute embeddings jobs for {model}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_sts_precompute_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    cmd = ["python3", generator_script, "--models", model, "--partition", partition]
    if tasks:
        cmd.extend(["--tasks"] + tasks)
    else:
        # Default to all confirmed trainable tasks (STS tasks, not classification)
        cmd.append("--all-tasks")

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        # Get job pattern
        model_dir_name = model.replace("-", "").lower() + "_sts_precompute"
        job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
        log_dir = f"{base_dir}/job_logs/{model_dir_name}"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit jobs manually:")
            print(f"  mkdir -p {log_dir}")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py sts-precompute --model {model} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep sts_precomp")
    else:
        print(f"\n✗ Job generation failed!")


def run_sts_optuna(model, methods, submit=False, tasks=None, partition='priority', v2=False):
    """Generate STS Optuna hyperparameter search jobs."""
    version_str = "V2 (Enhanced)" if v2 else "V1"
    print(f"\n{'='*70}")
    print(f"STS STEP 2: Generating Optuna jobs for {model} ({version_str})")
    print(f"Methods: {', '.join(methods)}")
    if v2:
        print(f"V2 Features:")
        print(f"  - pool_real_nodes_only: FIXED to True (pools only real nodes)")
        print(f"  - train_eps: Searchable [True, False]")
        print(f"  - node_to_choose: Searchable [mean, sum]")
        print(f"V2 Training: STSBenchmark only (evaluate on all STS tasks later)")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    # V2 workflow: Check for GIN method only (V2 is GIN-specific)
    if v2 and "gin" not in methods:
        print(f"❌ ERROR: V2 workflow is currently only supported for GIN method")
        print(f"   Requested methods: {', '.join(methods)}")
        print(f"   V2 supports: gin")
        return

    # Check if precomputed STS embeddings exist
    model_family, model_size = MODEL_FAMILY_MAP[model]
    embeddings_base = Path("precomputed_embeddings_sts") / f"{model_family}_{model_size}_mean_pooling"

    print("Checking for precomputed STS embeddings...")
    if not embeddings_base.exists():
        print(f"❌ ERROR: STS embeddings directory not found: {embeddings_base}")
        print(f"\nYou must run sts-precompute step first:")
        print(f"  python3 pipeline.py sts-precompute --model {model} --submit")
        return

    # V2: Validate STSBenchmark embeddings specifically
    if v2:
        stsbenchmark_h5 = embeddings_base / "STSBenchmark.h5"
        if not stsbenchmark_h5.exists():
            print(f"❌ ERROR: V2 requires STSBenchmark embeddings: {stsbenchmark_h5}")
            print(f"Available H5 files: {[f.name for f in embeddings_base.glob('*.h5')]}")
            print(f"\nPrecompute STSBenchmark embeddings first:")
            print(f"  python3 pipeline.py sts-precompute --model {model} --tasks STSBenchmark --submit")
            return
        print(f"  ✓ STSBenchmark embeddings found for V2 training")
        # V2 ignores --tasks argument, always trains on STSBenchmark only
        tasks = None  # Will be handled by V2 generator
    elif not tasks:
        # V1: Auto-detect available tasks from H5 files if not specified
        # Discover which H5 files exist
        available_h5_files = list(embeddings_base.glob("*.h5"))
        if not available_h5_files:
            print(f"❌ ERROR: No H5 files found in {embeddings_base}")
            print(f"Available files: {list(embeddings_base.iterdir())}")
            return

        # Extract task names from filenames (e.g., "STSBenchmark.h5" -> "STSBenchmark")
        tasks = [h5_file.stem for h5_file in available_h5_files]
        print(f"  ✓ Auto-detected {len(tasks)} tasks with embeddings: {', '.join(tasks)}")
    else:
        # V1: Validate provided tasks have embeddings
        missing_tasks = []
        for task in tasks:
            h5_file = embeddings_base / f"{task}.h5"
            if not h5_file.exists():
                missing_tasks.append(task)
        if missing_tasks:
            print(f"❌ ERROR: Missing embeddings for tasks: {', '.join(missing_tasks)}")
            print(f"Available H5 files: {[f.name for f in embeddings_base.glob('*.h5')]}")
            return
        print(f"  ✓ Verified embeddings exist for {len(tasks)} tasks")

    print()

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())

    # Select generator script based on version
    if v2:
        generator_script = "scripts_and_jobs/generate_sts_gin_cayley_jobs.py"
    else:
        generator_script = "scripts_and_jobs/generate_sts_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    if v2:
        # V2: Only supports GIN, trains on STSBenchmark only
        cmd = ["python3", generator_script, "--models", model, "--partition", partition]
        # V2 generator doesn't accept --tasks or --methods (fixed to GIN on STSBenchmark)
    else:
        # V1: Original workflow with tasks and methods
        cmd = ["python3", generator_script, "--models", model, "--methods"] + methods + ["--partition", partition]
        cmd.extend(["--tasks"] + tasks)

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            if v2:
                # V2: GIN only, cayley graph type
                model_str = model.replace("-", "").lower()
                model_dir_name = f"{model_str}_gin_sts_v2_cayley"
                job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
                print(f"\nGIN V2 (STS - cayley):")
                submit_jobs(job_pattern, since_timestamp=before_generation)
            else:
                # V1: Multiple methods
                for method in methods:
                    model_dir_name = model.replace("-", "").lower() + f"_{method}_sts"
                    job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
                    print(f"\n{method.upper()} (STS):")
                    submit_jobs(job_pattern, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all generated jobs manually:")
            if v2:
                # V2: GIN only, cayley graph type
                model_str = model.replace("-", "").lower()
                model_dir_name = f"{model_str}_gin_sts_v2_cayley"
                job_dir = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}"
                print(f"  # GIN V2 (STS - cayley)")
                print(f"  for job in {job_dir}/*.sh; do sbatch \"$job\"; done")
            else:
                # V1: Multiple methods
                for method in methods:
                    model_dir_name = model.replace("-", "").lower() + f"_{method}_sts"
                    job_dir = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}"
                    print(f"  # {method.upper()} (STS)")
                    print(f"  for job in {job_dir}/*.sh; do sbatch \"$job\"; done")
            v2_flag = " --v2" if v2 else ""
            print(f"\nOr use: python3 pipeline.py sts-optuna --model {model} --methods {' '.join(methods)}{v2_flag} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep sts")
    else:
        print(f"\n✗ Job generation failed!")


def run_sts_train(model, submit=False, filter_study=None, filter_task=None, transfer_from=None, v2=False):
    """Query Optuna DB and generate STS training jobs for best models."""
    version_str = "V2" if v2 else "V1"
    print(f"\n{'='*70}")
    print(f"STS STEP 3: Training best STS models from Optuna results ({version_str})")
    print(f"{'='*70}\n")

    # Get storage URL with automatic SSH tunnel for CS cluster
    storage_url, error = get_optuna_storage_url_with_tunnel()
    if error:
        print(f"ERROR: {error}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    train_script = "scripts_and_jobs/scripts/train_best_models_from_optuna.py"

    if not os.path.exists(train_script):
        print(f"ERROR: Training script not found: {train_script}")
        return

    # Build command (filter for STS studies)
    cmd = ["python3", train_script, "--storage_url", storage_url]

    # Add V2 flag if requested
    if v2:
        cmd.append("--v2")

    if filter_study:
        # Don't add "sts_" prefix if user already provides a method-specific filter
        # Check if filter is a method name (e.g., "deepset", "mlp") or starts with method_
        # Also check for linear variants (e.g., "linear_gin", "linear_mlp", "linear_deepset")
        method_names = ["gin", "mlp", "weighted", "deepset"]
        is_method_filter = (
            filter_study in method_names or
            any(filter_study.startswith(m + "_") for m in method_names) or
            (filter_study.startswith("linear_") and 
             any(filter_study[len("linear_"):] == m or 
                 filter_study[len("linear_"):].startswith(m + "_") for m in method_names))
        )
        if is_method_filter:
            cmd.extend(["--filter_study", filter_study])
        else:
            cmd.extend(["--filter_study", f"sts_{filter_study}"])
    else:
        # Default: only STS studies
        cmd.extend(["--filter_study", "sts_"])

    if model:
        model_family, model_size = MODEL_FAMILY_MAP[model]
        cmd.extend(["--filter_model", f"{model_family}_{model_size}"])

        # Transfer learning: Optional, use if --transfer-from is specified
        if transfer_from:
            transfer_model_family, transfer_model_size = MODEL_FAMILY_MAP[transfer_from]
            cmd.extend(["--source_model", f"{transfer_model_family}_{transfer_model_size}"])
            print(f"\n🔄 Transfer Learning: Using {transfer_from} hyperparameters for {model}")
        else:
            print(f"\n📊 Using native {model} Optuna results (no transfer)")

    if filter_task:
        cmd.extend(["--filter_task", f"sts_{filter_task}"])

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Training job files generated successfully!")

        job_pattern = "scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_*_sts_*.sh"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all generated jobs manually:")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py sts-train --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep train")
    else:
        print(f"\n✗ Training job generation failed!")


def run_sts_eval(model, submit=False, filter_study=None, filter_task=None, eval_tasks=None, partition='priority', v2=False):
    """Generate STS evaluation jobs for trained models."""
    version_str = "V2" if v2 else "V1"
    print(f"\n{'='*70}")
    print(f"STS STEP 4: Generating evaluation jobs for trained STS models ({version_str})")
    print(f"{'='*70}\n")

    if not model:
        raise ValueError("--model required for sts-eval step (e.g., Pythia-410m)")

    # Build command
    cmd = ["python3", "scripts_and_jobs/generate_sts_eval_jobs.py", "--model", model]

    if v2:
        cmd.append("--v2")

    if filter_study:  # Use filter_study to filter by encoder (gin/mlp/weighted)
        cmd.extend(["--encoder", filter_study])

    if filter_task:
        cmd.extend(["--task", filter_task])

    if eval_tasks:  # Specify which tasks to evaluate on
        cmd.extend(["--eval_tasks"] + eval_tasks)

    if partition:
        cmd.extend(["--partition", partition])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n⚠️  Job generation failed!")
        return

    # Submit jobs if requested
    if submit:
        job_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs/sts_eval")
        job_files = list(job_dir.glob("*.sh"))

        if not job_files:
            print(f"\n⚠️  No job files found in {job_dir}")
            return

        # Filter by timestamp - only submit newly generated jobs
        import time
        current_time = time.time()
        recent_jobs = [f for f in job_files if (current_time - f.stat().st_mtime) < 60]

        if not recent_jobs:
            print(f"\n⚠️  No recently generated jobs found (all jobs are older than 1 minute)")
            print(f"This prevents re-submitting old jobs. Re-run the command to generate fresh jobs.")
            return

        print(f"\n{'='*70}")
        print(f"Submitting {len(recent_jobs)} STS evaluation job(s)...")
        print(f"{'='*70}\n")

        submitted = 0
        for job_file in recent_jobs:
            result = subprocess.run(["sbatch", str(job_file)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {job_file.name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"  ✗ {job_file.name}: {result.stderr.strip()}")

        print(f"\n✓ Submitted {submitted}/{len(recent_jobs)} job(s)")
        print(f"\nMonitor with:")
        print(f"  squeue -u $USER | grep eval_sts")
        print(f"  tail -f job_logs/sts_eval/*.out")


def run_sts_summarize(output, model=None, filter_model=None, filter_task=None, linear=False, v2=False):
    """Summarize STS results to CSV."""
    print(f"\n{'='*70}")
    v2_label = " (V2)" if v2 else ""
    print(f"STS STEP 5: Summarizing STS results{v2_label}")
    print(f"{'='*70}\n")

    if not model:
        print("ERROR: --model is required for sts-summarize step")
        print("Usage: python3 pipeline.py sts-summarize --model Pythia-410m --output sts_summary.csv")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    summarizer_script = "scripts_and_jobs/scripts/summarize_sts_results.py"

    if not os.path.exists(summarizer_script):
        print(f"ERROR: Summarizer script not found: {summarizer_script}")
        return

    # Build command
    cmd = [
        "python3",
        summarizer_script,
        "--model", model,
        "--base_dir", "results",
        "--output_dir", "./summary",
    ]

    if output:
        cmd.extend(["--output", output])

    if filter_task:
        cmd.extend(["--filter-task", filter_task])

    # Filter by linear vs non-linear models
    if linear:
        # Include only linear models
        cmd.extend(["--filter-encoder", "linear"])
        print(f"Filter: Linear models only")
    else:
        # Exclude linear models (default: non-linear only)
        cmd.extend(["--filter-encoder", "!linear"])
        print(f"Filter: Non-linear models only (excluding linear)")

    # V2 filtering
    if v2:
        cmd.append("--v2")
        print(f"Filter: V2 mode (GIN/GCN with _v2 + MLP/Weighted/DeepSet)")
    else:
        print(f"Filter: V1 mode (GIN/GCN without _v2 + MLP/Weighted/DeepSet)")

    print(f"Model: {model}")
    if filter_task:
        print(f"Filter task: {filter_task}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n✓ STS results summarized successfully!")
        if output:
            print(f"Output file: summary/{output}")
        else:
            model_short = model.replace("-", "").lower()
            print(f"Output file: summary/sts_summary_{model_short}.csv")
    else:
        print(f"\n✗ STS summarization failed!")


# ============================================================================
# Retrieval Pipeline Functions
# ============================================================================

def run_retrieval_precompute(model, submit=False, max_samples=50000, partition='priority'):
    """Generate MS MARCO triplet precompute embeddings jobs."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL STEP 1: Generating MS MARCO precompute job for {model}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    print(f"Dataset: MS MARCO triplets (query, positive, hard_negative)")
    print(f"Max samples: {max_samples:,}")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_retrieval_precompute_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command (no --tasks, always MS MARCO)
    cmd = ["python3", generator_script, "--models", model, "--partition", partition,
           "--max-samples", str(max_samples)]

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        # Get job pattern
        model_dir_name = model.replace("-", "").lower() + "_retrieval_precompute"
        job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
        log_dir = f"{base_dir}/job_logs/{model_dir_name}"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit jobs manually:")
            print(f"  mkdir -p {log_dir}")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py retrieval-precompute --model {model} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep retr_precomp")
    else:
        print(f"\n✗ Job generation failed!")


def run_retrieval_optuna(model, methods, submit=False, tasks=None, partition='priority'):
    """Generate retrieval Optuna hyperparameter search jobs."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL STEP 2: Generating Optuna jobs for {model}")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    # Use recommended tasks if none specified
    if not tasks:
        tasks = RECOMMENDED_RETRIEVAL_TASKS
        print(f"Using recommended tasks for validation: {', '.join(tasks)}")

    # Check if precomputed retrieval embeddings exist
    model_family, model_size = MODEL_FAMILY_MAP[model]
    embeddings_base = Path("precomputed_embeddings_retrieval") / f"{model_family}_{model_size}_mean_pooling"

    print("Checking for precomputed retrieval embeddings...")
    if not embeddings_base.exists():
        print(f"❌ ERROR: Retrieval embeddings directory not found: {embeddings_base}")
        print(f"\nYou must run retrieval-precompute step first:")
        print(f"  python3 pipeline.py retrieval-precompute --model {model} --submit")
        return

    print(f"  ✓ Embeddings directory exists: {embeddings_base}")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_retrieval_optuna_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        print(f"\nNote: Retrieval Optuna generator not yet created.")
        print(f"Use the retrieval trainer directly for now:")
        print(f"  python3 -m experiments.utils.model_definitions.gnn.retrieval.retrieval_trainer --help")
        return

    # Build command
    cmd = ["python3", generator_script, "--models", model, "--methods"] + methods + ["--partition", partition]
    cmd.extend(["--tasks"] + tasks)

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            for method in methods:
                model_dir_name = model.replace("-", "").lower() + f"_{method}_retrieval"
                job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
                print(f"\n{method.upper()} (Retrieval):")
                submit_jobs(job_pattern, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all generated jobs manually:")
            for method in methods:
                model_dir_name = model.replace("-", "").lower() + f"_{method}_retrieval"
                job_dir = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}"
                print(f"  # {method.upper()} (Retrieval)")
                print(f"  for job in {job_dir}/*.sh; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py retrieval-optuna --model {model} --methods {' '.join(methods)} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep retrieval")
    else:
        print(f"\n✗ Job generation failed!")


def run_retrieval_train_fixed(model, submit=False, partition='priority'):
    """Generate SLURM jobs for 6 fixed retrieval configurations (no Optuna)."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL STEP 2: Generating training jobs (6 fixed configs)")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        print(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
        return

    print(f"Model: {model}")
    print(f"Configs: 9 total")
    print(f"  GIN variants (6): 2 pooling × 3 variants (GCN, GIN-1, GIN-2)")
    print(f"  Baselines (3): DeepSet (sum), MLP (last, 2 layers), Weighted")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/generate_retrieval_train_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        return

    # Build command
    cmd = ["python3", generator_script, "--models", model, "--partition", partition]

    # Record timestamp before generation
    before_generation = time.time()

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Job files generated successfully!")

        # Get job pattern
        model_dir_name = model.replace("-", "").lower() + "_retrieval_train"
        job_pattern = f"scripts_and_jobs/slurm_jobs/generated_jobs/{model_dir_name}/*.sh"
        log_dir = f"{base_dir}/job_logs/{model_dir_name}"

        if submit:
            print(f"\n--- Auto-submitting 9 jobs in parallel ---")
            submit_jobs(job_pattern, log_dir, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all 9 jobs manually:")
            print(f"  mkdir -p {log_dir}")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py retrieval-train --model {model} --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep retr_train")
        print(f"  watch -n 5 'squeue -u $USER | grep retr_train'")
        print(f"\nModels saved to: saved_models/retrieval/")
    else:
        print(f"\n✗ Job generation failed!")


def run_retrieval_train(model, submit=False, filter_study=None, filter_task=None):
    """Query Optuna DB and generate retrieval training jobs for best models."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL STEP 3: Training best retrieval models from Optuna results")
    print(f"{'='*70}\n")

    # Get storage URL with automatic SSH tunnel for CS cluster
    storage_url, error = get_optuna_storage_url_with_tunnel()
    if error:
        print(f"ERROR: {error}")
        return

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    train_script = "scripts_and_jobs/scripts/train_best_retrieval_models_from_optuna.py"

    if not os.path.exists(train_script):
        print(f"ERROR: Training script not found: {train_script}")
        print(f"\nNote: Retrieval training from Optuna not yet implemented.")
        print(f"Use the retrieval trainer directly for now:")
        print(f"  python3 -m experiments.utils.model_definitions.gnn.retrieval.retrieval_trainer --help")
        return

    # Build command (filter for retrieval studies)
    cmd = ["python3", train_script, "--storage_url", storage_url]
    if filter_study:
        cmd.extend(["--filter_study", f"retrieval_{filter_study}"])
    else:
        # Default: only retrieval studies
        cmd.extend(["--filter_study", "retrieval_"])

    if model:
        model_family, model_size = MODEL_FAMILY_MAP[model]
        cmd.extend(["--filter_model", f"{model_family}_{model_size}"])

    if filter_task:
        cmd.extend(["--filter_task", filter_task])

    # Record timestamp before generation
    before_generation = time.time()

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Training job files generated successfully!")

        job_pattern = "scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models/train_retrieval_*.sh"

        if submit:
            print(f"\n--- Auto-submitting jobs ---")
            submit_jobs(job_pattern, since_timestamp=before_generation)
        else:
            print(f"\nTo submit all generated jobs manually:")
            print(f"  for job in {job_pattern}; do sbatch \"$job\"; done")
            print(f"\nOr use: python3 pipeline.py retrieval-train --submit")

        print(f"\nTo monitor:")
        print(f"  squeue -u $USER | grep train")
    else:
        print(f"\n✗ Training job generation failed!")


def run_retrieval_eval(model, submit=False, filter_task=None):
    """Generate retrieval evaluation jobs using MTEB retrieval benchmarks."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL STEP 4: Generating evaluation jobs for trained retrieval models")
    print(f"{'='*70}\n")

    print("⚠️  Retrieval evaluation uses MTEB retrieval benchmarks (nDCG@10, Recall@100).")
    print("This evaluates the trained encoder on held-out queries.\n")

    if not model:
        raise ValueError("--model required for retrieval-eval step (e.g., Pythia-410m)")

    base_dir = os.getenv('GNN_REPO_DIR', os.getcwd())
    generator_script = "scripts_and_jobs/scripts/generate_retrieval_eval_jobs.py"

    if not os.path.exists(generator_script):
        print(f"ERROR: Generator script not found: {generator_script}")
        print(f"This script should be at: {generator_script}")
        return

    # Build command
    cmd = ["python3", generator_script, "--model", model]

    if filter_task:
        cmd.extend(["--task", filter_task])

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n⚠️  Job generation failed!")
        return

    # Submit jobs if requested
    if submit:
        job_dir = Path("scripts_and_jobs/slurm_jobs/generated_jobs/eval_retrieval_models")
        job_files = list(job_dir.glob("*.sh"))

        if not job_files:
            print(f"\n⚠️  No job files found in {job_dir}")
            return

        print(f"\n{'='*70}")
        print(f"Submitting {len(job_files)} retrieval evaluation job(s)...")
        print(f"{'='*70}\n")

        submitted = 0
        for job_file in job_files:
            result = subprocess.run(["sbatch", str(job_file)], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ {job_file.name}: {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"  ✗ {job_file.name}: {result.stderr.strip()}")

        print(f"\n✓ Submitted {submitted}/{len(job_files)} job(s)")
        print(f"\nMonitor with:")
        print(f"  squeue -u $USER | grep eval_retr")


def run_retrieval_direct(model, task, encoder="gin", epochs=30, batch_size=64,
                         node_to_choose="mean", gin_mlp_layers=1, graph_type="cayley"):
    """Run retrieval training directly (without job generation) for quick testing."""
    print(f"\n{'='*70}")
    print(f"RETRIEVAL DIRECT: Training {encoder} on {task} with {model}")
    print(f"Config: graph={graph_type}, pooling={node_to_choose}, gin_mlp_layers={gin_mlp_layers}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        return

    model_family, model_size = MODEL_FAMILY_MAP[model]

    # Check for precomputed triplet embeddings
    embeddings_base = Path("precomputed_embeddings_retrieval") / f"{model_family}_{model_size}_mean_pooling"
    train_queries = embeddings_base / task / "train_queries.npy"
    train_positives = embeddings_base / task / "train_positives.npy"
    train_negatives = embeddings_base / task / "train_negatives.npy"
    val_queries = embeddings_base / task / "val_queries.npy"
    val_positives = embeddings_base / task / "val_positives.npy"
    val_negatives = embeddings_base / task / "val_negatives.npy"

    print("Checking for precomputed triplet embeddings...")
    missing = []
    for p in [train_queries, train_positives, train_negatives,
              val_queries, val_positives, val_negatives]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        print(f"❌ Missing embeddings:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nRun precompute step first:")
        print(f"  python3 pipeline.py retrieval-precompute --model {model} --tasks {task}")
        return

    print("  ✓ All triplet embeddings found!\n")

    # Determine method name (gcn if gin_mlp_layers=0, else gin)
    method = "gcn" if gin_mlp_layers == 0 else "gin"

    # Build training command
    cmd = [
        "python3", "-m", "experiments.utils.model_definitions.gnn.retrieval.retrieval_trainer",
        "--task", task,
        "--model_family", model_family,
        "--model_size", model_size,
        "--encoder", encoder,
        "--train_queries_path", str(train_queries),
        "--train_positives_path", str(train_positives),
        "--train_negatives_path", str(train_negatives),
        "--val_queries_path", str(val_queries),
        "--val_positives_path", str(val_positives),
        "--val_negatives_path", str(val_negatives),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--graph_type", graph_type,
        "--node_to_choose", node_to_choose,
        "--gin_mlp_layers", str(gin_mlp_layers),
        "--save_dir", "./saved_models/retrieval",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Training completed!")
        print(f"Model saved as: retrieval_{method}_{task}_{model_family}_{model_size}_{graph_type}.pt")
    else:
        print(f"\n✗ Training failed!")


def run_retrieval_train_all(model, task, epochs=30, batch_size=64):
    """
    Train all 6 retrieval models with fixed hyperparameters.

    Models trained:
    - 2 pooling methods: mean, attention
    - 3 gin_mlp_layers: 0 (GCN), 1 (GIN), 2 (GIN)
    = 6 models total
    """
    print(f"\n{'='*70}")
    print(f"RETRIEVAL TRAIN-ALL: Training 6 models on {task} with {model}")
    print(f"{'='*70}\n")

    if model not in SUPPORTED_MODELS:
        print(f"ERROR: Unsupported model: {model}")
        return

    # Fixed hyperparameters (from literature: E5, Contriever, DPR)
    configs = [
        # (pooling, gin_mlp_layers, description)
        ("mean", 0, "GCN + mean pooling"),
        ("mean", 1, "GIN-1 + mean pooling"),
        ("mean", 2, "GIN-2 + mean pooling"),
        ("sum", 0, "GCN + sum pooling"),
        ("sum", 1, "GIN-1 + sum pooling"),
        ("sum", 2, "GIN-2 + sum pooling"),
    ]

    results = []

    for i, (pooling, gin_mlp_layers, desc) in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/6] Training: {desc}")
        print(f"{'='*70}")

        run_retrieval_direct(
            model=model,
            task=task,
            encoder="gin",
            epochs=epochs,
            batch_size=batch_size,
            node_to_choose=pooling,
            gin_mlp_layers=gin_mlp_layers,
            graph_type="cayley"
        )

        results.append((desc, pooling, gin_mlp_layers))

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: All 6 models trained!")
    print(f"{'='*70}")
    print(f"\nModels saved in ./saved_models/retrieval/:")
    model_family, model_size = MODEL_FAMILY_MAP[model]
    for desc, pooling, gin_mlp_layers in results:
        method = "gcn" if gin_mlp_layers == 0 else f"gin{gin_mlp_layers}"
        print(f"  - retrieval_{method}_{task}_{model_family}_{model_size}_cayley_{pooling}.pt")

    print(f"\nNext step: Evaluate on MTEB retrieval tasks")
    print(f"  python3 pipeline.py retrieval-eval --model {model}")


def run_latex_table(input_files, output, model_names=None, task_col="Task", 
                    deltas=False, no_bold_best=False, caption=None, label=None):
    """
    Convert CSV result tables to LaTeX format.
    
    Args:
        input_files: List of CSV file paths
        output: Output LaTeX file path
        model_names: Optional list of model names (one per input file)
        task_col: Name of the task column
        deltas: Whether to include delta calculations
        no_bold_best: Whether to disable bolding of best values
        caption: LaTeX table caption (default: "Performance Results")
        label: LaTeX table label (default: "tab:results")
    """
    print(f"\n{'='*70}")
    print(f"LATEX TABLE: Converting CSV to LaTeX")
    print(f"{'='*70}\n")
    
    if not input_files:
        print("ERROR: --input is required for latex-table step")
        print("Example: python3 pipeline.py latex-table --input summary/results.csv --output table.tex")
        sys.exit(1)
    
    # Import the CSV to LaTeX converter
    script_path = os.path.join(os.path.dirname(__file__), "scripts_and_jobs", "scripts", "csv_to_latex.py")
    if not os.path.exists(script_path):
        print(f"ERROR: CSV to LaTeX script not found at {script_path}")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, script_path]
    cmd.extend(["--input"] + input_files)
    cmd.extend(["--output", output])
    cmd.extend(["--task_col", task_col])
    
    if model_names:
        cmd.extend(["--model_names"] + model_names)
    
    if deltas:
        cmd.append("--deltas")
    
    if no_bold_best:
        cmd.append("--no_bold_best")
    
    if caption:
        cmd.extend(["--caption", caption])
    
    if label:
        cmd.extend(["--label", label])
    
    # Run the script
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ LaTeX table generation failed!")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"LATEX TABLE: Conversion complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ILSE Pipeline: Unified workflow orchestration (ALL MODELS USE PRECOMPUTED EMBEDDINGS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Classification):
    # Start PostgreSQL database (one-time setup or restart)
    python3 pipeline.py postgres

    # Complete workflow for Pythia-410m
    python3 pipeline.py precompute --model Pythia-410m --submit
    python3 pipeline.py optuna --model Pythia-410m --methods gin mlp weighted --submit
    python3 pipeline.py train --model Pythia-410m --submit
    python3 pipeline.py eval --model Pythia-410m --submit
    python3 pipeline.py summarize --output pythia410m_results.csv --filter-model Pythia_410m

    # Run specific method only
    python3 pipeline.py optuna --model Pythia-410m --methods gin
    python3 pipeline.py train --model Pythia-410m --filter-study gin

    # Run with cayley graph type (for Pythia-410m only)
    python3 pipeline.py optuna --model Pythia-410m --methods gin --graph_type cayley --submit
    python3 pipeline.py train --model Pythia-410m --graph_type cayley --submit
    python3 pipeline.py eval --model Pythia-410m --submit

    # Use precomputed embeddings for faster evaluation (MTEB-style, no LLM loading)
    python3 pipeline.py eval --model Pythia-410m --use-precomputed --submit

Examples (STS - Semantic Textual Similarity):
    # V1 workflow: Complete STS for Pythia-410m (uses all confirmed trainable tasks by default)
    python3 pipeline.py sts-precompute --model Pythia-410m --submit
    python3 pipeline.py sts-optuna --model Pythia-410m --methods gin mlp weighted --submit
    python3 pipeline.py sts-train --model Pythia-410m --submit
    python3 pipeline.py sts-eval --model Pythia-410m

    # V2 workflow: Enhanced GIN with pool_real_nodes_only=True, train_eps, sum pooling
    python3 pipeline.py sts-precompute --model Pythia-410m --tasks STSBenchmark --submit
    python3 pipeline.py sts-optuna --model Pythia-410m --methods gin --v2 --submit
    python3 pipeline.py sts-train --model Pythia-410m --v2 --submit  # Train V2 models
    python3 pipeline.py sts-eval --model Pythia-410m --submit        # Evaluate on all STS tasks

    # Use specific STS tasks (V1 only, default: all confirmed trainable = STSBenchmark, STS12)
    python3 pipeline.py sts-precompute --model Pythia-410m --tasks STSBenchmark
    python3 pipeline.py sts-optuna --model Pythia-410m --tasks STSBenchmark --methods gin

    # Query native Optuna results for larger models (default behavior)
    python3 pipeline.py sts-train --model TinyLlama-1.1B --submit
    python3 pipeline.py sts-train --model Llama3-8B --submit

    # Transfer learning (optional): Use Pythia-410m hyperparameters for larger models
    python3 pipeline.py sts-train --model TinyLlama-1.1B --transfer-from Pythia-410m --submit
    python3 pipeline.py sts-train --model Llama3-8B --transfer-from Pythia-410m --submit

    # Evaluate on general partition (shared pool)
    python3 pipeline.py sts-eval --model Pythia-410m --partition general --submit

    # Linear methods (V2): Use linear_gin, linear_mlp, linear_deepset for true linear models (no ReLU)
    python3 pipeline.py sts-optuna --model Pythia-410m --methods linear_gin --submit
    python3 pipeline.py sts-train --model Pythia-410m --filter-study linear_gin --submit
    python3 pipeline.py sts-eval --model Pythia-410m --submit
    python3 pipeline.py sts-summarize --model Pythia-410m --linear --output sts_linear.csv

    # Summarize non-linear STS results (default behavior)
    python3 pipeline.py sts-summarize --model Pythia-410m --output sts_nonlinear.csv

Examples (Retrieval - MS MARCO training):
    # Step 1: Precompute MS MARCO triplet embeddings (50K samples)
    python3 pipeline.py retrieval-precompute --model Pythia-410m --submit

    # Step 2: Generate and submit 9 training jobs (runs in parallel on SLURM)
    python3 pipeline.py retrieval-train --model Pythia-410m --submit

    # Step 3: Evaluate zero-shot on MTEB retrieval tasks
    python3 pipeline.py retrieval-eval --model Pythia-410m --submit

9 models trained (all run in parallel as separate SLURM jobs):
    GIN variants (6 models):
      - GCN + mean pooling (gin_mlp_layers=0)
      - GIN-1 + mean pooling (gin_mlp_layers=1)
      - GIN-2 + mean pooling (gin_mlp_layers=2)
      - GCN + attention pooling
      - GIN-1 + attention pooling
      - GIN-2 + attention pooling
    Baselines (3 models):
      - DeepSet (pre=1, post=1, sum pooling)
      - MLP (last layer, 2 layers)
      - Weighted (learned layer weights)

All models saved to: saved_models/retrieval/
    retrieval_gcn_MSMARCO_Pythia_410m_cayley_mean.pt
    retrieval_deepset_MSMARCO_Pythia_410m_pre1_post1_sum.pt
    retrieval_mlp_MSMARCO_Pythia_410m_last_layers2.pt
    retrieval_weighted_MSMARCO_Pythia_410m_softmax.pt
    (+ 5 more GIN variants)
        """
    )

    parser.add_argument(
        "step",
        choices=[
            "postgres",
            "precompute", "optuna", "train", "eval", "layer-baseline", "summarize",
            "sts-precompute", "sts-optuna", "sts-train", "sts-eval", "sts-summarize",
            "latex-table"
        ],
        help="Pipeline step to execute (classification, STS, or latex-table)"
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        help="Model name (e.g., Pythia-410m, TinyLlama-1.1B)"
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["gin", "mlp", "weighted", "deepset", "dwatt", "linear_gin", "linear_mlp", "linear_deepset"],
        default=["gin", "mlp", "weighted", "deepset"],
        help="Methods for optuna step (default: all non-linear). DWAtt: dwatt. Linear methods: linear_gin, linear_mlp, linear_deepset"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific tasks to process (default: all tasks)"
    )

    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        help="For sts-eval: Specific STS tasks to evaluate on (default: all STS tasks). Example: --eval_tasks STS17 STS22"
    )

    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="Specific layer indices for layer-baseline step (default: all layers)"
    )
    parser.add_argument(
        "--all-sts-tasks",
        action="store_true",
        help="For layer-baseline: evaluate on all STS tasks only"
    )

    parser.add_argument(
        "--filter-study",
        help="Filter by study name pattern (e.g., 'gin', 'mlp', 'weighted')"
    )

    parser.add_argument(
        "--filter-task",
        help="Filter by task name (e.g., 'EmotionClassification')"
    )


    parser.add_argument(
        "--filter-model",
        help="Filter model for summarize step (e.g., 'Pythia_410m')"
    )

    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output filename for summarize step (default: results.csv) or LaTeX file for latex-table"
    )

    # LaTeX table arguments
    parser.add_argument(
        "--input",
        nargs="+",
        help="For latex-table: Input CSV file(s)"
    )
    
    parser.add_argument(
        "--model_names",
        nargs="+",
        help="For latex-table: Model names for multiple CSV files"
    )
    
    parser.add_argument(
        "--task_col",
        type=str,
        default="Task",
        help="For latex-table: Name of the task column (default: 'Task')"
    )
    
    parser.add_argument(
        "--deltas",
        action="store_true",
        help="For latex-table: Include delta calculations"
    )
    
    parser.add_argument(
        "--no_bold_best",
        action="store_true",
        help="For latex-table: Disable bolding of best values"
    )
    
    parser.add_argument(
        "--caption",
        type=str,
        help="For latex-table: LaTeX table caption"
    )
    
    parser.add_argument(
        "--label",
        type=str,
        help="For latex-table: LaTeX table label"
    )

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Auto-submit jobs after generating (default: False, manual submission)"
    )

    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition (default), 'general' for shared pool"
    )

    parser.add_argument(
        "--graph_type",
        type=str,
        choices=["fully_connected", "cayley"],
        help="Filter by graph_type (e.g., 'cayley'). In optuna step: only tests this graph_type (study name stays the same). In train step: filters configs by trial params."
    )

    parser.add_argument(
        "--filter-method",
        "--method",
        dest="filter_method",
        type=str,
        choices=["gin", "gcn", "mlp", "weighted", "deepset", "dwatt"],
        help="Filter by method for eval step (e.g., 'deepset', 'gin', 'gcn', 'dwatt'). Can use --method or --filter-method"
    )

    parser.add_argument(
        "--use-precomputed",
        action="store_true",
        help="Use precomputed embeddings for evaluation (faster, MTEB-style evaluation without LLM loading)"
    )

    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use V2 workflow for STS (enhanced pooling, learnable epsilon, STSBenchmark-only training)"
    )

    # Retrieval-specific arguments
    parser.add_argument(
        "--node-to-choose",
        dest="node_to_choose",
        choices=["mean", "sum"],
        default="mean",
        help="Pooling method for GNN output (default: mean)"
    )

    parser.add_argument(
        "--gin-mlp-layers",
        dest="gin_mlp_layers",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="GIN MLP layers: 0=GCN, 1-2=GIN (default: 1)"
    )

    parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=50000,
        help="Max MS MARCO triplets to precompute (default: 50000)"
    )

    parser.add_argument(
        "--transfer-from",
        dest="transfer_from",
        type=str,
        choices=["Pythia-410m", "Pythia-2.8b", "TinyLlama-1.1B", "Llama3-8B"],
        help="Transfer hyperparameters from another model (for sts-train step)"
    )

    parser.add_argument(
        "--linear",
        action="store_true",
        help="For sts-summarize: include only linear models (default: non-linear only)"
    )

    args = parser.parse_args()

    # Route to appropriate function
    if args.step == "postgres":
        run_postgres()

    elif args.step == "precompute":
        if not args.model:
            print("ERROR: --model is required for precompute step")
            sys.exit(1)
        run_precompute(args.model, submit=args.submit, tasks=args.tasks, partition=args.partition)

    elif args.step == "optuna":
        if not args.model:
            print("ERROR: --model is required for optuna step")
            sys.exit(1)
        run_optuna(args.model, args.methods, submit=args.submit, tasks=args.tasks, partition=args.partition, graph_type=args.graph_type)

    elif args.step == "train":
        run_train(args.model, submit=args.submit, filter_study=args.filter_study, filter_task=args.filter_task, graph_type=args.graph_type)

    elif args.step == "eval":
        run_eval(args.model, submit=args.submit, filter_task=args.filter_task, graph_type=args.graph_type, use_precomputed=args.use_precomputed, tasks=args.tasks, filter_method=args.filter_method)

    elif args.step == "layer-baseline":
        if not args.model:
            print("ERROR: --model is required for layer-baseline step")
            sys.exit(1)
        run_layer_baseline(args.model, layers=args.layers, tasks=args.tasks, submit=args.submit, all_sts_tasks=args.all_sts_tasks, partition=args.partition)

    elif args.step == "summarize":
        run_summarize(args.output, model=args.model, filter_model=args.filter_model, filter_task=args.filter_task, use_precomputed=args.use_precomputed)

    # STS steps
    elif args.step == "sts-precompute":
        if not args.model:
            print("ERROR: --model is required for sts-precompute step")
            sys.exit(1)
        run_sts_precompute(args.model, submit=args.submit, tasks=args.tasks, partition=args.partition)

    elif args.step == "sts-optuna":
        if not args.model:
            print("ERROR: --model is required for sts-optuna step")
            sys.exit(1)
        run_sts_optuna(args.model, args.methods, submit=args.submit, tasks=args.tasks, partition=args.partition, v2=args.v2)

    elif args.step == "sts-train":
        run_sts_train(args.model, submit=args.submit, filter_study=args.filter_study, filter_task=args.filter_task, transfer_from=args.transfer_from, v2=args.v2)

    elif args.step == "sts-eval":
        run_sts_eval(args.model, submit=args.submit, filter_study=args.filter_study, filter_task=args.filter_task, eval_tasks=args.eval_tasks, partition=args.partition, v2=args.v2)

    elif args.step == "sts-summarize":
        run_sts_summarize(args.output, model=args.model, filter_model=args.filter_model, filter_task=args.filter_task, linear=args.linear, v2=args.v2)

    # Retrieval steps
    elif args.step == "retrieval-precompute":
        if not args.model:
            print("ERROR: --model is required for retrieval-precompute step")
            sys.exit(1)
        run_retrieval_precompute(args.model, submit=args.submit, max_samples=args.max_samples, partition=args.partition)

    elif args.step == "retrieval-optuna":
        if not args.model:
            print("ERROR: --model is required for retrieval-optuna step")
            sys.exit(1)
        run_retrieval_optuna(args.model, args.methods, submit=args.submit, tasks=args.tasks, partition=args.partition)

    elif args.step == "retrieval-train":
        if not args.model:
            print("ERROR: --model is required for retrieval-train step")
            sys.exit(1)
        run_retrieval_train_fixed(args.model, submit=args.submit, partition=args.partition)

    elif args.step == "retrieval-eval":
        run_retrieval_eval(args.model, submit=args.submit, filter_task=args.filter_task)

    elif args.step == "retrieval-direct":
        if not args.model:
            print("ERROR: --model is required for retrieval-direct step")
            sys.exit(1)
        if not args.tasks or len(args.tasks) != 1:
            print("ERROR: --tasks with exactly one task is required for retrieval-direct step")
            print("Example: python3 pipeline.py retrieval-direct --model Pythia-410m --tasks MSMARCO")
            sys.exit(1)
        run_retrieval_direct(
            args.model,
            args.tasks[0],
            encoder=args.methods[0] if args.methods else "gin",
            node_to_choose=getattr(args, 'node_to_choose', 'mean'),
            gin_mlp_layers=getattr(args, 'gin_mlp_layers', 1),
            graph_type=args.graph_type or "cayley"
        )

    elif args.step == "retrieval-train-all":
        if not args.model:
            print("ERROR: --model is required for retrieval-train-all step")
            sys.exit(1)
        if not args.tasks or len(args.tasks) != 1:
            print("ERROR: --tasks with exactly one task is required for retrieval-train-all step")
            print("Example: python3 pipeline.py retrieval-train-all --model Pythia-410m --tasks MSMARCO")
            sys.exit(1)
        run_retrieval_train_all(args.model, args.tasks[0])

    elif args.step == "latex-table":
        run_latex_table(
            input_files=args.input,
            output=args.output,
            model_names=args.model_names,
            task_col=args.task_col,
            deltas=args.deltas,
            no_bold_best=args.no_bold_best,
            caption=args.caption,
            label=args.label
        )

    print(f"\n{'='*70}")
    print("Pipeline step completed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
