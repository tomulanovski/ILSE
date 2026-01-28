#!/usr/bin/env python3
"""
Generate SLURM jobs for evaluating all layers of models.

Usage:
    python scripts_and_jobs/scripts/generate_layer_jobs.py

This creates:
- SLURM job files in: scripts_and_jobs/slurm_jobs/generated_jobs/
- A submit script to launch all jobs
- Results will be saved to: results/layer_baselines/
"""

import os
from pathlib import Path

# Configuration
JOBS_OUTPUT_DIR = "scripts_and_jobs/slurm_jobs/generated_jobs/best_single_layer"
LOGS_DIR = "job_logs/layer_sweep"
RESULTS_PATH = "results/layer_baselines"

# Models to evaluate: (family, size, num_layers)
MODELS = [
    ("TinyLlama", "1.1B", 23),
    ("Pythia", "2.8b", 32),
    ("Llama3", "8B", 32),
]

# SLURM configuration (A6000 with higher priority for TinyLlama)
SLURM_CONFIG = {
    "partition": "your-lab-partition",
    "qos": "owner",
    "time": "04:00:00",  # 4 hours per layer
    "memory": "32G",
    "cpus": 4,
    "gres": "gpu:1",
}

# Paths
# Load from environment variables (set in .env file)
CONDA_PATH = os.getenv("GNN_CONDA_PATH", "")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")
REPO_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())

# Validate required settings
if not CONDA_PATH:
    raise ValueError("GNN_CONDA_PATH not set in .env file. Copy .env.example to .env and configure it.")

# SLURM job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={memory}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres={gres}
#SBATCH --output={logs_dir}/{job_name}.out
#SBATCH --error={logs_dir}/{job_name}.err

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
cd "{repo_dir}"

echo "=========================================="
echo "Job started at: $(date)"
echo "Model: {model_family}-{model_size}"
echo "Layer: {layer}"
echo "=========================================="

# Run MTEB evaluation for this layer
python3 -u MTEB-Harness.py \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --revision main \\
    --evaluation_layer {layer} \\
    --base_results_path {results_path} \\
    --purpose run_tasks \\
    --filter_tasks classification_subset

echo "=========================================="
echo "Job finished at: $(date)"
echo "Results saved to: {results_path}/{model_family}/{model_size}/main/mteb/layer_{layer}/"
echo "=========================================="
"""


def create_job_file(model_family, model_size, layer, jobs_dir, logs_dir):
    """Create a single SLURM job file."""
    job_name = f"{model_family}_{model_size}_layer_{layer}"
    job_file = jobs_dir / f"{job_name}.sbatch"

    job_content = JOB_TEMPLATE.format(
        job_name=job_name,
        time=SLURM_CONFIG["time"],
        memory=SLURM_CONFIG["memory"],
        cpus=SLURM_CONFIG["cpus"],
        partition=SLURM_CONFIG["partition"],
        qos=SLURM_CONFIG["qos"],
        gres=SLURM_CONFIG["gres"],
        logs_dir=logs_dir,
        conda_path=CONDA_PATH,
        conda_env=CONDA_ENV,
        repo_dir=REPO_DIR,
        model_family=model_family,
        model_size=model_size,
        layer=layer,
        results_path=RESULTS_PATH,
    )

    with open(job_file, "w") as f:
        f.write(job_content)

    return job_name


def create_submit_script(all_jobs, jobs_dir):
    """Create a script to submit all jobs."""
    submit_file = jobs_dir / "submit_all_layers.sh"

    lines = [
        "#!/bin/bash\n",
        "#\n",
        "# Submit all layer sweep jobs\n",
        f"# Total jobs: {len(all_jobs)}\n",
        "#\n\n",
        "echo 'Submitting all layer sweep jobs...'\n",
        "echo ''\n\n",
    ]

    current_model = None
    for job_name, model_family, model_size in all_jobs:
        model_id = f"{model_family}-{model_size}"
        if current_model != model_id:
            current_model = model_id
            lines.append(f"\n# {model_id}\n")
            lines.append(f"echo 'Submitting {model_id} jobs...'\n")

        lines.append(f"sbatch {jobs_dir}/{job_name}.sbatch\n")
        lines.append("sleep 2\n")

    lines.append("\necho ''\n")
    lines.append("echo 'All jobs submitted!'\n")
    lines.append("echo 'Monitor with: squeue -u $USER'\n")

    with open(submit_file, "w") as f:
        f.writelines(lines)

    os.chmod(submit_file, 0o755)
    return submit_file


def main():
    # Create directories
    jobs_dir = Path(JOBS_OUTPUT_DIR)
    logs_dir = Path(LOGS_DIR)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Layer Sweep SLURM Jobs")
    print("=" * 70)
    print(f"Output directory: {jobs_dir}")
    print(f"Logs directory:   {logs_dir}")
    print(f"Results path:     {RESULTS_PATH}")
    print("")

    all_jobs = []

    # Generate jobs for each model
    for model_family, model_size, num_layers in MODELS:
        print(f"Generating {model_family}-{model_size} jobs ({num_layers} layers)...")

        for layer in range(num_layers):
            job_name = create_job_file(
                model_family, model_size, layer, jobs_dir, logs_dir
            )
            all_jobs.append((job_name, model_family, model_size))

        print(f"  ✓ Created {num_layers} jobs")

    # Create submit-all script
    print("\nCreating submit script...")
    submit_script = create_submit_script(all_jobs, jobs_dir)
    print(f"  ✓ Created {submit_script}")

    # Summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total jobs created: {len(all_jobs)}")
    for model_family, model_size, num_layers in MODELS:
        print(f"  - {model_family}-{model_size}: {num_layers} layers")

    print(f"\nJobs saved to: {jobs_dir}")
    print(f"Logs will be saved to: {logs_dir}")
    print(f"Results will be saved to: {RESULTS_PATH}")

    print("\n" + "-" * 70)
    print("To submit all jobs:")
    print(f"  bash {submit_script}")

    print("\nTo submit individual model:")
    for model_family, model_size, num_layers in MODELS:
        print(f"  # {model_family}-{model_size}")
        print(
            f"  for layer in {{0..{num_layers-1}}}; do sbatch {jobs_dir}/{model_family}_{model_size}_layer_${{layer}}.sbatch; done"
        )

    print("\nTo monitor progress:")
    print("  squeue -u $USER")
    print(f"  tail -f {logs_dir}/*.out")

    print("=" * 70)


if __name__ == "__main__":
    main()
