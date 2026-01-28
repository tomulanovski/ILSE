#!/usr/bin/env python3
"""
Generate SLURM job scripts from a CSV configuration file.
Each row in the CSV becomes a separate job script.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, Any


# Job template
JOB_TEMPLATE = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
#SBATCH --qos=public

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

# Change to the repo directory
cd "${GNN_REPO_DIR}/llm_gnn_proj/gnn_lbl"

echo "Job started at: $(date)"

# Run training
time python3 -m experiments.utils.model_definitions.gnn.basic_gin_trainer \\
  --task {task} \\
  --model_family {model_family} \\
  --model_size {model_size} \\
  --encoder {encoder} \\
  --gin_layers {gin_layers} \\
  --gin_mlp_layers {gin_mlp_layers} \\
  --graph_type {graph_type} \\
  --node_to_choose {node_to_choose} \\
  --gin_hidden_dim {gin_hidden_dim} \\
  --mlp_input {mlp_input} \\
  --mlp_layers {mlp_layers} \\
  --dropout {dropout} \\
  --epochs {epochs} \\
  --batch_size {batch_size} \\
  --lr {lr} \\
  --weight_decay {weight_decay} \\
  --save_dir {save_dir}

echo "Job finished at: $(date)"
"""


def convert_float_to_int(value: str) -> str:
    """Convert float strings like '1.0' to int strings like '1'."""
    if not value or value == '':
        return value
    try:
        float_val = float(value)
        if float_val.is_integer():
            return str(int(float_val))
        return value
    except (ValueError, AttributeError):
        return value


def generate_job_script(config: Dict[str, Any], output_dir: Path) -> str:
    """
    Generate a SLURM job script from configuration.
    
    Args:
        config: Dictionary with job configuration
        output_dir: Directory to save job scripts
        
    Returns:
        Path to generated job script
    """
    # Fields that should be integers (not floats)
    int_fields = ['gin_layers', 'gin_mlp_layers', 'gin_hidden_dim', 'epochs', 
                  'batch_size', 'mlp_layers', 'num_gpus']
    
    # Convert float values to int for specific fields
    cleaned_config = config.copy()
    for field in int_fields:
        if field in cleaned_config:
            cleaned_config[field] = convert_float_to_int(cleaned_config[field])
    
    # Fill in the template
    job_content = JOB_TEMPLATE.format(**cleaned_config)
    
    # Create output filename
    job_name = config['job_name']
    job_file = output_dir / f"{job_name}.sbatch"
    
    # Write job script
    with open(job_file, 'w') as f:
        f.write(job_content)
    
    # Make executable
    job_file.chmod(0o755)
    
    return str(job_file)


def load_config_from_csv(csv_path: str) -> list:
    """
    Load job configurations from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip empty rows or comments
            if not row or row.get('job_name', '').startswith('#'):
                continue
            configs.append(row)
    
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM job scripts from CSV configuration'
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to CSV configuration file')
    parser.add_argument('--output_dir', type=str, default='./generated_jobs',
                       help='Directory to save generated job scripts')    
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    print(f"Loading configurations from: {args.config}")
    configs = load_config_from_csv(args.config)
    print(f"Found {len(configs)} job configurations")
    
    # Generate job scripts
    generated_jobs = []
    for i, config in enumerate(configs, 1):
        job_name = config['job_name']
        print(f"\n[{i}/{len(configs)}] Generating job: {job_name}")
        
        try:
            job_file = generate_job_script(config, output_dir)
            generated_jobs.append(job_file)
            print(f"  ✓ Created: {job_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print(f"Successfully generated {len(generated_jobs)} job scripts")
    print(f"Location: {output_dir}")
    print("="*60)
        
    print("\nTo submit jobs, run:")
    print(f"  cd {output_dir}")
    print(f"  for job in *.sbatch; do sbatch $job; done")
    print("\nOr submit with 2-second delays:")
    print(f"  cd {output_dir}")
    print(f"  for job in *.sbatch; do sbatch \"$job\"; sleep 2; done")


if __name__ == "__main__":
    main()