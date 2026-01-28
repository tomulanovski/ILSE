#!/usr/bin/env python3
"""
Generate SLURM evaluation job files based on best training configurations.
"""

import csv
import os
from pathlib import Path

# Job template for evaluation
EVAL_JOB_TEMPLATE = """#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-general-pool
#SBATCH --account=your-account
#SBATCH --gres=gpu:A100:1
#SBATCH --output=job_logs/eval/eval_{job_name}.out
#SBATCH --error=job_logs/eval/eval_{job_name}.err
#SBATCH --qos=public

CONDA_PATH=${GNN_REPO_DIR}/miniconda3

# Source conda
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate your specific environment
conda activate lbl 

# Verify the environment was activated
echo "Current conda environment: $CONDA_PREFIX"

# Print GPU information
nvidia-smi

# Change to the repo directory
cd "${GNN_REPO_DIR}/llm_gnn_proj/gnn_lbl"

# Run evaluation
python -m scripts_and_jobs.scripts.eval.mteb_evaluator \\
  --model_family {model_family} \\
  --model_size {model_size} \\
  --model_path "{model_path}" \\
  --batch_size 8 \\
  {tasks_arg}\\
  {custom_tasks_arg}\\
  --output_dir ./mteb_results_gnn_best/{job_name}
"""


def generate_eval_jobs(csv_path, output_dir="eval_jobs", tasks=None, custom_tasks=None):
    """
    Generate evaluation job files from the best configs CSV.
    
    Args:
        csv_path: Path to the val_best_configs.csv file
        output_dir: Directory to save generated job files
        tasks: List of standard MTEB tasks to filter by (None = don't filter)
        custom_tasks: List of custom tasks to filter by (None = don't filter)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("job_logs/eval", exist_ok=True)
    
    # Read CSV and generate jobs
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        job_count = 0
        skipped_count = 0
        
        for row in reader:
            task_name = row['task']
            
            # Determine if this task should be included
            should_include = False
            if tasks is None and custom_tasks is None:
                # No filtering - include all
                should_include = True
            elif tasks and task_name in tasks:
                should_include = True
            elif custom_tasks and task_name in custom_tasks:
                should_include = True
            
            if not should_include:
                skipped_count += 1
                continue
            
            job_name = row['job_name']
            model_family = row['model_family']
            model_size = row['model_size']
            encoder = row['encoder']
            save_dir = row['save_dir']
            
            # Construct model path - assuming .pt extension based on your example
            # The model path should be where the trained model was saved
            model_dir = save_dir
            for file_name in os.listdir(os.path.abspath(model_dir)):
                if file_name.endswith(".pt"):
                    model_path = f"{save_dir}/{file_name}"

            
            # Determine which argument to use for this task
            tasks_arg = ""
            custom_tasks_arg = ""
            
            if tasks and task_name in tasks:
                tasks_arg = f"--tasks {task_name} "
            elif custom_tasks and task_name in custom_tasks:
                custom_tasks_arg = f"--custom_tasks {task_name} "
            elif tasks is None and custom_tasks is None:
                # Default: assume it's a custom task if no filtering specified
                custom_tasks_arg = f"--custom_tasks {task_name} "
            
            # Generate job content
            job_content = EVAL_JOB_TEMPLATE.format(
                job_name=job_name,
                model_family=model_family,
                model_size=model_size,
                model_path=model_path,
                tasks_arg=tasks_arg,
                custom_tasks_arg=custom_tasks_arg
            )
            
            # Write job file
            job_filename = f"{output_dir}/eval_{job_name}.sbatch"
            with open(job_filename, 'w') as job_file:
                job_file.write(job_content)
            
            job_count += 1
            print(f"Generated: {job_filename}")
    
    print(f"\nTotal jobs generated: {job_count}")
    if skipped_count > 0:
        print(f"Skipped (filtered out): {skipped_count}")
    print(f"Jobs saved to: {output_dir}/")
    print(f"\nTo submit all jobs with 2-second delay:")
    print(f"  cd {output_dir} && for job in *.sbatch; do sbatch \"$job\"; sleep 2; done")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation SLURM jobs from best configs CSV")
    parser.add_argument("--csv", type=str, default="val_best_configs.csv",
                        help="Path to the best configs CSV file")
    parser.add_argument("--output_dir", type=str, default="eval_jobs",
                        help="Directory to save generated job files")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Filter by standard MTEB tasks (space-separated). Will use --tasks flag in job.")
    parser.add_argument("--custom_tasks", type=str, nargs="+", default=None,
                        help="Filter by custom tasks (space-separated). Will use --custom_tasks flag in job.")
    
    args = parser.parse_args()
    
    if args.tasks or args.custom_tasks:
        if args.tasks:
            print(f"Filtering by standard tasks: {', '.join(args.tasks)}")
        if args.custom_tasks:
            print(f"Filtering by custom tasks: {', '.join(args.custom_tasks)}")
        print()
    else:
        print("Generating jobs for all tasks (will use --custom_tasks by default)\n")
    
    generate_eval_jobs(args.csv, args.output_dir, args.tasks, args.custom_tasks)