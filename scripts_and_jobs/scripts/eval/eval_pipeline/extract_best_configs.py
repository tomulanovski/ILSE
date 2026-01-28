#!/usr/bin/env python3
"""
Extract best configurations from combined results and generate job configuration CSV.
For each (model, graph_type, task) combination, finds the best performing config.
"""

import pandas as pd
import numpy as np
import argparse


def extract_best_configs(input_csv, output_csv, model_family="Pythia", model_size="410m"):
    """
    Extract best configurations and create job configuration CSV.
    
    Args:
        input_csv: Path to combined_table.csv
        output_csv: Path to output jobs_config.csv
        model_family: Model family for jobs (default: Pythia)
        model_size: Model size for jobs (default: 410m)
    """
    # Load the data
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total rows: {len(df)}")
    print(f"Unique tasks: {df['task_name'].unique()}")
    print(f"Unique encoders: {df['encoder'].unique()}")
    
    # Fix numeric columns if needed
    for col in ['best_val_acc', 'train_time_sec']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace("'", "").replace('nan', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    best_configs = []
    
    # Process each task
    for task in df['task_name'].unique():
        task_data = df[df['task_name'] == task]
        
        print(f"\nProcessing task: {task}")
        
        # Handle MLP separately (no graph_type)
        mlp_data = task_data[task_data['encoder'] == 'mlp']
        if not mlp_data.empty:
            best_mlp = mlp_data.loc[mlp_data['best_val_acc'].idxmax()]
            config = extract_config_row(best_mlp, task, model_family, model_size)
            best_configs.append(config)
            print(f"  MLP best: {best_mlp['best_val_acc']:.4f}")
        
        # Handle GNN models (gin, gcn) with graph_type
        gnn_data = task_data[task_data['encoder'].isin(['gin', 'gcn'])]
        
        if not gnn_data.empty:
            # Get unique (encoder, graph_type) combinations
            combinations = gnn_data[['encoder', 'graph_type']].drop_duplicates()
            
            for _, combo in combinations.iterrows():
                encoder = combo['encoder']
                graph_type = combo['graph_type']
                
                # Get best config for this combination
                subset = gnn_data[
                    (gnn_data['encoder'] == encoder) & 
                    (gnn_data['graph_type'] == graph_type)
                ]
                
                if not subset.empty:
                    best_row = subset.loc[subset['best_val_acc'].idxmax()]
                    config = extract_config_row(best_row, task, model_family, model_size)
                    best_configs.append(config)
                    print(f"  {encoder.upper()}-{graph_type}: {best_row['best_val_acc']:.4f}")
    
    # Create DataFrame
    configs_df = pd.DataFrame(best_configs)
    
    # Sort by task and encoder for better organization
    configs_df = configs_df.sort_values(['task', 'encoder', 'graph_type']).reset_index(drop=True)
    
    # Save to CSV
    configs_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print(f"Extracted {len(configs_df)} best configurations")
    print(f"Saved to: {output_csv}")
    print("="*60)
    
    # Print summary
    print("\nSummary by encoder:")
    print(configs_df['encoder'].value_counts())
    
    return configs_df


def extract_config_row(row, task, model_family, model_size):
    """
    Extract configuration from a result row and format for job generation.
    
    Args:
        row: pandas Series with result data
        task: Task name
        model_family: Model family
        model_size: Model size
        
    Returns:
        Dict with job configuration
    """
    encoder = row['encoder']
    graph_type = row.get('graph_type', 'N/A')
    
    # Create job name
    task_short = task.replace('Classification', '').lower()
    if pd.isna(graph_type) or graph_type == 'N/A':
        job_name = f"{encoder}_{task_short}"
    else:
        job_name = f"{encoder}_{graph_type}_{task_short}"
    
    # Create save directory
    save_dir = f"./models/{job_name}"
    
    # SLURM configuration (defaults)
    config = {
        'job_name': job_name,
        'time': '04:00:00',
        'mem': '64G',
        'cpus': '8',
        'partition': 'gpu-general-pool',
        'account': 'your-account',
        'gpu_type': 'A100',
        'num_gpus': '1',
        'output_log': f'job_logs/{job_name}.out',
        'error_log': f'job_logs/{job_name}.err',
    }
    
    # Training configuration
    config.update({
        'task': task,
        'model_family': model_family,
        'model_size': model_size,
        'encoder': encoder,
        'save_dir': save_dir,
    })
    
    # GNN-specific parameters
    if encoder in ['gin', 'gcn']:
        config.update({'encoder': 'gin'})
    else:
        config.update({'encoder': 'mlp'})

    config.update({
        'gin_layers': int(row['num_msg_pass_layers']) if not pd.isna(row['num_msg_pass_layers']) else 1,
        'gin_mlp_layers': int(row['num_gins_mlp_layers']) if not pd.isna(row['num_gins_mlp_layers']) else 1,
        'graph_type': graph_type if not pd.isna(graph_type) else 'cayley',
        'node_to_choose': row['node_to_choose'] if not pd.isna(row['node_to_choose']) else 'mean',
        'gin_hidden_dim': int(row['gin_hidden_dim']) if not pd.isna(row['gin_hidden_dim']) else 256,
    })
    
        # MLP defaults
    config.update({
        'mlp_input': row['mlp_input'] if not pd.isna(row['mlp_input']) else 'last',
        'mlp_layers': int(row['mlp_layers']) if not pd.isna(row['mlp_layers']) else 2,
    })

    # Common training parameters
    config.update({
        'dropout': row['dropout'] if not pd.isna(row['dropout']) else 0.0,
        'epochs': 50,  # Default
        'batch_size': int(row['batch_size']) if not pd.isna(row['batch_size']) else 64,
        'lr': row['lr'] if not pd.isna(row['lr']) else 0.001,
        'weight_decay': row['weight_decay'] if not pd.isna(row['weight_decay']) else 1e-4,
    })
    
    # Add metadata for reference
    config['best_val_acc'] = row['best_val_acc'] if not pd.isna(row['best_val_acc']) else 0.0
    config['trial_number'] = int(row['trial_number']) if not pd.isna(row['trial_number']) else 0
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Extract best configurations and generate job CSV'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (combined_table.csv)')
    parser.add_argument('--output', type=str, default='jobs_config_best.csv',
                       help='Output CSV file for job configurations')
    parser.add_argument('--model_family', type=str, default='Pythia',
                       help='Model family for jobs')
    parser.add_argument('--model_size', type=str, default='410m',
                       help='Model size for jobs')
    
    args = parser.parse_args()
    
    extract_best_configs(
        input_csv=args.input,
        output_csv=args.output,
        model_family=args.model_family,
        model_size=args.model_size
    )
    
    print("\nNext steps:")
    print(f"1. Review the generated config: {args.output}")
    print(f"2. Generate job scripts: python generate_jobs.py --config {args.output}")
    print(f"3. Submit jobs: python generate_jobs.py --config {args.output} --submit")


if __name__ == "__main__":
    main()