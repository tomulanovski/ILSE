#!/usr/bin/env python3
"""
Simple script to check MTEB task split sizes for English tasks of a specific type.
"""

import argparse
import mteb
from datasets import load_dataset_builder


def get_split_sizes(hf_path, hf_config=None):
    """Get split sizes from HuggingFace without downloading data."""
    try:
        if hf_config:
            builder = load_dataset_builder(hf_path, hf_config)
        else:
            builder = load_dataset_builder(hf_path)
        
        splits = {}
        if hasattr(builder, 'info') and hasattr(builder.info, 'splits'):
            for split_name, split_info in builder.info.splits.items():
                splits[split_name] = split_info.num_examples
        
        return splits
    except Exception as e:
        return {'error': str(e)[:80]}


def main(task_type):
    if not task_type:
        print("Error: Please specify a task type with --type")
        return
    
    print(f"Loading tasks of type: {task_type}\n")
    
    tasks = mteb.get_tasks()
    
    # Filter tasks by type and language
    filtered_tasks = []
    for task in tasks:
        # Check type
        t_type = getattr(task.metadata, 'type', '')
        if t_type.lower() != task_type.lower():
            continue
        
        # Check if 'eng' in languages
        languages = getattr(task.metadata, 'languages', [])
        if 'eng' not in languages:
            continue
        
        filtered_tasks.append(task)
    
    if not filtered_tasks:
        print(f"No tasks found for type '{task_type}' with 'eng' language")
        return
    
    print(f"Found {len(filtered_tasks)} tasks\n")
    print("=" * 90)
    print(f"{'TASK NAME':<45} {'TRAIN':<15} {'VAL':<15} {'TEST':<15}")
    print("=" * 90)
    
    for task in sorted(filtered_tasks, key=lambda x: x.metadata.name):
        # Get HuggingFace dataset path
        dataset_info = getattr(task.metadata, 'dataset', None)
        
        if not dataset_info:
            print(f"{task.metadata.name:<45} No HF dataset info")
            continue
        
        # Extract path and config
        if isinstance(dataset_info, dict):
            hf_path = dataset_info.get('path')
            hf_config = dataset_info.get('name')
        elif isinstance(dataset_info, str):
            hf_path = dataset_info
            hf_config = None
        else:
            print(f"{task.metadata.name:<45} Invalid dataset info")
            continue
        
        # Get split sizes
        splits = get_split_sizes(hf_path, hf_config)
        
        if 'error' in splits:
            print(f"{task.metadata.name:<45} Error: {splits['error']}")
        else:
            train = f"{splits.get('train', 0):,}" if splits.get('train', 0) > 0 else "-"
            val_count = splits.get('validation', splits.get('dev', 0))
            val = f"{val_count:,}" if val_count > 0 else "-"
            test = f"{splits.get('test', 0):,}" if splits.get('test', 0) > 0 else "-"
            
            print(f"{task.metadata.name:<45} {train:<15} {val:<15} {test:<15}")
    
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Check split sizes for English MTEB tasks of a specific type.'
    )
    
    parser.add_argument(
        '--type', '-t',
        type=str,
        required=True,
        help='Task type to filter (e.g., Classification, Retrieval, Clustering)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args.type)
    except Exception as e:
        print(f"Error: {e}")