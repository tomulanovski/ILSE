#!/usr/bin/env python3
"""
Extract best Linear GCN configuration for each graph topology from Optuna studies.
"""
import optuna
import json
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path to import optuna_storage
sys.path.insert(0, str(Path(__file__).parent))
from optuna_storage import get_optuna_storage_url


def extract_best_per_topology(study_name, storage_url, variant):
    """
    Extract best trial for each graph_type from an Optuna study.
    All trials in this study are the same variant (determined by study name).
    
    Args:
        study_name: Optuna study name
        storage_url: Optuna storage URL
        variant: Variant number (1 or 2) - determined by study name
    
    Returns:
        dict mapping (graph_type, variant) -> best trial info
        Format: {
            'virtual_node_v1': {...},  # or _cayley depending on variant param
            'cayley_v1': {...},
            'linear_v1': {...}
        }
    """
    print(f"\nLoading study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    # Get all completed trials
    completed_trials = [t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"Total completed trials: {len(completed_trials)}")
    print(f"All trials in this study are Variant {variant} (based on study name)")
    
    # Group by graph_type and find best for each
    best_configs = {}
    
    for graph_type in ['virtual_node', 'cayley', 'linear', 'cayley']:
        trials_for_topology = [t for t in completed_trials 
                              if t.params.get('graph_type') == graph_type]
        
        if not trials_for_topology:
            print(f"  ⚠️  No trials found for {graph_type}")
            continue
        
        best_trial = max(trials_for_topology, key=lambda t: t.value)
        
        # Key format: "virtual_node_v1", "cayley_v2", etc.
        config_key = f"{graph_type}_v{variant}"
        
        best_configs[config_key] = {
            'trial_number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params,
            'graph_type': graph_type,
            'variant': variant,  # Set based on study name
            'n_trials_tested': len(trials_for_topology)
        }
        
        print(f"\n  ✓ {graph_type}:")
        print(f"    Best accuracy: {best_trial.value:.4f}")
        print(f"    Trial number: {best_trial.number}")
        print(f"    Trials tested: {len(trials_for_topology)}")
        print(f"    Params: {best_trial.params}")
    
    return best_configs


def main():
    parser = argparse.ArgumentParser(description="Extract best Linear GCN configs per topology")
    parser.add_argument("--storage_url", type=str, 
                       default=None,
                       help="Optuna storage URL (auto-detected if not provided)")
    parser.add_argument("--output_dir", type=str, default="./linear_gcn_best_configs",
                       help="Directory to save best configs")
    parser.add_argument("--tasks", nargs="+", 
                       default=["EmotionClassification", "Banking77Classification", 
                               "MTOPDomainClassification", "MTOPIntentClassification",
                               "PoemSentimentClassification"],
                       help="Tasks to process")
    parser.add_argument("--model", type=str, default="Pythia_410m",
                       help="Model family_size (e.g., Pythia_410m)")
    
    args = parser.parse_args()
    
    # Auto-detect storage URL if not provided (handles SSH tunnel for CS cluster)
    if args.storage_url is None:
        print("Auto-detecting Optuna database...")
        args.storage_url = get_optuna_storage_url()
        print(f"Using storage URL: {args.storage_url}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Extracting Best Linear GCN Configurations per (Topology, Variant)")
    print("="*80)
    
    all_results = {}
    
    for task in args.tasks:
        base_study_name = f"linear_gcn_{task}_{args.model}"
        print(f"\n{'='*80}")
        print(f"Task: {task}")
        print(f"{'='*80}")
        
        all_best_configs = {}
        
        # Check old study (no suffix) - all trials are variant 2 (no variant field in params)
        old_study_name = base_study_name
        print(f"\nChecking baseline study (Variant 2 only): {old_study_name}")
        try:
            # Extract by topology only, assign variant=2 based on study name (no suffix = cayley)
            old_configs = extract_best_per_topology(old_study_name, args.storage_url, variant=2)
            all_best_configs.update(old_configs)
            print(f"  ✓ Found {len(old_configs)} Variant 2 configs")
        except Exception as e:
            print(f"  ⚠️  Baseline study not found or error: {e}")

        # Check new study (with _v1 suffix) - all trials are variant 1
        new_study_name = f"{base_study_name}_v1"
        print(f"\nChecking cayley study (Variant 1 only): {new_study_name}")
        try:
            # Extract by topology only, assign variant=1 based on study name (_v1 suffix = cayley)
            new_configs = extract_best_per_topology(new_study_name, args.storage_url, variant=1)
            all_best_configs.update(new_configs)
            print(f"  ✓ Found {len(new_configs)} Variant 1 configs")
        except Exception as e:
            print(f"  ⚠️  New study not found or error: {e}")
        
        if not all_best_configs:
            print(f"  ✗ No configs found for {task}")
            all_results[task] = {"error": "No studies found"}
        else:
            all_results[task] = all_best_configs
            
            # Save individual task results
            task_output = output_dir / f"{task}_best_configs.json"
            with open(task_output, 'w') as f:
                json.dump(all_best_configs, f, indent=2)
            print(f"\n  ✓ Saved {len(all_best_configs)} configs to: {task_output}")
    
    # Save combined results
    combined_output = output_dir / "all_tasks_best_configs.json"
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {combined_output}")
    print(f"{'='*80}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Task':<35} {'Config':<20} {'Accuracy':<12} {'Trials':<8}")
    print("-"*80)
    
    for task, configs in all_results.items():
        if "error" in configs:
            print(f"{task:<35} ERROR: {configs['error']}")
            continue
        
        for config_key, config in sorted(configs.items()):
            acc = config['value']
            n_trials = config['n_trials_tested']
            print(f"{task:<35} {config_key:<20} {acc:<12.4f} {n_trials:<8}")
    
    print("="*80)


if __name__ == "__main__":
    main()

