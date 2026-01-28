#!/usr/bin/env python3
"""
Summarize STS results: layer baselines + trained models.
Creates CSV with: Task, last_layer, best_layer, best_layer_idx, gin_*, mlp_*, weighted_*
Extracts main_score from each JSON file.

Usage:
    python3 summarize_sts_results.py \\
        --model Pythia-410m \\
        --base_dir results \\
        --output sts_summary_pythia410m.csv
"""
import os
import sys
import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add project root to Python path
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications
from transformers import AutoConfig

# Model configurations: (family, size)
MODEL_MAP = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B"),
    "Pythia-410m": ("Pythia", "410m"),
    "Pythia-2.8b": ("Pythia", "2.8b"),
    "Llama3-8B": ("Llama3", "8B"),
}

# All STS tasks
ALL_STS_TASKS = [
    "STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16",
    "STS17", "STS22", "BIOSSES", "SICK-R"
]


def get_num_layers(model_name):
    """Dynamically get the number of layers (including embedding) from model config."""
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_family, model_size = MODEL_MAP[model_name]
    
    # Create model specs to get the model path
    model_specs = TextModelSpecifications(model_family, model_size, revision="main")
    model_path = model_specs.model_path_func(model_family, model_size)
    
    # Load config and get num_layers = num_hidden_layers + 1 (embedding layer)
    config = AutoConfig.from_pretrained(model_path, revision="main")
    num_layers = config.num_hidden_layers + 1  # +1 for embedding layer
    
    return num_layers


def extract_main_score(result_file: Path) -> Optional[float]:
    """Extract main_score from STS result JSON."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            # MTEB format for STS tasks
            if 'scores' in data and 'test' in data['scores']:
                test_scores = data['scores']['test']
                if test_scores and len(test_scores) > 0:
                    main_score = test_scores[0].get('main_score', None)
                    if main_score is not None:
                        return float(main_score)
    except Exception as e:
        return None
    return None


def get_layer_baseline_scores(model_name: str, base_dir: str, tasks: list) -> Dict[str, Dict[str, float]]:
    """
    Get layer baseline scores for each task (last layer and best layer only).
    
    Returns:
        dict: {task_name: {'last_layer': score, 'best_layer': score, 'best_layer_idx': idx}}
    """
    if model_name not in MODEL_MAP:
        return {}
    
    model_family, model_size = MODEL_MAP[model_name]
    num_layers = get_num_layers(model_name)
    
    # Layer baseline results path
    results_base = Path(base_dir) / "layer_baselines" / model_family / model_size / "main" / "mteb"
    
    if not results_base.exists():
        print(f"  Note: Layer baseline results not found at {results_base}")
        return {}
    
    layer_scores = {}
    last_layer_idx = num_layers - 1
    
    for task in tasks:
        task_scores = {}
        
        # Get last layer score
        last_layer_file = results_base / f"layer_{last_layer_idx}" / f"{task}.json"
        if last_layer_file.exists():
            last_score = extract_main_score(last_layer_file)
            if last_score is not None:
                task_scores['last_layer'] = last_score
        
        # Find best layer
        best_layer_idx = None
        best_score = None
        for layer_idx in range(num_layers):
            layer_file = results_base / f"layer_{layer_idx}" / f"{task}.json"
            if layer_file.exists():
                score = extract_main_score(layer_file)
                if score is not None:
                    if best_score is None or score > best_score:
                        best_score = score
                        best_layer_idx = layer_idx
        
        if best_layer_idx is not None and best_score is not None:
            task_scores['best_layer'] = best_score
            task_scores['best_layer_idx'] = best_layer_idx
        
        if task_scores:
            layer_scores[task] = task_scores
    
    return layer_scores


def get_trained_model_scores(model_name: str, base_dir: str, tasks: list, filter_encoder: str = None, v2: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Get trained model scores (gin, mlp, weighted, linear_gin, etc.) for each task.
    Handles nested directory structure: {encoder}/{config}/no_model_name_available/no_revision_available/{task}.json

    Args:
        filter_encoder: Optional filter string. If provided, only include encoders that start with this string.
                       E.g., "linear" includes all linear_ encoders, "linear_gcn" includes only linear_gcn.
        v2: If True, only include cayley models (configs ending with _cayley). If False, exclude cayley models.

    Returns:
        dict: {task_name: {'method_config': main_score}}
    """
    if model_name not in MODEL_MAP:
        return {}
    
    model_family, model_size = MODEL_MAP[model_name]
    
    # STS results path
    results_base = Path(base_dir) / "sts_results" / model_family / model_size / "main" / "mteb"
    
    if not results_base.exists():
        print(f"  Note: STS model results not found at {results_base}")
        return {}
    
    model_scores = {}

    # Automatically detect all encoder directories (including linear_ variants)
    # This way we pick up: gin, gcn, mlp, weighted, deepset, linear_gin, linear_gcn, etc.
    if not results_base.exists():
        return {}

    for encoder_dir in results_base.iterdir():
        if not encoder_dir.is_dir():
            continue

        encoder = encoder_dir.name  # e.g., "gin", "gcn", "linear_gcn", etc.

        # Apply encoder filter if specified
        # Support exclusion with "!" prefix (e.g., "!linear" excludes all linear_ encoders)
        if filter_encoder:
            if filter_encoder.startswith("!"):
                # Exclusion mode: skip encoders that start with the pattern after "!"
                exclude_pattern = filter_encoder[1:]  # Remove "!" prefix
                if encoder.startswith(exclude_pattern):
                    continue
            else:
                # Inclusion mode: only include encoders that start with the pattern
                if not encoder.startswith(filter_encoder):
                    continue

        # Check each config directory
        for config_dir in encoder_dir.iterdir():
            if not config_dir.is_dir():
                continue

            config = config_dir.name

            # Filter by cayley suffix - ONLY for GIN/GCN encoders (which have cayley variants)
            # Other encoders (MLP, Weighted, DeepSet) don't have cayley versions, so include them always
            is_v2_config = config.endswith("_cayley")
            has_v2_variants = encoder in ["gin", "gcn", "linear_gin", "linear_gcn"]

            if has_v2_variants:
                # For encoders with cayley variants: filter by v2 flag
                if v2 and not is_v2_config:
                    # cayley mode: skip non-cayley configs for GIN/GCN
                    continue
                elif not v2 and is_v2_config:
                    # baseline mode: skip cayley configs for GIN/GCN
                    continue
            # For other encoders (MLP, Weighted, DeepSet): include all configs regardless of v2 flag
            
            # Search for task JSON files - handle nested structure
            # Path can be: {config}/{task}.json or {config}/no_model_name_available/no_revision_available/{task}.json
            for task in tasks:
                # Try direct path first
                task_file = config_dir / f"{task}.json"
                
                # If not found, try nested path
                if not task_file.exists():
                    nested_path = config_dir / "no_model_name_available" / "no_revision_available" / f"{task}.json"
                    if nested_path.exists():
                        task_file = nested_path
                    else:
                        # Try recursive search
                        for json_file in config_dir.rglob(f"{task}.json"):
                            task_file = json_file
                            break
                        else:
                            continue
                
                if task_file.exists():
                    main_score = extract_main_score(task_file)
                    if main_score is not None:
                        method_key = f"{encoder}_{config}"
                        if task not in model_scores:
                            model_scores[task] = {}
                        model_scores[task][method_key] = main_score
    
    return model_scores


def create_summary_dataframe(layer_scores: Dict, model_scores: Dict, tasks: list) -> pd.DataFrame:
    """Create a summary DataFrame with all results (last layer, best layer, and trained models)."""
    rows = []
    
    for task in tasks:
        row = {'Task': task}
        
        # Add layer baseline scores (last layer and best layer only)
        if task in layer_scores:
            task_layer_scores = layer_scores[task]
            row['last_layer'] = task_layer_scores.get('last_layer', None)
            row['best_layer'] = task_layer_scores.get('best_layer', None)
            row['best_layer_idx'] = task_layer_scores.get('best_layer_idx', None)
        else:
            row['last_layer'] = None
            row['best_layer'] = None
            row['best_layer_idx'] = None
        
        # Add trained model scores
        if task in model_scores:
            task_model_scores = model_scores[task]
            for method_key, score in task_model_scores.items():
                row[method_key] = score
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns: Task, last_layer, best_layer, best_layer_idx, then methods
    method_cols = [col for col in df.columns if col not in ['Task', 'last_layer', 'best_layer', 'best_layer_idx']]
    
    column_order = ['Task', 'last_layer', 'best_layer', 'best_layer_idx'] + sorted(method_cols)
    df = df[[col for col in column_order if col in df.columns]]
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Summarize STS results: layer baselines + trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model name"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results",
        help="Base directory containing results (default: results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./summary",
        help="Output directory for summary CSV (default: ./summary)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: sts_summary_{model}.csv)"
    )
    parser.add_argument(
        "--filter-task",
        type=str,
        default=None,
        help="Filter to specific task (optional)"
    )
    parser.add_argument(
        "--filter-encoder",
        type=str,
        default=None,
        help="Filter encoders by prefix. Use '!prefix' to exclude (e.g., '!linear' excludes all linear models). Examples: 'linear' (only linear), 'linear_gcn' (specific), '!linear' (non-linear only). Optional."
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="cayley mode: Include GIN/GCN models with _cayley suffix + MLP/Weighted/DeepSet. Default (baseline): GIN/GCN without _cayley + MLP/Weighted/DeepSet."
    )

    args = parser.parse_args()

    # Determine tasks
    tasks = ALL_STS_TASKS
    if args.filter_task:
        if args.filter_task in ALL_STS_TASKS:
            tasks = [args.filter_task]
        else:
            print(f"Warning: Task '{args.filter_task}' not in STS tasks. Using all tasks.")
    
    print(f"\n{'='*70}")
    print(f"Summarizing STS Results")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Base directory: {args.base_dir}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"{'='*70}\n")
    
    # Get layer baseline scores
    print("Loading layer baseline results...")
    layer_scores = get_layer_baseline_scores(args.model, args.base_dir, tasks)
    if layer_scores:
        print(f"  ✓ Found layer baseline results for {len(layer_scores)} task(s)")
    else:
        print(f"  ⚠ No layer baseline results found")
    
    # Get trained model scores
    print("\nLoading trained model results...")
    if args.filter_encoder:
        print(f"  Filtering encoders starting with: '{args.filter_encoder}'")
    if args.v2:
        print(f"  cayley mode: GIN/GCN with _cayley suffix + MLP/Weighted/DeepSet (no cayley variants)")
    else:
        print(f"  baseline mode: GIN/GCN without _cayley suffix + MLP/Weighted/DeepSet (no cayley variants)")
    model_scores = get_trained_model_scores(args.model, args.base_dir, tasks, filter_encoder=args.filter_encoder, v2=args.v2)
    if model_scores:
        total_methods = sum(len(scores) for scores in model_scores.values())
        print(f"  ✓ Found trained model results: {total_methods} method-task combinations")
    else:
        print(f"  ⚠ No trained model results found")
    
    # Create summary DataFrame
    print("\nCreating summary table...")
    df = create_summary_dataframe(layer_scores, model_scores, tasks)
    
    # Determine output path
    if args.output:
        output_filename = args.output
    else:
        model_short = args.model.replace("-", "").lower()
        output_filename = f"sts_summary_{model_short}.csv"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Summary saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Display summary
    print("Summary Table (first few rows):")
    print(df.head().to_string())
    print(f"\nFull table shape: {df.shape}")
    print(f"  Rows (tasks): {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Tasks with layer baselines: {sum(1 for task in tasks if task in layer_scores)}")
    print(f"  Tasks with trained models: {sum(1 for task in tasks if task in model_scores)}")
    
    if layer_scores:
        print(f"\nBest layer per task:")
        for task in tasks:
            if task in layer_scores:
                task_scores = layer_scores[task]
                if 'best_layer' in task_scores and 'best_layer_idx' in task_scores:
                    best_idx = task_scores['best_layer_idx']
                    best_score = task_scores['best_layer']
                    print(f"  {task}: layer_{best_idx} = {best_score:.4f}")
    
    print()


if __name__ == "__main__":
    main()

