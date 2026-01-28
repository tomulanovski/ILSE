#!/usr/bin/env python3
"""
Enhanced summarizer that includes layer baseline results.
Creates CSV with: Task, last_layer, best_single_layer, gin_*, mlp_*, weighted_*

Usage:
    python3 summarize_with_layers.py \\
        --model TinyLlama-1.1B \\
        --base_dir mteb_results_best_models \\
        --output tinyllama_complete.csv
"""
import os
import json
import pandas as pd
import argparse
from pathlib import Path

# Model configurations
MODEL_MAP = {
    "TinyLlama-1.1B": ("TinyLlama", "1.1B", 23),
    "Pythia-410m": ("Pythia", "410m", 24),
    "Pythia-2.8b": ("Pythia", "2.8b", 32),
    "Llama3-8B": ("Llama3", "8B", 32),
}


def extract_model_name(filepath, base_dir):
    """Extract model name from path structure (same as original summarize_to_csv.py)."""
    rel_path = os.path.relpath(filepath, base_dir)
    parts = rel_path.split(os.sep)

    if len(parts) < 1:
        return None

    method = parts[0]

    # For simple methods, include method + model name
    if method in ["mlp", "weighted", "best_single_layer"]:
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        else:
            return method

    # For GNN, extract full configuration
    if len(parts) >= 3:
        model_name = "_".join(parts[:3])
        return model_name

    return None


def extract_accuracy_from_layer_baseline(result_file):
    """Extract test accuracy from layer baseline MTEB result JSON."""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            # MTEB format
            if 'test' in data and 'accuracy' in data['test']:
                return data['test']['accuracy'] * 100
            elif 'scores' in data and 'test' in data['scores']:
                return data['scores']['test'][0]['main_score'] * 100
            elif 'accuracy' in data:
                return data['accuracy'] * 100
            else:
                return None
    except Exception as e:
        return None


def get_layer_baseline_scores(model_name, tasks):
    """
    Get last_layer and best_single_layer scores for each task.

    Returns:
        dict: {task_name: {'last_layer': score, 'best_single_layer': score}}
    """
    if model_name not in MODEL_MAP:
        return {}

    model_family, model_size, num_layers = MODEL_MAP[model_name]
    # MTEB-Harness.py creates: base/family/size/revision/mteb/layer_N/
    results_base = Path(f"results/layer_baselines/{model_family}/{model_size}/main/mteb")

    if not results_base.exists():
        print(f"  Note: Layer baseline results not found at {results_base}")
        print(f"  Run: python3 pipeline.py layer-baseline --model {model_name} --submit")
        return {}

    layer_scores = {}

    for task in tasks:
        task_scores = {}
        last_layer_idx = num_layers - 1

        # Get last layer score
        last_layer_file = results_base / f"layer_{last_layer_idx}" / f"{task}.json"
        if last_layer_file.exists():
            last_score = extract_accuracy_from_layer_baseline(last_layer_file)
            if last_score is not None:
                task_scores['last_layer'] = last_score

        # Find best layer
        best_score = None
        for layer_idx in range(num_layers):
            layer_file = results_base / f"layer_{layer_idx}" / f"{task}.json"
            if layer_file.exists():
                score = extract_accuracy_from_layer_baseline(layer_file)
                if score is not None:
                    if best_score is None or score > best_score:
                        best_score = score

        if best_score is not None:
            task_scores['best_single_layer'] = best_score

        if task_scores:
            layer_scores[task] = task_scores

    return layer_scores


def main(model_name, base_dir, output_dir, output_filename, filter_model=None, filter_task=None):
    """
    Create summary CSV with layer baselines and trained models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store results: {task_name: {model_name: score}}
    results = {}

    # Walk through model results directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json") and file != "model_meta.json":
                filepath = os.path.join(root, file)

                # Extract model name from path
                model_name_from_path = extract_model_name(filepath, base_dir)

                if model_name_from_path is None:
                    continue

                # Apply model filter if specified
                if filter_model and filter_model not in model_name_from_path:
                    continue

                # Extract task name
                task_name = file.replace(".json", "")

                # Apply task filter if specified
                if filter_task and filter_task != task_name:
                    continue

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    # Extract main_score
                    main_score = data["scores"]["test"][0]["main_score"] * 100  # Convert to percentage

                    # Store the result
                    if task_name not in results:
                        results[task_name] = {}

                    results[task_name][model_name_from_path] = main_score

                except Exception as e:
                    print(f"  Warning: Skipping {filepath}, error: {e}")

    if not results:
        print("ERROR: No results found!")
        print(f"  Base directory: {base_dir}")
        print(f"  Filter model: {filter_model}")
        print(f"  Filter task: {filter_task}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results).T
    df = df.sort_index()
    df = df.sort_index(axis=1)

    # Get layer baseline scores
    if model_name:
        print(f"\nLooking for layer baseline results for {model_name}...")
        layer_scores = get_layer_baseline_scores(model_name, list(df.index))

        if layer_scores:
            print(f"  ✓ Found layer baseline results for {len(layer_scores)} tasks")

            # Add layer baseline columns
            for task in df.index:
                if task in layer_scores:
                    if 'last_layer' in layer_scores[task]:
                        df.loc[task, 'last_layer'] = layer_scores[task]['last_layer']
                    if 'best_single_layer' in layer_scores[task]:
                        df.loc[task, 'best_single_layer'] = layer_scores[task]['best_single_layer']

            # Reorder columns: last_layer, best_single_layer, then rest
            other_cols = [c for c in df.columns if c not in ['last_layer', 'best_single_layer']]
            if 'last_layer' in df.columns and 'best_single_layer' in df.columns:
                df = df[['last_layer', 'best_single_layer'] + other_cols]
            elif 'last_layer' in df.columns:
                df = df[['last_layer'] + other_cols]
            elif 'best_single_layer' in df.columns:
                df = df[['best_single_layer'] + other_cols]
        else:
            print(f"  Note: No layer baseline results found")
    else:
        print("\n  Note: --model not specified, skipping layer baseline integration")

    # Save to CSV
    output_file = os.path.join(output_dir, output_filename)
    df.to_csv(output_file)

    print(f"\n✓ Saved results to {output_file}")
    print(f"  Shape: {df.shape[0]} tasks × {df.shape[1]} models")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"  Tasks: {list(df.index)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize MTEB results with layer baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With layer baselines
    python3 summarize_with_layers.py \\
        --model TinyLlama-1.1B \\
        --base_dir mteb_results_best_models \\
        --output tinyllama_complete.csv

    # Filter by model and task
    python3 summarize_with_layers.py \\
        --model Pythia-410m \\
        --base_dir mteb_results_best_models \\
        --filter-model Pythia_410m \\
        --filter-task EmotionClassification \\
        --output pythia_emotion.csv
        """
    )

    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        help="Model name (for layer baseline integration)"
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory containing MTEB results"
    )
    parser.add_argument(
        "--output_dir",
        default="./summary",
        help="Output directory (default: ./summary)"
    )
    parser.add_argument(
        "--output",
        dest="output_filename",
        required=True,
        help="Output CSV filename"
    )
    parser.add_argument(
        "--filter-model",
        help="Filter by model pattern (e.g., 'Pythia_410m')"
    )
    parser.add_argument(
        "--filter-task",
        help="Filter by task name"
    )

    args = parser.parse_args()

    main(
        args.model,
        args.base_dir,
        args.output_dir,
        args.output_filename,
        args.filter_model,
        args.filter_task
    )
