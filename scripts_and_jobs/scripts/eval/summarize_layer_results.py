#!/usr/bin/env python3
"""
Summarize layer evaluation results into a DataFrame.
Rows = layers, Columns = tasks, Values = accuracy
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List


def load_layer_results(results_base_dir: str, model_name: str) -> Dict[str, Dict[int, dict]]:
    """
    Load all layer evaluation results for a model.

    Args:
        results_base_dir: Base directory containing results
        model_name: Model name

    Returns:
        Dict mapping task_name -> {layer_idx: results_dict}
    """
    results_dir = Path(results_base_dir) / model_name

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = {}

    # Walk through task directories
    for task_dir in results_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        task_results = {}

        # Load all layer JSON files
        for json_file in task_dir.glob("layer_*.json"):
            # Extract layer index from filename
            layer_idx = int(json_file.stem.split("_")[1])

            # Load results
            with open(json_file, 'r') as f:
                results = json.load(f)

            task_results[layer_idx] = results

        if task_results:
            all_results[task_name] = task_results

    return all_results


def create_summary_dataframe(
    results: Dict[str, Dict[int, dict]],
    metric: str = "accuracy"
) -> pd.DataFrame:
    """
    Create summary DataFrame from results.

    Args:
        results: Dict mapping task_name -> {layer_idx: results_dict}
        metric: Metric to extract (accuracy, f1_macro, f1_micro)

    Returns:
        DataFrame with layers as rows and tasks as columns
    """
    # Determine all layers and tasks
    all_layers = set()
    all_tasks = list(results.keys())

    for task_results in results.values():
        all_layers.update(task_results.keys())

    all_layers = sorted(all_layers)

    # Build DataFrame
    data = {}

    for task_name in all_tasks:
        task_results = results[task_name]
        task_values = []

        for layer_idx in all_layers:
            if layer_idx in task_results:
                value = task_results[layer_idx].get(metric, None)
            else:
                value = None

            task_values.append(value)

        data[task_name] = task_values

    df = pd.DataFrame(data, index=all_layers)
    df.index.name = "layer"

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Summarize layer evaluation results into DataFrame"
    )

    parser.add_argument("--results_dir", type=str, default="./layer_eval_res",
                        help="Base directory with layer evaluation results")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (e.g., Pythia_410m)")
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "f1_macro", "f1_micro"],
                        help="Metric to summarize")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save summary CSV (default: {results_dir}/{model_name}_summary.csv)")
    parser.add_argument("--output_excel", type=str, default=None,
                        help="Path to save summary Excel file (optional)")

    args = parser.parse_args()

    print("=" * 70)
    print("LAYER EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Results dir: {args.results_dir}")
    print(f"Metric: {args.metric}")
    print("=" * 70)

    # Load results
    print("\nLoading results...")
    results = load_layer_results(args.results_dir, args.model_name)

    if not results:
        print("No results found!")
        return

    print(f"Found results for {len(results)} tasks:")
    for task_name, task_results in results.items():
        print(f"  {task_name}: {len(task_results)} layers")

    # Create summary DataFrame
    print(f"\nCreating summary DataFrame (metric={args.metric})...")
    df = create_summary_dataframe(results, metric=args.metric)

    print(f"\nSummary DataFrame shape: {df.shape}")
    print("\nPreview:")
    print(df)

    # Compute statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    print("\nBest layer per task:")
    for task in df.columns:
        best_layer = df[task].idxmax()
        best_value = df[task].max()
        print(f"  {task}: Layer {best_layer} ({args.metric}={best_value:.4f})")

    print("\nMean performance per layer:")
    layer_means = df.mean(axis=1)
    best_overall_layer = layer_means.idxmax()
    print(f"  Best overall layer: {best_overall_layer} (mean {args.metric}={layer_means[best_overall_layer]:.4f})")

    # Save CSV
    if args.output_csv is None:
        output_csv = Path(args.results_dir) / f"{args.model_name}_summary_{args.metric}.csv"
    else:
        output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    print(f"\n✓ Summary saved to: {output_csv}")

    # Save Excel if requested
    if args.output_excel:
        output_excel = Path(args.output_excel)
        output_excel.parent.mkdir(parents=True, exist_ok=True)

        # Create Excel with multiple sheets
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # Summary sheet
            df.to_excel(writer, sheet_name='Summary')

            # Statistics sheet
            stats_data = {
                'Task': [],
                'Best Layer': [],
                f'Best {args.metric}': []
            }

            for task in df.columns:
                stats_data['Task'].append(task)
                stats_data['Best Layer'].append(df[task].idxmax())
                stats_data['Best {args.metric}'].append(df[task].max())

            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        print(f"✓ Excel saved to: {output_excel}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
