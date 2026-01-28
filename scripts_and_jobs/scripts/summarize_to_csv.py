"""
Summarize MTEB Evaluation Results to CSV

This script aggregates MTEB test results from multiple models into a single CSV file
with tasks as rows and models as columns.

Usage:
    python3 summarize_to_csv.py \
        --base_dir mteb_results_best_models \
        --output_dir ./summary \
        --output_filename pythia_410m_all_methods.csv

Input:
    - base_dir: Directory containing MTEB results organized by method
      Structure: base_dir/{method}/{model}/{config}/{task}.json

      Example directory structure:
        mteb_results_best_models/
        ├── best_single_layer/Pythia_410m/{task}.json
        ├── weighted/Pythia_410m/{task}.json
        ├── mlp/Pythia_410m/{task}.json
        ├── gin/Pythia_410m/cayley/{task}.json
        └── gcn/Pythia_410m/linear/{task}.json

Output:
    - CSV file with:
      * Rows: Task names (Banking77Classification, EmotionClassification, etc.)
      * Columns: Model configurations
        - Simple methods: best_single_layer, weighted, mlp (just method name)
        - GNN methods: gin_Pythia_410m_cayley, gcn_Pythia_410m_linear (full config)
      * Values: Test accuracy (main_score from MTEB results)

Example output CSV:
    Task                          best_single_layer  weighted  mlp   gin_Pythia_410m_cayley  gcn_Pythia_410m_linear
    Banking77Classification       0.667              0.626     0.71  0.685                   0.672
    EmotionClassification         0.35               0.33      0.42  0.38                    0.36
    ...
"""
import os
import json
import pandas as pd
import argparse

def extract_model_name(filepath, base_dir):
    """
    Extract model name from path structure.

    - GNN: base_dir/gcn/Pythia_410m/cayley/... -> gcn_Pythia_410m_cayley
    - MLP: base_dir/mlp/Pythia_410m/... -> mlp_Pythia_410m
    - Weighted: base_dir/weighted/Pythia_410m/... -> weighted_Pythia_410m
    - Best Single Layer: base_dir/best_single_layer/Pythia_410m/... -> best_single_layer_Pythia_410m
    """
    # Get the relative path from base_dir
    rel_path = os.path.relpath(filepath, base_dir)

    # Split the path and take the first 3 components (method/model/config)
    parts = rel_path.split(os.sep)

    if len(parts) < 1:
        return None

    method = parts[0]

    # For simple methods (MLP, Weighted, Best Single Layer), include method + model name
    if method in ["mlp", "weighted", "best_single_layer"]:
        if len(parts) >= 2:
            # Return method_model (e.g., mlp_Pythia_410m, weighted_TinyLlama_1.1B)
            return f"{parts[0]}_{parts[1]}"
        else:
            return method

    # For GNN (gcn/gin), extract full configuration
    if len(parts) >= 3:
        model_name = "_".join(parts[:3])
        return model_name

    return None

def main(base_dir, output_dir, output_filename):
    """
    Process JSON files and create a summary CSV with tasks as rows and models as columns.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results: {task_name: {model_name: score}}
    results = {}
    
    # Walk through directory and subdirectories
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json") and file != "model_meta.json":
                filepath = os.path.join(root, file)
                
                # Extract model name from path
                model_name = extract_model_name(filepath, base_dir)
                
                if model_name is None:
                    print(f"Skipping {filepath}, couldn't extract model name")
                    continue
                
                # Extract task name (filename without .json)
                task_name = file.replace(".json", "")
                
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    
                    # Extract main_score
                    main_score = data["scores"]["test"][0]["main_score"]
                    
                    # Store the result
                    if task_name not in results:
                        results[task_name] = {}
                    
                    results[task_name][model_name] = main_score
                    
                except Exception as e:
                    print(f"Skipping {filepath}, error: {e}")
    
    # Convert to DataFrame
    # Each row is a task, each column is a model
    df = pd.DataFrame(results).T  # Transpose so tasks are rows
    
    # Sort by task name (index) and model name (columns)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    
    # Save to CSV
    output_file = os.path.join(output_dir, output_filename)
    df.to_csv(output_file)
    
    print(f"Saved results to {output_file}")
    print(f"Shape: {df.shape[0]} tasks x {df.shape[1]} models")
    print(f"\nModels found: {list(df.columns)}")
    print(f"\nFirst few tasks: {list(df.index[:5])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize MTEB results into a task x model CSV")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing the results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./summary",
        help="Output directory for the CSV file (default: ./summary)"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="task_model_scores.csv",
        help="Output CSV filename (default: task_model_scores.csv)"
    )
    
    args = parser.parse_args()
    
    main(args.base_dir, args.output_dir, args.output_filename)
