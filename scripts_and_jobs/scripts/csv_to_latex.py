#!/usr/bin/env python3
"""
Convert CSV result tables to LaTeX format.

This script converts CSV files (tasks as rows, methods as columns) into
publication-ready LaTeX tables with optional delta calculations.

Features:
- Bold best result per task per model
- Blue color for 2nd best result per task per model
- Red color for 3rd best result per task per model

Note: Your LaTeX document must include: \\usepackage{xcolor}

Usage:
    python scripts_and_jobs/scripts/csv_to_latex.py \\
        --input results.csv \\
        --output table.tex \\
        [--deltas] \\
        [--bold_best] \\
        [--caption "Performance Results"] \\
        [--label "tab:results"]

    # Multiple CSV files (one per model)
    python scripts_and_jobs/scripts/csv_to_latex.py \\
        --input model1.csv model2.csv \\
        --model_names "Model 1" "Model 2" \\
        --output combined_table.tex
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to Python path
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def should_include_column(col_name: str) -> bool:
    """
    Determine if a column should be included in the LaTeX table.
    
    Now includes ALL methods (no filtering).
    
    Args:
        col_name: Column name from CSV
        
    Returns:
        True (always include all columns)
    """
    # Include all columns
    return True


def get_standard_method_name(col_name: str) -> str:
    """
    Standardize method names for LaTeX display.
    
    Rules:
    - cayley → "Cayley"
    - mlp_last_<anything> → "MLP Last"
    - mlp_best_<anything> → "MLP Best"
    - Handle pooling variants (mean, sum)
    
    Args:
        col_name: Original column name from CSV
        
    Returns:
        Standardized method name for LaTeX
    """
    col_lower = col_name.lower()
    
    # Handle MLP variants
    if 'mlp_last' in col_lower:
        return 'MLP Last'
    if 'mlp_best' in col_lower:
        return 'MLP Best'
    # Note: No generic 'MLP' - only MLP Last and MLP Best exist
    
    # Handle other non-graph methods
    if 'deepset' in col_lower:
        return 'DeepSet'
    if 'weighted' in col_lower:
        return 'Weighted'
    if 'last_layer' in col_lower:
        return 'Last Layer'
    if 'best_layer' in col_lower:
        return 'Best Single Layer'
    if 'dwatt' in col_lower:
        return 'DWATT'
    
    # Handle graph-based methods (GIN/GCN)
    prefix = ""
    if 'gcn' in col_lower:
        prefix = "GCN"
    elif 'gin' in col_lower:
        prefix = "GIN"
    
    if not prefix:
        return col_name
    
    # Extract graph topology
    topology = ""

    if 'cayley' in col_lower:
        topology = "Cayley"
    elif 'fully_c' in col_lower or 'fully-c' in col_lower or 'fullyconnected' in col_lower:
        topology = "Fully Connected"
    elif 'linear' in col_lower:
        topology = "Linear"
    elif 'virtual' in col_lower or 'virtualnode' in col_lower:
        topology = "Virtual Node"
    
    # Extract pooling method
    pooling = ""
    if 'mean' in col_lower:
        pooling = "Mean"
    elif 'sum' in col_lower:
        pooling = "Sum"
    elif 'max' in col_lower:
        pooling = "Max"
    
    # Build the final name
    if topology and pooling:
        return f"{prefix} {topology} ({pooling})"
    elif topology:
        return f"{prefix} {topology}"
    elif pooling:
        return f"{prefix} ({pooling})"
    else:
        # Fallback to original name if we can't parse it
        return col_name


def clean_task_name(task_name: str) -> str:
    """
    Clean task names for LaTeX headers.
    
    Removes common suffixes like "Classification", "-amh", etc.
    
    Args:
        task_name: Original task name
        
    Returns:
        Cleaned task name
    """
    cleaned = task_name.replace('Classification', '').replace('-amh', '')
    cleaned = cleaned.strip()
    return cleaned if cleaned else task_name


def filter_and_standardize_dataframe(df: pd.DataFrame, task_col: str = 'Task') -> pd.DataFrame:
    """
    Filter columns and standardize method names.
    
    Args:
        df: Input DataFrame with tasks as rows, methods as columns
        task_col: Name of the task column (default: 'Task')
        
    Returns:
        Filtered and standardized DataFrame
    """
    # Identify task column
    if task_col not in df.columns:
        # Try to find it in index or columns
        if df.index.name == task_col or task_col in df.index.names:
            df = df.reset_index()
        else:
            # Assume first column or index is tasks
            if df.index.name is None:
                df = df.reset_index()
                task_col = df.columns[0]
    
    # Get all columns except task column
    method_cols = [col for col in df.columns if col != task_col]
    
    # Filter columns based on inclusion rules
    cols_to_include = [task_col]
    for col in method_cols:
        if should_include_column(col):
            cols_to_include.append(col)
    
    df_filtered = df[cols_to_include].copy()
    
    # Standardize method names
    rename_dict = {}
    for col in df_filtered.columns:
        if col != task_col:
            rename_dict[col] = get_standard_method_name(col)
    
    df_filtered = df_filtered.rename(columns=rename_dict)
    
    return df_filtered, task_col


def generate_latex_table(
    dfs: List[Tuple[str, pd.DataFrame]],
    task_col: str = 'Task',
    bold_best: bool = True,
    deltas: bool = False,
    caption: str = "Performance Results",
    label: str = "tab:results"
) -> str:
    """
    Generate LaTeX table from one or more DataFrames.
    
    Args:
        dfs: List of (model_name, dataframe) tuples
        task_col: Name of the task column
        bold_best: Whether to bold the best value per task
        deltas: Whether to include delta calculations
        caption: LaTeX table caption
        label: LaTeX table label
        
    Returns:
        LaTeX code as string
    """
    if not dfs:
        return ""
    
    # Get canonical task list from first dataframe
    canonical_tasks = dfs[0][1][task_col].tolist()
    
    # Collect all method names across all dataframes
    all_methods = set()
    for _, df in dfs:
        method_cols = [col for col in df.columns if col != task_col]
        all_methods.update(method_cols)
    
    # Define method order (if methods exist)
    method_order = [
        'Last Layer', 'Best Single Layer', 
        'MLP Last', 'MLP Best',
        'Weighted', 'DWATT',
        # GCN variants
        'GCN Cayley (Mean)', 'GCN Cayley (Attention)', 'GCN Cayley (Sum)', 'GCN Cayley (Max)',
        'GCN Cayley',
        'GCN Fully Connected (Mean)', 'GCN Fully Connected (Attention)', 'GCN Fully Connected (Sum)', 'GCN Fully Connected (Max)',
        'GCN Fully Connected',
        'GCN Linear (Mean)', 'GCN Linear (Attention)', 'GCN Linear (Sum)', 'GCN Linear (Max)',
        'GCN Linear',
        'GCN Virtual Node (Mean)', 'GCN Virtual Node (Attention)', 'GCN Virtual Node (Sum)', 'GCN Virtual Node (Max)',
        'GCN Virtual Node',
        # GIN variants
        'GIN Cayley (Mean)', 'GIN Cayley (Attention)', 'GIN Cayley (Sum)', 'GIN Cayley (Max)',
        'GIN Cayley',
        'GIN Fully Connected (Mean)', 'GIN Fully Connected (Attention)', 'GIN Fully Connected (Sum)', 'GIN Fully Connected (Max)',
        'GIN Fully Connected',
        'GIN Linear (Mean)', 'GIN Linear (Attention)', 'GIN Linear (Sum)', 'GIN Linear (Max)',
        'GIN Linear',
        'GIN Virtual Node (Mean)', 'GIN Virtual Node (Attention)', 'GIN Virtual Node (Sum)', 'GIN Virtual Node (Max)',
        'GIN Virtual Node',
    ]
    
    # Order methods: first by predefined order, then any remaining
    ordered_methods = [m for m in method_order if m in all_methods]
    remaining_methods = sorted([m for m in all_methods if m not in method_order])
    final_method_order = ordered_methods + remaining_methods
    
    # Clean task names for headers
    header_tasks = [clean_task_name(t) for t in canonical_tasks]
    
    # Start LaTeX code
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{ll" + "c" * len(canonical_tasks) + "}")
    latex_lines.append(r"\toprule")
    
    # Header row
    header = "Base Model & Method & " + " & ".join(header_tasks) + r" \\"
    latex_lines.append(header)
    latex_lines.append(r"\midrule")
    
    # Process each model's dataframe
    delta_storage = {} if deltas else None

    for model_idx, (model_name, df) in enumerate(dfs):
        # Ensure task column exists and set as index for reindexing
        if task_col not in df.columns:
            # Task might be in index
            if df.index.name == task_col or (df.index.name is None and len(df) > 0):
                df = df.reset_index()

        # Reindex to canonical tasks
        df_indexed = df.set_index(task_col)
        df_reindexed = df_indexed.reindex(canonical_tasks).reset_index()

        # Work with numeric data
        method_cols = [col for col in df_reindexed.columns if col != task_col]
        df_numeric = df_reindexed.set_index(task_col)[method_cols].apply(pd.to_numeric, errors='coerce')

        # Remove duplicate indices (keep first occurrence) - fix at source
        # Use reset_index and drop_duplicates for more reliable duplicate removal
        if df_numeric.index.duplicated().any():
            # Reset index, drop duplicates on the task column, then set index back
            df_numeric = df_numeric.reset_index()
            task_col_name = df_numeric.columns[0]  # The task column is now first
            df_numeric = df_numeric.drop_duplicates(subset=[task_col_name], keep='first')
            df_numeric = df_numeric.set_index(task_col_name)

        # Calculate top 3 values per task (row) for formatting - WITHIN THIS MODEL ONLY
        # Only consider numeric values, skip NaN
        # Only consider methods that are actually being displayed
        top3_per_task = None
        if bold_best:
            # Filter to only methods that will be displayed
            displayed_methods = [m for m in method_cols if m in final_method_order]
            if displayed_methods:
                # Build top 3 values per task
                top3_per_task = {}
                for task in canonical_tasks:
                    if task in df_numeric.index:
                        task_values = []
                        for method in displayed_methods:
                            if method in df_numeric.columns:
                                val = df_numeric.loc[task, method]
                                # Handle case where loc returns Series
                                if isinstance(val, pd.Series):
                                    val = val.iloc[0] if len(val) > 0 else None
                                if val is not None and not pd.isna(val):
                                    try:
                                        task_values.append(float(val))
                                    except (ValueError, TypeError):
                                        pass
                        if task_values:
                            # Sort in descending order by ACTUAL values (not rounded)
                            # Remove exact duplicates only
                            unique_values = list(set(task_values))
                            sorted_values = sorted(unique_values, reverse=True)

                            top3_per_task[task] = {
                                'first': sorted_values[0] if len(sorted_values) >= 1 else None,
                                'second': sorted_values[1] if len(sorted_values) >= 2 else None,
                                'third': sorted_values[2] if len(sorted_values) >= 3 else None
                            }

                # Keep as dict for easier access
                if not top3_per_task:
                    top3_per_task = None
        
        # Count rows for this model (methods + optional deltas)
        n_methods = len([m for m in final_method_order if m in method_cols])
        n_delta_rows = 0
        
        if deltas:
            # Calculate deltas
            model_deltas = calculate_deltas(df_numeric, method_cols)
            if model_deltas:
                n_delta_rows = len(model_deltas)
                if delta_storage is not None:
                    for key, series in model_deltas.items():
                        if key not in delta_storage:
                            delta_storage[key] = []
                        delta_storage[key].append(series)
        else:
            model_deltas = {}
        
        total_rows = n_methods + n_delta_rows
        first_row = True
        
        # Print method rows
        for method in final_method_order:
            if method not in method_cols:
                continue
            
            row_vals = []
            for task in canonical_tasks:
                if task not in df_numeric.index:
                    row_vals.append("-")
                    continue
                
                try:
                    val = df_numeric.at[task, method]  # Use .at for scalar access
                    # Handle case where .at might return Series (shouldn't happen, but defensive)
                    if isinstance(val, pd.Series):
                        if len(val) > 0:
                            val = val.iloc[0]
                        else:
                            val = None
                except (KeyError, IndexError):
                    val = None
                
                # Check if val is None or NaN (handle both scalar and Series cases)
                if val is None:
                    row_vals.append("-")
                elif isinstance(val, pd.Series):
                    # Shouldn't happen, but handle it defensively
                    if len(val) == 0:
                        row_vals.append("-")
                    else:
                        val_scalar = val.iloc[0]
                        if pd.isna(val_scalar):
                            row_vals.append("-")
                        else:
                            val_float = float(val_scalar)
                            val_str = f"{val_float:.2f}"
                            formatted = False
                            # Format based on rank (1st=bold, 2nd=blue, 3rd=red)
                            if bold_best and top3_per_task is not None:
                                try:
                                    # Check that task exists in top3_per_task
                                    if task in top3_per_task:
                                        top3 = top3_per_task[task]

                                        # Check if this is 1st place (best) - compare actual values
                                        if top3['first'] is not None and abs(val_float - top3['first']) < 1e-9:
                                            val_str = f"\\textbf{{{val_str}}}"
                                            formatted = True

                                        # Check if this is 2nd place
                                        if not formatted and top3['second'] is not None and abs(val_float - top3['second']) < 1e-9:
                                            val_str = f"\\textcolor{{blue}}{{{val_str}}}"
                                            formatted = True

                                        # Check if this is 3rd place
                                        if not formatted and top3['third'] is not None and abs(val_float - top3['third']) < 1e-9:
                                            val_str = f"\\textcolor{{red}}{{{val_str}}}"
                                            formatted = True
                                except (KeyError, IndexError, ValueError, TypeError):
                                    pass
                            row_vals.append(val_str)
                else:
                    # val is a scalar
                    try:
                        if pd.isna(val):
                            row_vals.append("-")
                        else:
                            val_float = float(val)  # Ensure it's a float
                            val_str = f"{val_float:.2f}"
                            formatted = False
                            # Format based on rank (1st=bold, 2nd=blue, 3rd=red)
                            if bold_best and top3_per_task is not None:
                                try:
                                    # Check that task exists in top3_per_task
                                    if task in top3_per_task:
                                        top3 = top3_per_task[task]

                                        # Check if this is 1st place (best) - compare actual values
                                        if top3['first'] is not None and abs(val_float - top3['first']) < 1e-9:
                                            val_str = f"\\textbf{{{val_str}}}"
                                            formatted = True

                                        # Check if this is 2nd place
                                        if not formatted and top3['second'] is not None and abs(val_float - top3['second']) < 1e-9:
                                            val_str = f"\\textcolor{{blue}}{{{val_str}}}"
                                            formatted = True

                                        # Check if this is 3rd place
                                        if not formatted and top3['third'] is not None and abs(val_float - top3['third']) < 1e-9:
                                            val_str = f"\\textcolor{{red}}{{{val_str}}}"
                                            formatted = True
                                except (KeyError, IndexError, ValueError, TypeError):
                                    pass
                            row_vals.append(val_str)
                    except (ValueError, TypeError):
                        row_vals.append("-")
            
            model_cell = ""
            if first_row and total_rows > 0:
                model_cell = f"\\multirow{{{total_rows}}}{{*}}{{{model_name}}}"
                first_row = False
            
            latex_lines.append(f"{model_cell} & {method} & " + " & ".join(row_vals) + r" \\")
        
        # Print delta rows if enabled
        if deltas and model_deltas:
            if n_methods > 0:
                latex_lines.append(r"\cmidrule{2-" + str(2 + len(canonical_tasks)) + "}")
            
            for key, series in model_deltas.items():
                row_vals = []
                for task in canonical_tasks:
                    try:
                        if hasattr(series, 'index') and task in series.index:
                            val = series.at[task]  # Use .at for scalar access
                        else:
                            val = series.get(task) if hasattr(series, 'get') else None
                    except (KeyError, IndexError):
                        val = None
                    
                    if val is None or pd.isna(val):
                        row_vals.append("-")
                    else:
                        val_float = float(val)  # Ensure it's a float
                        row_vals.append(f"{val_float:+.2f}")
                
                display_label = format_delta_label(key)
                latex_lines.append(f" & $\\Delta$ {display_label} & " + " & ".join(row_vals) + r" \\")
        
        # Model separator
        if model_idx < len(dfs) - 1:
            latex_lines.append(r"\midrule \midrule")
    
    # Average deltas across models (if multiple models and deltas enabled)
    if deltas and delta_storage and len(dfs) > 1:
        latex_lines.append(r"\midrule \midrule")
        avg_deltas = {}
        for key, series_list in delta_storage.items():
            if series_list:
                df_concat = pd.concat(series_list, axis=1)
                avg_deltas[key] = df_concat.mean(axis=1)
        
        first_avg = True
        for key, avg_series in avg_deltas.items():
            row_vals = []
            for task in canonical_tasks:
                try:
                    if hasattr(avg_series, 'index') and task in avg_series.index:
                        val = avg_series.at[task]  # Use .at for scalar access
                    else:
                        val = avg_series.get(task) if hasattr(avg_series, 'get') else None
                except (KeyError, IndexError):
                    val = None
                
                if val is None or pd.isna(val):
                    row_vals.append("-")
                else:
                    val_float = float(val)  # Ensure it's a float
                    row_vals.append(f"{val_float:+.2f}")
            
            cell = ""
            if first_avg:
                cell = f"\\multirow{{{len(avg_deltas)}}}{{*}}{{Average $\\Delta$}}"
                first_avg = False
            
            display_label = format_delta_label(key)
            latex_lines.append(f"{cell} & {display_label} & " + " & ".join(row_vals) + r" \\")
    
    # End table
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    latex_lines.append(r"\end{table*}")
    
    return "\n".join(latex_lines)


def calculate_deltas(df_numeric: pd.DataFrame, method_cols: List[str]) -> Dict[str, pd.Series]:
    """
    Calculate delta improvements between methods.
    
    Compares GCN Cayley and GIN Cayley (best across pooling variants) vs all baselines:
    - DeepSet
    - MLP Last Layer
    - MLP Best Layer (if exists)
    - Best Single Layer
    - Last Layer
    - DWATT (if exists)
    
    Args:
        df_numeric: DataFrame with numeric values (tasks as index, methods as columns)
        method_cols: List of method column names
        
    Returns:
        Dictionary of delta series keyed by comparison name
    """
    deltas = {}
    
    # Helper to get series or None, ensuring unique index
    def get_series(name):
        if name not in method_cols:
            return None
        # df_numeric should already have no duplicates, but ensure it
        series = df_numeric[name].copy()
        # Remove duplicates by grouping and taking first (defensive)
        if series.index.duplicated().any():
            series = series.groupby(series.index).first()
        # Ensure index is unique
        if not series.index.is_unique:
            series = series[~series.index.duplicated(keep='first')]
        return series
    
    # Helper to safely subtract two Series, handling duplicate indices
    def safe_subtract(s1, s2):
        """Subtract s2 from s1, handling duplicate indices."""
        if s1 is None or s2 is None:
            return None
        
        # Remove duplicates from both series using groupby (keep first occurrence)
        if s1.index.duplicated().any():
            s1_clean = s1.groupby(s1.index).first()
        else:
            s1_clean = s1.copy()
            
        if s2.index.duplicated().any():
            s2_clean = s2.groupby(s2.index).first()
        else:
            s2_clean = s2.copy()
        
        # Get common indices (intersection) - ensure it's unique
        common_idx = s1_clean.index.intersection(s2_clean.index)
        
        if len(common_idx) == 0:
            return None
        
        # Ensure common_idx is unique (remove any duplicates)
        if isinstance(common_idx, pd.Index) and common_idx.duplicated().any():
            common_idx = common_idx[~common_idx.duplicated(keep='first')]
        
        # Use direct indexing instead of reindex to avoid alignment issues
        # Extract values for common indices
        s1_values = []
        s2_values = []
        valid_idx = []
        
        for idx in common_idx:
            try:
                # Use .get() or .loc[] instead of .at[] for safer access
                if idx in s1_clean.index and idx in s2_clean.index:
                    val1 = s1_clean.loc[idx]
                    val2 = s2_clean.loc[idx]
                    # Handle case where loc returns Series (if duplicates somehow exist)
                    if isinstance(val1, pd.Series):
                        val1 = val1.iloc[0]
                    if isinstance(val2, pd.Series):
                        val2 = val2.iloc[0]
                    
                    if val1 is not None and val2 is not None and not pd.isna(val1) and not pd.isna(val2):
                        s1_values.append(float(val1))
                        s2_values.append(float(val2))
                        valid_idx.append(idx)
            except (KeyError, IndexError, ValueError):
                continue
        
        if len(valid_idx) == 0:
            return None
        
        # Create new Series with the result
        result_values = np.array(s1_values) - np.array(s2_values)
        result = pd.Series(result_values, index=valid_idx)
        
        return result
    
    # Get baseline methods
    s_deepset = get_series('DeepSet')
    s_mlp_last = get_series('MLP Last')
    s_mlp_best = get_series('MLP Best')
    s_best_single = get_series('Best Single Layer')
    s_last = get_series('Last Layer')
    s_dwatt = get_series('DWATT')  # Optional, may not exist
    
    # Get best GCN Cayley (max across Mean and Sum only, excluding Attention)
    gcn_cayley_methods = [m for m in method_cols 
                          if m.startswith('GCN Cayley') and '(Attention)' not in m]
    s_best_gcn_cayley = None
    if gcn_cayley_methods:
        gcn_df = df_numeric[gcn_cayley_methods].copy()
        # Remove duplicate indices by grouping and taking first
        if gcn_df.index.duplicated().any():
            gcn_df = gcn_df.groupby(gcn_df.index).first()
        s_best_gcn_cayley = gcn_df.max(axis=1)
        # Ensure result has no duplicates
        if s_best_gcn_cayley.index.duplicated().any():
            s_best_gcn_cayley = s_best_gcn_cayley.groupby(s_best_gcn_cayley.index).first()
    
    # Get best GIN Cayley (max across Mean and Sum only, excluding Attention)
    gin_cayley_methods = [m for m in method_cols 
                          if m.startswith('GIN Cayley') and '(Attention)' not in m]
    s_best_gin_cayley = None
    if gin_cayley_methods:
        gin_df = df_numeric[gin_cayley_methods].copy()
        # Remove duplicate indices by grouping and taking first
        if gin_df.index.duplicated().any():
            gin_df = gin_df.groupby(gin_df.index).first()
        s_best_gin_cayley = gin_df.max(axis=1)
        # Ensure result has no duplicates
        if s_best_gin_cayley.index.duplicated().any():
            s_best_gin_cayley = s_best_gin_cayley.groupby(s_best_gin_cayley.index).first()
    
    # Calculate deltas: GCN Cayley vs all baselines
    if s_best_gcn_cayley is not None:
        if s_deepset is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_deepset)
            if delta is not None:
                deltas['GCN Cayley vs DeepSet'] = delta
        
        if s_mlp_last is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_mlp_last)
            if delta is not None:
                deltas['GCN Cayley vs MLP Last'] = delta
        
        if s_mlp_best is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_mlp_best)
            if delta is not None:
                deltas['GCN Cayley vs MLP Best'] = delta
        
        if s_best_single is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_best_single)
            if delta is not None:
                deltas['GCN Cayley vs Best Single Layer'] = delta
        
        if s_last is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_last)
            if delta is not None:
                deltas['GCN Cayley vs Last Layer'] = delta
        
        if s_dwatt is not None:
            delta = safe_subtract(s_best_gcn_cayley, s_dwatt)
            if delta is not None:
                deltas['GCN Cayley vs DWATT'] = delta
    
    # Calculate deltas: GIN Cayley vs all baselines
    if s_best_gin_cayley is not None:
        if s_deepset is not None:
            delta = safe_subtract(s_best_gin_cayley, s_deepset)
            if delta is not None:
                deltas['GIN Cayley vs DeepSet'] = delta
        
        if s_mlp_last is not None:
            delta = safe_subtract(s_best_gin_cayley, s_mlp_last)
            if delta is not None:
                deltas['GIN Cayley vs MLP Last'] = delta
        
        if s_mlp_best is not None:
            delta = safe_subtract(s_best_gin_cayley, s_mlp_best)
            if delta is not None:
                deltas['GIN Cayley vs MLP Best'] = delta
        
        if s_best_single is not None:
            delta = safe_subtract(s_best_gin_cayley, s_best_single)
            if delta is not None:
                deltas['GIN Cayley vs Best Single Layer'] = delta
        
        if s_last is not None:
            delta = safe_subtract(s_best_gin_cayley, s_last)
            if delta is not None:
                deltas['GIN Cayley vs Last Layer'] = delta
        
        if s_dwatt is not None:
            delta = safe_subtract(s_best_gin_cayley, s_dwatt)
            if delta is not None:
                deltas['GIN Cayley vs DWATT'] = delta
    
    return deltas


def format_delta_label(key: str) -> str:
    """
    Format delta key for LaTeX display.
    
    Args:
        key: Delta comparison key (e.g., "GCN Cayley (Mean) vs MLP")
        
    Returns:
        Formatted label
    """
    return key.replace(' vs ', ' vs ')


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV result tables to LaTeX format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input CSV file(s). Multiple files will be combined with model names."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output LaTeX file path"
    )
    
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Model names for multiple CSV files (must match number of input files)"
    )
    
    parser.add_argument(
        "--task_col",
        type=str,
        default="Task",
        help="Name of the task column (default: 'Task')"
    )
    
    parser.add_argument(
        "--deltas",
        action="store_true",
        help="Include delta calculations (improvements between methods)"
    )
    
    parser.add_argument(
        "--bold_best",
        action="store_true",
        default=True,
        help="Bold the best value per task (default: True)"
    )
    
    parser.add_argument(
        "--no_bold_best",
        action="store_true",
        help="Disable bolding of best values"
    )
    
    parser.add_argument(
        "--caption",
        type=str,
        default="Performance Results",
        help="LaTeX table caption"
    )
    
    parser.add_argument(
        "--label",
        type=str,
        default="tab:results",
        help="LaTeX table label"
    )
    
    args = parser.parse_args()
    
    # Handle bold_best flag
    bold_best = args.bold_best and not args.no_bold_best
    
    # Validate inputs
    input_files = args.input
    if args.model_names:
        if len(args.model_names) != len(input_files):
            print(f"ERROR: Number of model names ({len(args.model_names)}) must match number of input files ({len(input_files)})")
            sys.exit(1)
        model_names = args.model_names
    else:
        # Auto-detect model names from filenames
        def extract_model_name(filepath):
            """Extract model name from filepath based on patterns."""
            filename = Path(filepath).stem  # Get filename without extension
            filename_lower = filename.lower()
            
            # Check for model patterns in filename
            if 'pythia410' in filename_lower or 'pythia-410' in filename_lower:
                return 'Pythia 410m'
            elif 'tinyllama' in filename_lower or 'tiny-llama' in filename_lower:
                return 'TinyLlama 1.1B'
            elif 'llama3' in filename_lower or 'llama-3' in filename_lower:
                return 'Llama3 8B'
            else:
                # Fallback to filename stem if no pattern matches
                return filename
        
        model_names = [extract_model_name(f) for f in input_files]
    
    # Load and process dataframes
    print(f"Loading {len(input_files)} CSV file(s)...")
    dfs = []
    
    for i, (input_file, model_name) in enumerate(zip(input_files, model_names)):
        if not os.path.exists(input_file):
            print(f"ERROR: File not found: {input_file}")
            sys.exit(1)
        
        print(f"  [{i+1}/{len(input_files)}] Processing: {input_file} (Model: {model_name})")
        df = pd.read_csv(input_file)
        
        # Filter and standardize
        df_filtered, task_col = filter_and_standardize_dataframe(df, args.task_col)
        dfs.append((model_name, df_filtered))
    
    # Generate LaTeX
    print("\nGenerating LaTeX table...")
    latex_code = generate_latex_table(
        dfs,
        task_col=task_col,
        bold_best=bold_best,
        deltas=args.deltas,
        caption=args.caption,
        label=args.label
    )
    
    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"\n{'='*70}")
    print(f"LaTeX table saved to: {output_path}")
    print(f"{'='*70}\n")
    print("Preview (first 20 lines):")
    print("-" * 70)
    print("\n".join(latex_code.split("\n")[:20]))
    if len(latex_code.split("\n")) > 20:
        print("...")
        print(f"(Total: {len(latex_code.split())} lines)")


if __name__ == "__main__":
    main()

