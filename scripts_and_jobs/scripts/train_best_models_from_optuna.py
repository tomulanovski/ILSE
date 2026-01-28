#!/usr/bin/env python3
"""
Generate SLURM job files to train best GIN/GCN models from Optuna results.

Queries the Optuna database for best hyperparameters per (task, model, graph_type, gin_mlp_layers)
and generates SLURM job files that call basic_gin_trainer.py with those configurations.

Usage:
    python3 train_best_models_from_optuna.py --dry_run  # Preview
    python3 train_best_models_from_optuna.py            # Generate jobs
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import optuna

# Add parent directory to path to import cluster_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from cluster_config import get_cluster_config, get_cluster_type

# Load environment variables
load_dotenv()

# Get paths from .env
CONDA_PATH = os.getenv("GNN_CONDA_PATH")
CONDA_ENV = os.getenv("GNN_CONDA_ENV", "final_project")
BASE_DIR = os.getenv("GNN_REPO_DIR", os.getcwd())


def get_best_configurations(storage_url: str, filter_study: str = None, v2: bool = False) -> List[Dict]:
    """
    Query Optuna database and extract best trial for each:
    (task, model_family, model_size, graph_type, gin_mlp_layers/other params)

    Args:
        storage_url: PostgreSQL connection URL
        filter_study: Optional pattern to filter study names (e.g., "gin_", "Pythia_410m")
        v2: If True, only consider _cayley_precomputed studies. If False, only consider _precomputed (exclude cayley)

    Returns:
        List of configuration dicts with hyperparameters and metadata
    """
    print("Querying Optuna database...")
    summaries = optuna.get_all_study_summaries(storage=storage_url)

    # Filter studies based on pattern if provided
    if filter_study:
        summaries = [s for s in summaries if filter_study in s.study_name]
        print(f"Filtered to {len(summaries)} studies matching '{filter_study}'")

    # Filter for GIN/MLP/Weighted/LoRA/DeepSet/DWAtt studies (exclude other experiments)
    # Include both regular and linear versions
    valid_methods = ["gin", "mlp", "weighted", "lora", "deepset", "dwatt", "linear_gin", "linear_mlp", "linear_deepset"]
    studies = [s for s in summaries if any(s.study_name.startswith(m + "_") for m in valid_methods)]
    print(f"Found {len(studies)} studies (GIN/MLP/Weighted/LoRA/DeepSet/DWAtt + Linear variants)")

    configurations = []

    # Filter studies based on cayley flag
    if v2:
        # cayley mode: Only keep studies ending with _cayley_precomputed
        studies = [s for s in studies if s.study_name.endswith("_cayley_precomputed")]
        print(f"cayley mode: Filtered to {len(studies)} cayley studies (_cayley_precomputed)")
    else:
        # baseline mode: Only keep studies ending with _precomputed (exclude _cayley_precomputed)
        studies = [s for s in studies if s.study_name.endswith("_precomputed") and not s.study_name.endswith("_cayley_precomputed")]
        print(f"baseline mode: Filtered to {len(studies)} baseline studies (_precomputed, excluding cayley)")

    # Group studies by their base name (without _precomputed or _cayley_precomputed suffix) to avoid duplicates
    study_map = {}
    for summary in studies:
        study_name = summary.study_name

        # Strip suffix for base name
        if v2:
            base_name = study_name.replace("_cayley_precomputed", "")
        else:
            base_name = study_name.replace("_precomputed", "")

        # Since we already filtered by v2 flag, just deduplicate by base name
        if base_name not in study_map:
            study_map[base_name] = summary

    print(f"After deduplication: {len(study_map)} unique configurations\n")

    for base_name, summary in study_map.items():
        study_name = summary.study_name

        # Strip both _cayley_precomputed and _precomputed suffix for parsing (but keep original for loading)
        study_name_for_parsing = study_name.replace("_cayley_precomputed", "").replace("_precomputed", "")

        # Parse study name - handle formats:
        # 1. STS GIN: {method}_{graph_type}_sts_{task}_{model}_{size}
        # 2. Classification GIN (NEW): {method}_{graph_type}_{task}_{model}_{size}
        # 3. Classification GIN (OLD suffix): {method}_{task}_{model}_{size}_{graph_type}
        # 4. Old format: {method}_{task}_{model}_{size} (no graph_type)
        parts = study_name_for_parsing.split("_")
        if len(parts) < 4:
            print(f"  ⚠️ Skipping {study_name} - unexpected format")
            continue

        # Handle compound method names like "linear_gin" (split on "_" breaks these)
        if parts[0] == "linear" and len(parts) > 1 and parts[1] in ["gin", "mlp", "deepset"]:
            method = f"{parts[0]}_{parts[1]}"  # "linear_gin", "linear_mlp", "linear_deepset"
        else:
            method = parts[0]  # "gin", "mlp", "weighted", "lora", "deepset"

        # Detect linear methods and extract base method
        is_linear = method.startswith("linear_")
        base_method = method.replace("linear_", "") if is_linear else method

        known_graph_types = ["linear", "virtual_node", "cayley", "cayley", "hierarchical", "fully_connected"]

        # Initialize variables
        graph_type_from_name = None
        model_size = None
        model_family = None
        task = None

        # Check for multi-part graph types first (e.g., "fully_connected") - only for non-linear GIN
        if method == "gin" and len(parts) >= 6:
            # Try two-part graph type first (e.g., "fully_connected")
            two_part_graph = f"{parts[1]}_{parts[2]}"
            if two_part_graph in known_graph_types:
                if parts[3] == "sts":
                    # STS GIN format: gin_{graph_type}_sts_{task}_{model}_{size}
                    graph_type_from_name = two_part_graph
                    model_size = parts[-1]
                    model_family = parts[-2]
                    task = "_".join(parts[3:-2])  # e.g., "sts_STSBenchmark"
                else:
                    # Classification GIN format: gin_{graph_type}_{task}_{model}_{size}
                    graph_type_from_name = two_part_graph
                    model_size = parts[-1]
                    model_family = parts[-2]
                    task = "_".join(parts[3:-2])  # Everything between graph_type and model

        # If not found, check single-part graph types
        if graph_type_from_name is None:
            # Check for STS GIN format: gin_{graph_type}_sts_...
            if method == "gin" and len(parts) >= 5 and parts[1] in known_graph_types and parts[2] == "sts":
                # STS GIN format: gin_{graph_type}_sts_{task}_{model}_{size}
                graph_type_from_name = parts[1]  # e.g., "linear"
                model_size = parts[-1]            # e.g., "1.1B"
                model_family = parts[-2]          # e.g., "TinyLlama"
                task = "_".join(parts[2:-2])      # e.g., "sts_STSBenchmark"
            elif method in ["linear_gin", "linear_mlp", "linear_deepset"]:
                # Linear STS format: linear_{method}_sts_{task}_{model}_{size}
                # No graph_type in name (fixed architecture)
                graph_type_from_name = "cayley" if method == "linear_gin" else None
                model_size = parts[-1]            # e.g., "410m"
                model_family = parts[-2]          # e.g., "Pythia"
                # Skip first 2 parts (linear + gin/mlp/deepset) to get task
                task = "_".join(parts[2:-2])      # e.g., "sts_STSBenchmark"
            elif method == "gin" and len(parts) >= 5 and parts[1] in known_graph_types and parts[2] != "sts":
                # NEW Classification GIN format: gin_{graph_type}_{task}_{model}_{size}
                graph_type_from_name = parts[1]  # e.g., "cayley"
                model_size = parts[-1]            # e.g., "410m"
                model_family = parts[-2]          # e.g., "Pythia"
                task = "_".join(parts[2:-2])      # Everything between graph_type and model
            elif len(parts) >= 5 and parts[-1] in known_graph_types:
                # OLD Classification GIN format (suffix): {method}_{task}_{model}_{size}_{graph_type}
                graph_type_from_name = parts[-1]
                model_size = parts[-2]    # e.g., "410m"
                model_family = parts[-3]  # e.g., "Pythia"
                task = "_".join(parts[1:-3])  # Everything between method and model
            else:
                # Old format: no graph_type in name (MLP, Weighted, DeepSet, or old GIN)
                graph_type_from_name = None
                model_family = parts[-2]  # e.g., "Pythia"
                model_size = parts[-1]    # e.g., "410m"
                task = "_".join(parts[1:-2])  # Everything between method and model

        # Detect if this is an STS task (task starts with "sts_")
        is_sts_task = task.startswith("sts_")

        # Load study
        try:
            study = optuna.load_study(study_name, storage=storage_url)
        except Exception as e:
            print(f"  ⚠️ Error loading {study_name}: {e}")
            continue

        # Get all completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if not completed_trials:
            print(f"  ⚠️ {study_name}: No completed trials")
            continue

        # Group trials based on method-specific grouping keys
        # Use base_method for grouping (strips "linear_" prefix)
        groups = {}
        for trial in completed_trials:
            if base_method == "gin":
                # Group by (graph_type, method_type, node_to_choose)
                # method_type: "gcn" if gin_mlp_layers=0, else "gin"
                # node_to_choose: pooling method (mean, sum, last, etc.)
                # If study name has graph_type suffix, prefer it (for filtered studies)
                # Otherwise, use graph_type from trial params
                if graph_type_from_name is not None:
                    graph_type = graph_type_from_name
                else:
                    graph_type = trial.params.get('graph_type', 'unknown')
                gin_mlp_layers = trial.params.get('gin_mlp_layers', 1)
                method_type = "gcn" if gin_mlp_layers == 0 else "gin"
                node_to_choose = trial.params.get('node_to_choose', 'mean')
                key = (graph_type, method_type, node_to_choose)
            elif base_method == "mlp":
                # Don't group MLP - just take the single best trial per task
                key = ('best',)
            elif base_method == "weighted":
                # Don't group Weighted - just take the single best trial per task
                key = ('best',)
            elif base_method == "dwatt":
                # Don't group DWAtt - just take the single best trial per task
                # Paper-faithful architecture, only training hyperparameters vary
                key = ('best',)
            elif base_method == "lora":
                # Group LoRA by lora_r (rank)
                lora_r = trial.params.get('lora_r', 2)
                key = (lora_r,)
            elif base_method == "deepset":
                # Group DeepSet by variant (for linear_deepset: pre1_post0 vs pre0_post1)
                # For non-linear deepset, group by (pooling_type, pre_layers, post_layers)
                if is_linear:
                    # Linear DeepSet: group by variant
                    variant = trial.params.get('variant', 'pre0_post1')
                    key = (variant,)
                else:
                    # Non-linear DeepSet: group by full architecture
                    pooling = trial.params.get('deepset_pooling_type', 'mean')
                    pre = trial.params.get('deepset_pre_pooling_layers', 0)
                    post = trial.params.get('deepset_post_pooling_layers', 1)
                    key = (pooling, pre, post)
            else:
                key = ('default',)

            if key not in groups:
                groups[key] = []
            groups[key].append(trial)

        # Extract best trial from each group
        for group_key, trials in groups.items():
            best_trial = max(trials, key=lambda t: t.value)

            # Base config for all methods
            config = {
                'method': method,
                'base_method': base_method,  # NEW: base method without "linear_" prefix
                'is_linear': is_linear,      # NEW: flag for linear models
                'task': task,
                'is_sts': is_sts_task,
                'model_family': model_family,
                'model_size': model_size,
                'batch_size': best_trial.params.get('batch_size', 64),
                'lr': best_trial.params.get('lr', 1e-3),
                'weight_decay': best_trial.params.get('weight_decay', 1e-4),
                'val_acc': best_trial.value,
                'trial_number': best_trial.number,
                'study_name': study_name,
            }

            # Add method-specific parameters
            if base_method == "gin":
                # For linear_gin, some parameters are FIXED (not in trial.params)
                if is_linear:
                    # Linear GIN: fixed to 1 layer, hidden_dim=256, graph=cayley
                    gin_layers = 1  # FIXED for linear_gin
                    gin_hidden_dim = 256  # FIXED for linear_gin
                else:
                    # Non-linear GIN: these are searchable hyperparameters
                    gin_layers = best_trial.params.get('gin_layers', 2)
                    gin_hidden_dim = best_trial.params.get('gin_hidden_dim', 256)

                # Determine pool_real_nodes_only
                # - cayley studies: FIXED to True (not in trial.params)
                # - Linear GIN: FIXED to True (not in trial.params)
                # - Non-linear baseline studies: searchable (in trial.params), defaults to False
                if v2 or is_linear:
                    pool_real_nodes_only = True
                else:
                    pool_real_nodes_only = best_trial.params.get('pool_real_nodes_only', False)

                config.update({
                    'graph_type': group_key[0],
                    'method_type': group_key[1],  # "gin" or "gcn"
                    'node_to_choose': group_key[2],  # pooling method from group key
                    'gin_mlp_layers': best_trial.params.get('gin_mlp_layers', 1),
                    'gin_hidden_dim': gin_hidden_dim,
                    'gin_layers': gin_layers,
                    'dropout': best_trial.params.get('dropout', 0.1),
                    'pool_real_nodes_only': pool_real_nodes_only,
                    'train_eps': best_trial.params.get('train_eps', False),
                })
            elif base_method == "mlp":
                # For linear_mlp, fixed parameters are not in trial.params, so use correct defaults
                if is_linear:
                    # Linear MLP: fixed to 1 layer, last input, hidden_dim=256
                    mlp_layers = 1  # FIXED for linear_mlp
                    mlp_input = 'last'  # FIXED for linear_mlp
                    mlp_hidden_dim = 256  # FIXED for linear_mlp
                else:
                    # Non-linear MLP: these are searchable hyperparameters
                    mlp_layers = best_trial.params.get('mlp_layers', 2)
                    mlp_input = best_trial.params.get('mlp_input', 'last')
                    mlp_hidden_dim = best_trial.params.get('mlp_hidden_dim', 256)

                config.update({
                    'mlp_input': mlp_input,
                    'mlp_layers': mlp_layers,
                    'mlp_hidden_dim': mlp_hidden_dim,
                    'dropout': best_trial.params.get('dropout', 0.1),
                })
            elif base_method == "weighted":
                # Weighted has minimal parameters (no method-specific architecture params)
                pass
            elif base_method == "dwatt":
                # DWAtt: Depth-Wise Attention (ElNokrashy et al. 2024)
                # Paper-faithful architecture: bottleneck_ratio=0.5, pos_embed_dim=24
                config.update({
                    'dwatt_hidden_dim': best_trial.params.get('dwatt_hidden_dim', None),  # None = paper-faithful
                    'dwatt_bottleneck_ratio': best_trial.params.get('dwatt_bottleneck_ratio', 0.5),
                    'dwatt_pos_embed_dim': best_trial.params.get('dwatt_pos_embed_dim', 24),
                    'dropout': best_trial.params.get('dropout', 0.1),
                })
            elif base_method == "lora":
                config.update({
                    'lora_r': group_key[0],
                    'lora_alpha': best_trial.params.get('lora_alpha', 16),
                    'lora_dropout': best_trial.params.get('lora_dropout', 0.1),
                })
            elif base_method == "deepset":
                # For linear_deepset, derive pre/post layers from variant parameter
                if is_linear:
                    # Linear DeepSet: variant determines architecture
                    variant = best_trial.params.get('variant', 'pre0_post1')
                    if variant == 'pre1_post0':
                        pre_layers, post_layers = 1, 0
                    else:  # pre0_post1
                        pre_layers, post_layers = 0, 1
                    # Pooling type is searchable
                    pooling_type = best_trial.params.get('deepset_pooling_type', 'mean')
                    hidden_dim = 256  # FIXED for linear_deepset
                else:
                    # Non-linear DeepSet: all are searchable hyperparameters
                    pre_layers = best_trial.params.get('deepset_pre_pooling_layers', 0)
                    post_layers = best_trial.params.get('deepset_post_pooling_layers', 1)
                    pooling_type = best_trial.params.get('deepset_pooling_type', 'mean')
                    hidden_dim = best_trial.params.get('deepset_hidden_dim', 256)

                config.update({
                    'deepset_pooling_type': pooling_type,
                    'deepset_pre_pooling_layers': pre_layers,
                    'deepset_post_pooling_layers': post_layers,
                    'deepset_hidden_dim': hidden_dim,
                    'dropout': best_trial.params.get('dropout', 0.1),
                    # Store variant for linear_deepset (used in grouping display)
                    'deepset_variant': best_trial.params.get('variant', None) if is_linear else None,
                })

            configurations.append(config)

    # Sort by task, model, method
    configurations.sort(key=lambda c: (c['task'], c['model_family'], c['model_size'], c['method']))

    return configurations


def generate_slurm_job(config: Dict, output_dir: str, partition_type: str = 'priority', v2: bool = False) -> str:
    """
    Generate a SLURM job file for training with the best configuration.
    Supports GIN/GCN, MLP, Weighted, DeepSet for both classification and STS tasks.
    Also supports linear variants (linear_gin, linear_mlp, linear_deepset).

    Args:
        config: Configuration dict with task, model, and hyperparameters
        output_dir: Directory to save the job file
        partition_type: 'priority' or 'general' for SLURM partition selection
        v2: If True, add _cayley suffix to job names, logs, and model outputs

    Returns:
        Path to generated job file
    """
    method = config['method']
    base_method = config['base_method']  # NEW: method without "linear_" prefix
    is_linear = config['is_linear']      # NEW: linear model flag
    task = config['task']
    is_sts = config.get('is_sts', False)
    model_family = config['model_family']
    model_size = config['model_size']

    # Get cluster configuration
    cluster = get_cluster_type()
    cluster_config = get_cluster_config(partition_type=partition_type)

    # Set time limit based on model size
    if model_size in ["1.1B", "1.1b"]:
        # TinyLlama
        time_limit = "08:00:00"
    elif model_size in ["8B", "8b"]:
        # Llama3-8B
        time_limit = "12:00:00"
    else:
        # Pythia-410m, Pythia-2.8b and others
        time_limit = "10:00:00"

    # Skip LoRA for now
    if base_method == "lora":
        return None

    # Determine embeddings directory and trainer script based on task type
    if is_sts:
        # STS tasks: extract the actual task name (remove "sts_" prefix)
        actual_task = task.replace("sts_", "", 1)  # Remove first occurrence of "sts_"
        embeddings_dir = f"precomputed_embeddings_sts/{model_family}_{model_size}_mean_pooling"
        trainer_script = "experiments.utils.model_definitions.gnn.sts_gin_trainer_precomputed"
        metric_name = "Spearman"
    else:
        # Classification tasks
        actual_task = task
        embeddings_dir = f"precomputed_embeddings/{model_family}_{model_size}_mean_pooling"
        trainer_script = "experiments.utils.model_definitions.gnn.basic_gin_trainer_precomputed"
        metric_name = "Acc"

    # Determine output file naming (use full method name to distinguish linear variants)
    if base_method == "gin":
        graph_type = config['graph_type']
        gin_mlp_layers = config['gin_mlp_layers']
        encoder_method = config['method_type']  # "gin" or "gcn"
        node_to_choose = config['node_to_choose']  # pooling method
        # Include "linear_" prefix in filename for linear variants
        method_prefix = "linear_" if is_linear else ""
        # Add cayley suffix if cayley mode
        v2_suffix = "_cayley" if v2 else ""
        job_filename = f"train_{method_prefix}{encoder_method}_{task}_{model_family}_{model_size}_{graph_type}_{node_to_choose}{v2_suffix}.sh"
        # Config suffix for logs - include linear prefix
        config_suffix = f"{method_prefix}{encoder_method}_{graph_type}_{node_to_choose}{v2_suffix}"
    elif base_method == "mlp":
        mlp_input = config['mlp_input']
        mlp_layers = config['mlp_layers']
        method_prefix = "linear_" if is_linear else ""
        job_filename = f"train_{method_prefix}mlp_{task}_{model_family}_{model_size}_{mlp_input}_layers{mlp_layers}.sh"
        # Config suffix for logs - include linear prefix
        config_suffix = f"{method_prefix}mlp_{mlp_input}_layers{mlp_layers}"
    elif base_method == "weighted":
        job_filename = f"train_weighted_{task}_{model_family}_{model_size}_softmax.sh"
        config_suffix = "weighted_softmax"
    elif base_method == "deepset":
        pooling_type = config['deepset_pooling_type']
        pre_layers = config['deepset_pre_pooling_layers']
        post_layers = config['deepset_post_pooling_layers']
        method_prefix = "linear_" if is_linear else ""
        # Config suffix for logs - include linear prefix
        config_suffix = f"{method_prefix}deepset_{pooling_type}_pre{pre_layers}_post{post_layers}"
        job_filename = f"train_{method_prefix}deepset_{task}_{model_family}_{model_size}_{config_suffix}.sh"
    elif base_method == "dwatt":
        # DWAtt: Depth-Wise Attention (ElNokrashy et al. 2024)
        dwatt_hidden_dim = config.get('dwatt_hidden_dim')
        hidden_suffix = f"_hidden{dwatt_hidden_dim}" if dwatt_hidden_dim else ""
        job_filename = f"train_dwatt_{task}_{model_family}_{model_size}{hidden_suffix}.sh"
        config_suffix = f"dwatt{hidden_suffix}"
    else:
        return None

    job_path = os.path.join(output_dir, job_filename)

    # Build training command based on method
    if base_method == "gin":
        encoder_method = config['method_type']
        node_to_choose = config['node_to_choose']
        linear_label = " (LINEAR)" if is_linear else ""
        method_info = f"{encoder_method.upper()} {graph_type} ({node_to_choose}){linear_label}"

        # Model filename: include "linear_" prefix for linear variants and cayley suffix
        method_prefix = "linear_" if is_linear else ""
        v2_suffix = "_cayley" if v2 else ""
        model_filename = f"{method_prefix}{encoder_method}_{actual_task}_{model_family}_{model_size}_{graph_type}_{node_to_choose}{v2_suffix}.pt"

        # Base training command (same for both STS and classification)
        # Add --use_linear flag for linear variants
        use_linear_flag = " \\\n    --use_linear" if is_linear else ""
        # Add cayley flags if enabled
        pool_real_nodes_flag = " \\\n    --pool_real_nodes_only" if config.get('pool_real_nodes_only', False) else ""
        train_eps_flag = " \\\n    --train_eps" if config.get('train_eps', False) else ""
        train_command = f"""python3 -m {trainer_script} \\
    --task {actual_task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --embeddings_dir {embeddings_dir} \\
    --encoder gin \\
    --gin_hidden_dim {config['gin_hidden_dim']} \\
    --gin_layers {config['gin_layers']} \\
    --dropout {config['dropout']} \\
    --gin_mlp_layers {config['gin_mlp_layers']} \\
    --node_to_choose {config['node_to_choose']} \\
    --graph_type {graph_type} \\
    --epochs 25 \\
    --batch_size {config['batch_size']} \\
    --lr {config['lr']} \\
    --weight_decay {config['weight_decay']} \\
    --save_dir saved_models \\
    --seed 42{use_linear_flag}{pool_real_nodes_flag}{train_eps_flag}"""

    elif base_method == "mlp":
        linear_label = " (LINEAR)" if is_linear else ""
        method_info = f"MLP {mlp_input} layers={mlp_layers}{linear_label}"

        # Model filename: include "linear_" prefix for linear variants
        method_prefix = "linear_" if is_linear else ""
        model_filename = f"{method_prefix}mlp_{actual_task}_{model_family}_{model_size}_{mlp_input}_layers{mlp_layers}.pt"

        # Base training command (same for both STS and classification)
        # Add --use_linear flag for linear variants
        use_linear_flag = " \\\n    --use_linear" if is_linear else ""
        train_command = f"""python3 -m {trainer_script} \\
    --task {actual_task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --embeddings_dir {embeddings_dir} \\
    --encoder mlp \\
    --mlp_input {config['mlp_input']} \\
    --mlp_hidden_dim {config['mlp_hidden_dim']} \\
    --mlp_layers {config['mlp_layers']} \\
    --dropout {config['dropout']} \\
    --epochs 25 \\
    --batch_size {config['batch_size']} \\
    --lr {config['lr']} \\
    --weight_decay {config['weight_decay']} \\
    --save_dir saved_models \\
    --seed 42{use_linear_flag}"""

    elif base_method == "weighted":
        method_info = "Weighted (softmax)"

        # Model filename always includes model_family and model_size (STS and classification)
        model_filename = f"weighted_{actual_task}_{model_family}_{model_size}_softmax.pt"

        # Base training command (same for both STS and classification)
        # Note: Weighted is already linear, no use_linear flag needed
        train_command = f"""python3 -m {trainer_script} \\
    --task {actual_task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --embeddings_dir {embeddings_dir} \\
    --encoder weighted \\
    --epochs 25 \\
    --batch_size {config['batch_size']} \\
    --lr {config['lr']} \\
    --weight_decay {config['weight_decay']} \\
    --save_dir saved_models \\
    --seed 42"""

    elif base_method == "deepset":
        pooling_type = config['deepset_pooling_type']
        pre_layers = config['deepset_pre_pooling_layers']
        post_layers = config['deepset_post_pooling_layers']
        linear_label = " (LINEAR)" if is_linear else ""
        method_info = f"DeepSet ({pooling_type}, pre={pre_layers}, post={post_layers}){linear_label}"

        # Model filename: include "linear_" prefix for linear variants
        method_prefix = "linear_" if is_linear else ""
        model_filename = f"{method_prefix}deepset_{actual_task}_{model_family}_{model_size}_{config_suffix}.pt"

        # Training command
        # Add --use_linear flag for linear variants
        use_linear_flag = " \\\n    --use_linear" if is_linear else ""
        train_command = f"""python3 -m {trainer_script} \\
    --task {actual_task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --embeddings_dir {embeddings_dir} \\
    --encoder deepset \\
    --deepset_hidden_dim {config['deepset_hidden_dim']} \\
    --deepset_pre_pooling_layers {config['deepset_pre_pooling_layers']} \\
    --deepset_post_pooling_layers {config['deepset_post_pooling_layers']} \\
    --deepset_pooling_type {config['deepset_pooling_type']} \\
    --dropout {config['dropout']} \\
    --epochs 25 \\
    --batch_size {config['batch_size']} \\
    --lr {config['lr']} \\
    --weight_decay {config['weight_decay']} \\
    --save_dir saved_models \\
    --seed 42{use_linear_flag}"""

    elif base_method == "dwatt":
        # DWAtt: Depth-Wise Attention (ElNokrashy et al. 2024)
        dwatt_hidden_dim = config.get('dwatt_hidden_dim')
        if dwatt_hidden_dim:
            method_info = f"DWAtt (hidden_dim={dwatt_hidden_dim})"
            hidden_dim_arg = f" \\\n    --dwatt_hidden_dim {dwatt_hidden_dim}"
        else:
            method_info = "DWAtt (paper-faithful)"
            hidden_dim_arg = ""

        # Model filename
        hidden_suffix = f"_hidden{dwatt_hidden_dim}" if dwatt_hidden_dim else ""
        model_filename = f"dwatt_{actual_task}_{model_family}_{model_size}{hidden_suffix}.pt"

        # Training command
        train_command = f"""python3 -m {trainer_script} \\
    --task {actual_task} \\
    --model_family {model_family} \\
    --model_size {model_size} \\
    --embeddings_dir {embeddings_dir} \\
    --encoder dwatt \\
    --dwatt_bottleneck_ratio {config['dwatt_bottleneck_ratio']} \\
    --dwatt_pos_embed_dim {config['dwatt_pos_embed_dim']} \\
    --dropout {config['dropout']} \\
    --epochs 25 \\
    --batch_size {config['batch_size']} \\
    --lr {config['lr']} \\
    --weight_decay {config['weight_decay']} \\
    --save_dir saved_models \\
    --seed 42{hidden_dim_arg}"""

    # SLURM script content (cluster-aware)
    script = f"""#!/bin/bash
#SBATCH --job-name=train_{method}_{task[:20]}
#SBATCH --time={time_limit}
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
{cluster_config.to_sbatch_lines()}
#SBATCH --output={BASE_DIR}/job_logs/train_best/{task}_{model_family}_{model_size}_{config_suffix}.out
#SBATCH --error={BASE_DIR}/job_logs/train_best/{task}_{model_family}_{model_size}_{config_suffix}.err

# Source conda
source "{CONDA_PATH}/etc/profile.d/conda.sh"

# Activate environment
conda activate {CONDA_ENV} || {{
    echo "ERROR: Could not activate {CONDA_ENV} env"
    exit 1
}}

echo "Current conda environment: $CONDA_PREFIX"

# Print GPU info
nvidia-smi

# Go to repo directory
cd "{BASE_DIR}"

echo "=========================================="
echo "Training Best Model"
echo "=========================================="
echo "Task: {task}"
echo "Model: {model_family}-{model_size}"
echo "Method: {method_info}"
echo "Val {metric_name} (from Optuna): {config['val_acc']:.4f}"
echo "Started at: $(date)"
echo "=========================================="

# Train model with best hyperparameters
{train_command}

EXIT_CODE=$?

echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Model saved to: saved_models/{model_filename}"
echo "=========================================="

exit $EXIT_CODE
"""

    with open(job_path, 'w') as f:
        f.write(script)

    # Make executable
    os.chmod(job_path, 0o755)

    return job_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM jobs to train best models from Optuna results"
    )
    parser.add_argument(
        "--storage_url",
        type=str,
        default=None,
        help="PostgreSQL storage URL for Optuna (if not provided, auto-detects via optuna_storage.py)"
    )
    parser.add_argument(
        "--filter_study",
        type=str,
        default=None,
        help="Filter studies by name pattern (e.g., 'gin_', 'Pythia_410m', 'EmotionClassification')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scripts_and_jobs/slurm_jobs/generated_jobs/train_best_models",
        help="Directory to save generated job files"
    )
    parser.add_argument(
        "--partition",
        choices=['priority', 'general'],
        default='priority',
        help="Partition type: 'priority' for lab partition, 'general' for shared (default: priority)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without generating files"
    )
    parser.add_argument(
        "--filter_task",
        type=str,
        default=None,
        help="Only generate jobs for specific task (e.g., EmotionClassification)"
    )
    parser.add_argument(
        "--filter_model",
        type=str,
        default=None,
        help="Only generate jobs for specific model (e.g., Pythia_410m)"
    )
    parser.add_argument(
        "--filter_graph_type",
        type=str,
        choices=["linear", "virtual_node", "cayley", "cayley", "fully_connected"],
        default=None,
        help="Only generate jobs for specific graph_type (e.g., 'cayley'). Applies to 'gin' method only."
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default=None,
        help="Use hyperparameters from this model (e.g., Pythia_410m) instead of --filter_model. For transfer learning."
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use cayley studies (_cayley_precomputed suffix). Default: baseline studies (_precomputed, excluding cayley)"
    )

    args = parser.parse_args()

    # Auto-detect storage URL if not provided
    if args.storage_url is None:
        import subprocess
        try:
            result = subprocess.run(
                ["python3", "scripts_and_jobs/scripts/optuna_storage.py"],
                capture_output=True,
                text=True,
                check=True
            )
            args.storage_url = result.stdout.strip()
        except Exception as e:
            print(f"ERROR: Could not auto-detect Optuna storage URL: {e}")
            print("Please provide --storage_url manually or ensure optuna_storage.py works")
            sys.exit(1)

    print("="*70)
    print("Generate SLURM Jobs for Best Models from Optuna")
    print("="*70)

    # Determine which model to query for hyperparameters
    query_model = args.source_model if args.source_model else args.filter_model
    target_model = args.filter_model

    if args.source_model and args.filter_model:
        print(f"\n🔄 Transfer Learning Mode:")
        print(f"  Querying hyperparameters from: {args.source_model}")
        print(f"  Training models for: {args.filter_model}")

    # Query Optuna database
    configs = get_best_configurations(args.storage_url, args.filter_study, v2=args.v2)

    # Apply filters
    # Default: only the 5 standard classification tasks (unless explicitly filtered)
    DEFAULT_CLASSIFICATION_TASKS = [
        "EmotionClassification",
        "Banking77Classification",
        "MTOPIntentClassification",
        "MTOPDomainClassification",
        "PoemSentimentClassification",
    ]

    if not args.filter_task and not args.filter_study:
        # No explicit filter: use default classification tasks only
        before_count = len(configs)
        configs = [c for c in configs if c['task'] in DEFAULT_CLASSIFICATION_TASKS]
        excluded_count = before_count - len(configs)
        if excluded_count > 0:
            print(f"\nFiltered to {len(configs)} default classification task configurations")
            print(f"  Excluded: {excluded_count} other tasks (STS, AmazonCounterfactual, Massive, etc.)")
            print(f"  (Use --filter_task or --filter_study to include other tasks)")

    if args.filter_task:
        configs = [c for c in configs if c['task'] == args.filter_task]
        print(f"\nFiltered to task: {args.filter_task}")

    # Use query_model (source) for filtering Optuna results
    if query_model:
        model_family, model_size = query_model.split("_")
        configs = [c for c in configs if c['model_family'] == model_family and c['model_size'] == model_size]
        if args.source_model:
            print(f"\nFiltered to source model hyperparameters: {query_model}")
        else:
            print(f"\nFiltered to model: {query_model}")

    # If using transfer learning, update configs to use target model
    if args.source_model and target_model:
        target_family, target_size = target_model.split("_")
        for config in configs:
            config['model_family'] = target_family
            config['model_size'] = target_size

    if args.filter_graph_type:
        # First, try to find configs from studies with graph_type in name (new format)
        # These are studies like: gin_cayley_{task}_{model}_{size}
        graph_type_in_name_configs = [
            c for c in configs 
            if c.get('graph_type') == args.filter_graph_type 
            and args.filter_graph_type in c.get('study_name', '')
        ]
        
        if graph_type_in_name_configs:
            configs = graph_type_in_name_configs
            print(f"\nFound {len(configs)} configurations from studies with '{args.filter_graph_type}' in name")
        else:
            # Fallback: filter by trial params from old-format studies
            # These are studies like: gin_{task}_{model}_{size} with graph_type in trial params
            configs = [c for c in configs if c.get('graph_type') == args.filter_graph_type]
            if configs:
                print(f"\nFound {len(configs)} configurations from old-format studies (filtered by trial params)")
            else:
                print(f"\n⚠️ No configurations found for graph_type '{args.filter_graph_type}'")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Found {len(configs)} configurations to train")
    print(f"{'='*70}")

    # Group by model for better display
    current_model = None
    for config in configs:
        model_key = f"{config['model_family']}-{config['model_size']}"
        if model_key != current_model:
            current_model = model_key
            print(f"\n{model_key}:")

        is_linear_display = config.get('is_linear', False)
        method = config['method'].upper()

        if config['method'] == 'gin' or config['method'] == 'linear_gin':
            method = config.get('method_type', 'gin').upper()
            if is_linear_display:
                method = f"LINEAR_{method}"
            detail = config['graph_type']
        elif config['method'] == 'mlp' or config['method'] == 'linear_mlp':
            if is_linear_display:
                method = "LINEAR_MLP"
            detail = f"{config['mlp_input']}_layers{config['mlp_layers']}"
        elif config['method'] == 'lora':
            detail = f"r={config['lora_r']}"
        elif config['method'] == 'deepset' or config['method'] == 'linear_deepset':
            if is_linear_display:
                method = "LINEAR_DEEPSET"
                # For linear deepset, show variant if available
                variant = config.get('deepset_variant')
                if variant:
                    detail = f"variant={variant}"
                else:
                    pooling = config['deepset_pooling_type']
                    pre = config['deepset_pre_pooling_layers']
                    post = config['deepset_post_pooling_layers']
                    detail = f"{pooling}_pre{pre}_post{post}"
            else:
                pooling = config['deepset_pooling_type']
                pre = config['deepset_pre_pooling_layers']
                post = config['deepset_post_pooling_layers']
                detail = f"{pooling}_pre{pre}_post{post}"
        elif config['method'] == 'dwatt':
            method = "DWATT"
            dwatt_hidden_dim = config.get('dwatt_hidden_dim')
            if dwatt_hidden_dim:
                detail = f"hidden={dwatt_hidden_dim}"
            else:
                detail = "paper-faithful"
        else:
            detail = ""

        print(f"  {config['task']:45s} | {method} {detail:20s} | val_acc={config['val_acc']:.4f}")

    # Generate job files or just print
    if args.dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - No files generated")
        print(f"{'='*70}")
        print(f"\nWould generate {len(configs)} job files in:")
        print(f"  {args.output_dir}")
    else:
        print(f"\n{'='*70}")
        print("Generating job files...")
        print(f"{'='*70}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Create log directory
        os.makedirs("job_logs/train_best", exist_ok=True)

        # Generate each job file
        generated_count = 0
        skipped_count = 0
        for config in configs:
            job_file = generate_slurm_job(config, args.output_dir, partition_type=args.partition, v2=args.v2)
            if job_file:
                generated_count += 1
                print(f"  ✓ {os.path.basename(job_file)}")
            else:
                skipped_count += 1

        if skipped_count > 0:
            print(f"\n  ⚠️ Skipped {skipped_count} studies (LoRA support coming soon)")

        print(f"\n{'='*70}")
        print("Job files generated successfully!")
        print(f"{'='*70}")
        print(f"\nTo submit all jobs:")
        print(f"  cd {BASE_DIR}")
        print(f"  for job in {args.output_dir}/*.sh; do sbatch \"$job\"; done")
        print(f"\nTo submit selectively:")
        print(f"  sbatch {args.output_dir}/train_EmotionClassification_*.sh")
        print(f"\nTo check status:")
        print(f"  squeue -u $USER | grep train")


if __name__ == "__main__":
    main()

