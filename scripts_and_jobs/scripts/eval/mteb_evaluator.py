#!/usr/bin/env python3
"""
MTEB evaluation script for trained GNN/MLP/Weighted/DeepSet/DWAtt models.
Uses the appropriate wrapper based on model type.
Supports both MTEB tasks and custom tasks.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import mteb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications
from scripts_and_jobs.scripts.eval.gnn_wrapper import GNNWrapper
from scripts_and_jobs.scripts.eval.mlp_wrapper import MLPWrapper
from scripts_and_jobs.scripts.eval.weighted_wrapper import WeightedWrapper
from scripts_and_jobs.scripts.eval.deepset_wrapper import DeepSetWrapper
from scripts_and_jobs.scripts.eval.dwatt_wrapper import DWAttWrapper
from scripts_and_jobs.scripts.eval.lora_wrapper import LoRAWrapper
from scripts_and_jobs.scripts.eval.custom_task_evaluator import evaluate_custom_tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on MTEB tasks")
    
    # Model configuration
    parser.add_argument("--model_family", type=str, default="Pythia", 
                       help="Base model family")
    parser.add_argument("--model_size", type=str, default="14m",
                       help="Base model size")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint (GNN or MLP)")
    parser.add_argument("--model_type", type=str, default="auto",
                       choices=["gin", "gcn", "mlp", "weighted", "deepset", "dwatt", "lora", "auto"],
                       help="Model type (auto-detects from filename if not specified)")
    
    # MTEB evaluation
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=None,
                       help="MTEB tasks to evaluate on")
    parser.add_argument("--custom_tasks", type=str, nargs="+", default=None,
                       help="Custom task names (not in MTEB)")  # <-- ADDED
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for encoding")
    parser.add_argument("--hf_subsets", type=str, nargs="+", default=None,
                       help="HuggingFace subsets to evaluate (e.g., 'en' for English only)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./mteb_results_clean",
                       help="Directory to save evaluation results")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping for base model")
    
    args = parser.parse_args()
    
    # Auto-detect model type from filename if not specified
    if args.model_type == "auto":
        filename = os.path.basename(args.model_path)
        if filename.startswith("gin_") or filename.startswith("gcn_") or "_gin" in filename or "_gcn" in filename:
            args.model_type = "gnn"  # Unified GNN type
        elif filename.startswith("dwatt_") or "_dwatt" in filename:
            args.model_type = "dwatt"
        elif filename.startswith("mlp_") or "_mlp" in filename:
            args.model_type = "mlp"
        elif filename.startswith("weighted_") or "_weighted" in filename:
            args.model_type = "weighted"
        elif filename.startswith("deepset_") or "_deepset" in filename:
            args.model_type = "deepset"
        elif filename.startswith("lora_") or "_lora" in filename:
            args.model_type = "lora"
        else:
            print(f"Error: Cannot auto-detect model type from {args.model_path}")
            print("Please specify --model_type gin, --model_type gcn, --model_type mlp, --model_type weighted, --model_type deepset, --model_type dwatt, or --model_type lora")
            return
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    print("="*60)
    print(f"MTEB {args.model_type.upper()} Model Evaluation")
    print("="*60)
    print(f"Base Model: {args.model_family}-{args.model_size}")
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type.upper()}")
    print(f"Tasks: {args.tasks}")
    if args.custom_tasks:  # <-- ADDED
        print(f"Custom Tasks: {args.custom_tasks}")  # <-- ADDED
    print(f"Output Dir: {args.output_dir}")
    print("="*60)

    results = None
    
    # Initialize the appropriate wrapper
    try:
        model_specs = TextModelSpecifications(
            args.model_family, args.model_size, "main", ignore_checks=True
        )
        
        if args.model_type in ["gnn", "gin", "gcn"]:
            # Use GNN wrapper - no classifier overhead!
            model = GNNWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )
            
            # Load the GNN encoder directly
            model.load_model(args.model_path)
            print(f"✓ {args.model_type.upper()} encoder loaded successfully")
            
        elif args.model_type == "mlp":
            # Use MLP wrapper - no classifier overhead!
            model = MLPWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )

            model.load_model(args.model_path)
            print("✓ MLP encoder loaded successfully")

        elif args.model_type == "weighted":
            # Use Weighted wrapper - no classifier overhead!
            model = WeightedWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )

            model.load_model(args.model_path)
            print("✓ Weighted encoder loaded successfully")

        elif args.model_type == "deepset":
            # Use DeepSet wrapper - no classifier overhead!
            model = DeepSetWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )

            model.load_model(args.model_path)
            print("✓ DeepSet encoder loaded successfully")

        elif args.model_type == "dwatt":
            # Use DWAtt wrapper - Depth-Wise Attention baseline
            model = DWAttWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )

            model.load_model(args.model_path)
            print("✓ DWAtt encoder loaded successfully")

        elif args.model_type == "lora":
            # Use LoRA wrapper - PEFT-finetuned base model baseline
            model = LoRAWrapper(
                model_specs=model_specs,
                device_map=args.device_map,
                model_path=args.model_path
            )

            model.load_model(args.model_path)
            print("✓ LoRA adapter loaded successfully")

    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get MTEB tasks - filter for English only    
    try:        
        if args.tasks:
            tasks = mteb.get_tasks(tasks=args.tasks, languages=["eng"])
            print(f"✓ Loaded {len(tasks)} MTEB tasks (English only)")

            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print("\nStarting MTEB evaluation...")
            evaluation = mteb.MTEB(tasks=tasks)
            
            # Run evaluation - both wrappers have encode() methods
            run_kwargs = {
                "model": model,
                "output_folder": str(output_dir),
                "eval_splits": ["test"]
            }
            
            # Add hf_subsets if specified
            if args.hf_subsets:
                run_kwargs["hf_subsets"] = args.hf_subsets
                
            results = evaluation.run(**run_kwargs)
            
            print("\n" + "="*60)
            print("✓ MTEB evaluation completed successfully!")
            print(f"Results saved to: {output_dir}")
        
        else:
            print(f"No regular MTEB tasks\n")

    except Exception as e:
        print(f"✗ Error loading and evaluating MTEB tasks: {e}")
        print(f"tasks: {args.tasks}")
        return
    
            
    # ========== EVALUATE CUSTOM TASKS (ADDED) ==========
    if args.custom_tasks:
        try:
            custom_results = evaluate_custom_tasks(
                model=model,
                custom_task_names=args.custom_tasks,
                batch_size=args.batch_size
            )
            
            # Merge custom results with MTEB results
            if custom_results:
                if results:
                    results.update(custom_results)
                else:
                    results = custom_results
                    
        except Exception as e:
            print(f"✗ Error during custom task evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"No custom MTEB tasks\n")
    # ========== END CUSTOM TASKS ==========
    
    try:
        # Print summary results
        if results:
            print("\nSummary Results:")
            if isinstance(results, list):
                print(f"  Evaluation completed successfully with {len(results)} result(s)")
                for i, result in enumerate(results):
                    print(f"  Result {i+1}: {result}")
            else:
                for task_name, task_results in results.items():
                    if isinstance(task_results, dict) and "test" in task_results:
                        test_results = task_results["test"]
                        if isinstance(test_results, dict):
                            # Find the main metric
                            main_metrics = ["accuracy", "f1", "main_score", "spearman", "cosine_spearman"]
                            for metric in main_metrics:
                                if metric in test_results:
                                    print(f"  {task_name}: {metric}={test_results[metric]:.4f}")
                                    break
                            else:
                                # Print first available metric
                                if test_results:
                                    first_metric = list(test_results.keys())[0]
                                    print(f"  {task_name}: {first_metric}={test_results[first_metric]:.4f}")
    except Exception as e:
        print(f"✗ Error during printing results: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)


if __name__ == "__main__":
    main()