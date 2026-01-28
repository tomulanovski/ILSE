#!/usr/bin/env python3
"""
Evaluate a trained STS model using MTEB.

Usage:
    python3 -m scripts_and_jobs.scripts.eval.evaluate_sts_model \
        --model_path saved_models/gin_STSBenchmark_Pythia_410m_cayley.pt \
        --model_family Pythia \
        --model_size 410m \
        --encoder gin \
        --tasks STSBenchmark SICK-R
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import mteb
from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications
from scripts_and_jobs.scripts.eval.sts_gin_wrapper import STSGINWrapper
from scripts_and_jobs.scripts.eval.sts_mlp_wrapper import STSMLPWrapper
from scripts_and_jobs.scripts.eval.sts_weighted_wrapper import STSWeightedWrapper
from scripts_and_jobs.scripts.eval.sts_deepset_wrapper import STSDeepSetWrapper
from scripts_and_jobs.scripts.eval.sts_dwatt_wrapper import STSDWAttWrapper


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained STS model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (Pythia, TinyLlama, etc.)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (410m, 1.1B, etc.)")
    parser.add_argument("--encoder", type=str, required=True,
                        help="Encoder type (gin/gcn/mlp/weighted/deepset/dwatt/linear_gin/linear_gcn/linear_mlp/linear_deepset)")
    parser.add_argument("--config", type=str, required=True,
                        help="Model configuration (e.g., cayley, linear, last_layers2, softmax)")
    parser.add_argument("--tasks", type=str, nargs="+", default=["STSBenchmark"],
                        help="STS tasks to evaluate on")
    parser.add_argument("--output_dir", type=str, default="results/sts_results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n{'='*70}")
    print(f"STS Model Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model_family}-{args.model_size}")
    print(f"Encoder: {args.encoder}")
    print(f"Path: {model_path}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"{'='*70}\n")

    # Create model specifications
    model_specs = TextModelSpecifications(
        model_family=args.model_family,
        model_size=args.model_size,
        revision="main"
    )

    # Load appropriate wrapper based on encoder type
    # Strip "linear_" prefix if present (linear models use same wrappers, just different checkpoints)
    base_encoder = args.encoder.replace("linear_", "") if args.encoder.startswith("linear_") else args.encoder

    print(f"Loading {args.encoder.upper()} wrapper...")
    if base_encoder in ["gin", "gcn"]:
        # Both GIN and GCN use the same wrapper (GCN is just GIN with gin_mlp_layers=0)
        wrapper = STSGINWrapper(
            model_specs=model_specs,
            model_path=str(model_path),
            device_map="auto"
        )
    elif base_encoder == "mlp":
        wrapper = STSMLPWrapper(
            model_specs=model_specs,
            model_path=str(model_path),
            device_map="auto"
        )
    elif base_encoder == "weighted":
        wrapper = STSWeightedWrapper(
            model_specs=model_specs,
            model_path=str(model_path),
            device_map="auto"
        )
    elif base_encoder == "deepset":
        wrapper = STSDeepSetWrapper(
            model_specs=model_specs,
            model_path=str(model_path),
            device_map="auto"
        )
    elif base_encoder == "dwatt":
        wrapper = STSDWAttWrapper(
            model_specs=model_specs,
            model_path=str(model_path),
            device_map="auto"
        )
    else:
        raise ValueError(f"Unsupported encoder: {args.encoder}")

    # Clear GPU cache after loading wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✓ Wrapper loaded successfully\n")

    # Load tasks
    print(f"Loading MTEB tasks...")
    tasks = [mteb.get_task(task_name) for task_name in args.tasks]
    print(f"✓ Loaded {len(tasks)} task(s)\n")

    # Create output directory - include config to separate different graph_types/configs
    output_dir = Path(args.output_dir) / args.model_family / args.model_size / "main" / "mteb" / args.encoder / args.config
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}\n")

    # Run evaluation
    print(f"{'='*70}")
    print(f"Starting Evaluation")
    print(f"{'='*70}\n")

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        wrapper,
        output_folder=str(output_dir),
        batch_size=args.batch_size,
        overwrite_results=False
    )

    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"{'='*70}\n")

    # Print summary
    for task_name in args.tasks:
        result_file = output_dir / f"{task_name}.json"
        if result_file.exists():
            import json
            with open(result_file, 'r') as f:
                data = json.load(f)

            # STS tasks use Spearman correlation
            if 'test' in data.get('scores', {}):
                spearman = data['scores']['test'][0].get('cos_sim', {}).get('spearman', 'N/A')
                print(f"  {task_name}: Spearman = {spearman:.4f}" if isinstance(spearman, float) else f"  {task_name}: {spearman}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
