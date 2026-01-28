#!/usr/bin/env python3
"""
Precompute and cache layer-wise embeddings for STS tasks.

For STS tasks, we have sentence PAIRS (text_a, text_b) with similarity scores.
This script precomputes embeddings for both sentences in each pair and saves to HDF5.

Usage:
    python3 precompute_sts.py \
        --task STSBenchmark \
        --model_family Pythia \
        --model_size 410m \
        --output_dir ./precomputed_embeddings_sts
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from ..model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from ..model_definitions.gnn.gnn_datasets import load_task_data, compute_layerwise


def save_sts_embeddings_to_h5(
    output_path: Path,
    train_lw_a: list,
    train_lw_b: list,
    train_scores: list,
    val_lw_a: list,
    val_lw_b: list,
    val_scores: list,
    test_lw_a: list = None,
    test_lw_b: list = None,
    test_scores: list = None,
):
    """
    Save STS precomputed embeddings to HDF5 file.

    Args:
        output_path: Path to output HDF5 file
        train_lw_a: List of train sentence A embeddings [N_train, L, D]
        train_lw_b: List of train sentence B embeddings [N_train, L, D]
        train_scores: List of train similarity scores [N_train]
        val_lw_a: List of val sentence A embeddings
        val_lw_b: List of val sentence B embeddings
        val_scores: List of val similarity scores
        test_lw_a: Optional test sentence A embeddings
        test_lw_b: Optional test sentence B embeddings
        test_scores: Optional test similarity scores
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving precomputed STS embeddings to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # Save train split
        f.create_dataset('train/embeddings_a', data=np.array(train_lw_a, dtype=np.float32), compression='gzip')
        f.create_dataset('train/embeddings_b', data=np.array(train_lw_b, dtype=np.float32), compression='gzip')
        f.create_dataset('train/scores', data=np.array(train_scores, dtype=np.float32), compression='gzip')

        # Save validation split
        f.create_dataset('validation/embeddings_a', data=np.array(val_lw_a, dtype=np.float32), compression='gzip')
        f.create_dataset('validation/embeddings_b', data=np.array(val_lw_b, dtype=np.float32), compression='gzip')
        f.create_dataset('validation/scores', data=np.array(val_scores, dtype=np.float32), compression='gzip')

        # Save test split if provided
        if test_lw_a is not None:
            f.create_dataset('test/embeddings_a', data=np.array(test_lw_a, dtype=np.float32), compression='gzip')
            f.create_dataset('test/embeddings_b', data=np.array(test_lw_b, dtype=np.float32), compression='gzip')
            f.create_dataset('test/scores', data=np.array(test_scores, dtype=np.float32), compression='gzip')

        # Save metadata
        L, D = train_lw_a[0].shape
        f.attrs['num_layers'] = L
        f.attrs['layer_dim'] = D
        f.attrs['num_train'] = len(train_lw_a)
        f.attrs['num_val'] = len(val_lw_a)
        f.attrs['num_test'] = len(test_lw_a) if test_lw_a is not None else 0

    print(f"✓ Saved {len(train_lw_a)} train, {len(val_lw_a)} val pairs")
    if test_lw_a is not None:
        print(f"✓ Saved {len(test_lw_a)} test pairs")
    print(f"✓ Dimensions: {L} layers × {D} features")
    print(f"✓ File size: {output_path.stat().st_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Precompute STS embeddings")
    parser.add_argument("--task", type=str, required=True, help="STS task name (e.g., STSBenchmark, SICK-R)")
    parser.add_argument("--model_family", type=str, required=True, help="Model family (Pythia, TinyLlama, Llama3)")
    parser.add_argument("--model_size", type=str, required=True, help="Model size (410m, 1.1B, 8B)")
    parser.add_argument("--output_dir", type=str, default="./precomputed_embeddings_sts",
                        help="Directory to save precomputed embeddings")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding extraction")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Precomputing STS embeddings")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_family}-{args.model_size}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)
    print(f"✓ Loaded {args.model_family}-{args.model_size}")

    # Adjust batch size for large models to prevent OOM
    original_batch_size = args.batch_size
    if args.model_family == "Llama3" and "8B" in args.model_size:
        # Llama3-8B is huge (~28 GB), use very small batch size
        args.batch_size = min(args.batch_size, 4)
        print(f"⚠️  Large model detected! Reducing batch size: {original_batch_size} → {args.batch_size}")
    elif args.model_family == "TinyLlama" or (args.model_family == "Pythia" and "2.8b" in args.model_size):
        # Medium models, use moderate batch size
        args.batch_size = min(args.batch_size, 32)
        if args.batch_size != original_batch_size:
            print(f"⚠️  Adjusting batch size for model: {original_batch_size} → {args.batch_size}")

    # Check available splits first
    print(f"\nChecking available splits for {args.task}...")
    try:
        from mteb import MTEB
        mteb_task = MTEB(tasks=[args.task]).tasks[0]
        available_splits = list(mteb_task.dataset.keys()) if hasattr(mteb_task.dataset, 'keys') else []

        # Try to get splits from the actual dataset
        if not available_splits:
            test_data_check = load_task_data(args.task, "test")
            available_splits = ["test"]
            try:
                train_data_check = load_task_data(args.task, "train")
                available_splits.insert(0, "train")
            except:
                pass
            try:
                val_data_check = load_task_data(args.task, "validation")
                if "validation" not in available_splits:
                    available_splits.insert(-1, "validation")
            except:
                pass
    except:
        # Fallback: try loading and see what happens
        available_splits = []

    print(f"Available splits: {available_splits}")

    # Check if task has train split (required for training)
    has_train = False
    try:
        train_data_test = load_task_data(args.task, "train")
        has_train = True
    except Exception as e:
        pass

    if not has_train:
        print(f"\n⚠️  ERROR: Task '{args.task}' does not have a 'train' split!")
        print(f"This task is evaluation-only (test split only).")
        print(f"Cannot train models on this task - skipping precompute.")
        print(f"\nAvailable splits: {available_splits}")
        print(f"\nFor training, use tasks with train splits like:")
        print(f"  - STSBenchmark")
        print(f"  - STS12, STS13, STS14, STS15, STS16, STS17")
        return

    # Load task data
    print(f"\nLoading task data: {args.task}")
    train_data = load_task_data(args.task, "train")

    try:
        val_data = load_task_data(args.task, "validation")
    except Exception as e:
        print(f"Warning: No validation split found: {e}")
        print("Using 10% of train as validation...")
        # Split train data
        n_train = len(train_data["text_a"])
        n_val = max(1, n_train // 10)

        val_data = {
            "text_a": train_data["text_a"][:n_val],
            "text_b": train_data["text_b"][:n_val],
            "scores": train_data["scores"][:n_val],
        }
        train_data = {
            "text_a": train_data["text_a"][n_val:],
            "text_b": train_data["text_b"][n_val:],
            "scores": train_data["scores"][n_val:],
        }

    # Try to load test data (optional)
    try:
        test_data = load_task_data(args.task, "test")
        has_test = True
    except Exception:
        print("No test split available (will be evaluated via MTEB)")
        test_data = None
        has_test = False

    print(f"✓ Loaded {len(train_data['text_a'])} train pairs")
    print(f"✓ Loaded {len(val_data['text_a'])} val pairs")
    if has_test:
        print(f"✓ Loaded {len(test_data['text_a'])} test pairs")

    # Extract embeddings
    print(f"\nExtracting layer-wise embeddings (batch_size={args.batch_size})...")

    import torch

    print("  Train sentence A...")
    train_lw_a = compute_layerwise(wrapper, train_data["text_a"], batch_size=args.batch_size, token_pooling_method="mean")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  Train sentence B...")
    train_lw_b = compute_layerwise(wrapper, train_data["text_b"], batch_size=args.batch_size, token_pooling_method="mean")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  Val sentence A...")
    val_lw_a = compute_layerwise(wrapper, val_data["text_a"], batch_size=args.batch_size, token_pooling_method="mean")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("  Val sentence B...")
    val_lw_b = compute_layerwise(wrapper, val_data["text_b"], batch_size=args.batch_size, token_pooling_method="mean")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if has_test:
        print("  Test sentence A...")
        test_lw_a = compute_layerwise(wrapper, test_data["text_a"], batch_size=args.batch_size, token_pooling_method="mean")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  Test sentence B...")
        test_lw_b = compute_layerwise(wrapper, test_data["text_b"], batch_size=args.batch_size, token_pooling_method="mean")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        test_lw_a = None
        test_lw_b = None
        test_data = {"scores": None}

    print("✓ Embedding extraction complete")

    # Save to HDF5
    output_dir = Path(args.output_dir)
    model_dir = output_dir / f"{args.model_family}_{args.model_size}_mean_pooling"
    output_path = model_dir / f"{args.task}.h5"

    save_sts_embeddings_to_h5(
        output_path=output_path,
        train_lw_a=train_lw_a,
        train_lw_b=train_lw_b,
        train_scores=train_data["original_scores"],  # Use ORIGINAL scores [0, 5], not normalized [0, 1]
        val_lw_a=val_lw_a,
        val_lw_b=val_lw_b,
        val_scores=val_data["original_scores"],  # Use ORIGINAL scores
        test_lw_a=test_lw_a,
        test_lw_b=test_lw_b,
        test_scores=test_data["original_scores"] if has_test else None,  # Use ORIGINAL scores
    )

    print(f"\n{'='*70}")
    print(f"✓ Precomputation complete!")
    print(f"{'='*70}\n")

    print("To use these precomputed embeddings:")
    print(f"  python3 -m experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_sts_gin_precomputed \\")
    print(f"    --task {args.task} \\")
    print(f"    --model_family {args.model_family} \\")
    print(f"    --model_size {args.model_size} \\")
    print(f"    --embeddings_dir {model_dir} \\")
    print(f"    --study_name \"gin_sts_{args.task}_{args.model_family}_{args.model_size}\" \\")
    print(f"    --storage_url \"<your-storage-url>\" \\")
    print(f"    --n_trials 100")
    print()


if __name__ == "__main__":
    main()
