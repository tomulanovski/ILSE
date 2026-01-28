#!/usr/bin/env python3
"""
Pipeline for precomputing layer-wise embeddings for multiple tasks.
Extracts embeddings once and saves to HDF5 for fast GNN training/evaluation.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import gc
from datetime import datetime

from ..model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from ..model_definitions.gnn.gnn_datasets import load_task_data, compute_layerwise
from .h5_utils import save_embeddings_to_h5, validate_h5_file, get_h5_metadata, ChunkedH5Writer


def get_output_dir(
    base_dir: str,
    model_family: str,
    model_size: str,
    pooling_method: str
) -> Path:
    """Construct output directory path."""
    dir_name = f"{model_family}_{model_size}_{pooling_method}_pooling"
    return Path(base_dir) / dir_name


def extract_and_save_task_split(
    wrapper: TextLayerwiseAutoModelWrapper,
    task_name: str,
    split_name: str,
    output_dir: Path,
    pooling_method: str = "mean",
    batch_size: int = 256,
    force_recompute: bool = False,
    save_texts: bool = True,
    chunk_size: int = 5000
) -> bool:
    """
    Extract layer-wise embeddings for a single task split and save to H5.
    Processes data in chunks to avoid OOM errors.

    Args:
        wrapper: Pre-loaded model wrapper
        task_name: Name of the task
        split_name: 'train', 'validation', or 'test'
        output_dir: Directory to save embeddings
        pooling_method: Token pooling method
        batch_size: Batch size for extraction
        force_recompute: If True, recompute even if file exists
        save_texts: If True, save original texts for traceability
        chunk_size: Number of samples to process at once (to avoid OOM)

    Returns:
        True if successful, False otherwise
    """
    # Create task directory
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Output file path
    h5_path = task_dir / f"{split_name}.h5"

    # Check if already computed
    expected_metadata = {'pooling_method': pooling_method}
    if not force_recompute and validate_h5_file(h5_path, expected_metadata):
        metadata = get_h5_metadata(h5_path)
        print(f"  ✓ {split_name}: Already exists ({metadata['num_samples']} samples)")
        return True

    # Load task data
    try:
        data = load_task_data(task_name, split_name)
    except Exception as e:
        print(f"  ✗ {split_name}: Failed to load data - {e}")
        return False

    texts = data["text"]
    labels = np.array(data["labels"])
    num_classes = data.get("num_classes", len(set(labels)))
    num_samples = len(texts)

    print(f"  → {split_name}: Extracting {num_samples} samples in chunks of {chunk_size}...")

    # Process first chunk to get dimensions
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    first_chunk_size = min(chunk_size, num_samples)
    first_chunk_texts = texts[:first_chunk_size]

    try:
        print(f"    Chunk 1/{num_chunks}: samples 0-{first_chunk_size}...")
        first_chunk_embeddings = compute_layerwise(
            wrapper,
            first_chunk_texts,
            batch_size=batch_size,
            token_pooling_method=pooling_method
        )
        first_chunk_array = np.stack(first_chunk_embeddings, axis=0)  # (chunk_size, L, D)

        # Get dimensions
        _, num_layers, hidden_dim = first_chunk_array.shape

        # Prepare metadata
        metadata = {
            'task_name': task_name,
            'split_name': split_name,
            'pooling_method': pooling_method,
            'num_classes': num_classes,
            'model_family': wrapper.model_specs.model_family,
            'model_size': wrapper.model_specs.model_size,
            'extraction_timestamp': datetime.now().isoformat(),
        }

        # Create chunked writer (writes directly to disk, no RAM accumulation)
        print(f"  → Writing to HDF5 in chunks (avoids RAM accumulation)...")
        with ChunkedH5Writer(
            save_path=h5_path,
            total_samples=num_samples,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            labels=labels,
            texts=texts if save_texts else None,
            metadata=metadata
        ) as writer:
            # Write first chunk
            writer.write_chunk(first_chunk_array)

            # Free first chunk memory
            del first_chunk_embeddings
            del first_chunk_array
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process and write remaining chunks
            for chunk_idx in range(1, num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, num_samples)
                chunk_texts = texts[start_idx:end_idx]

                print(f"    Chunk {chunk_idx + 1}/{num_chunks}: samples {start_idx}-{end_idx}...")

                # Extract embeddings for this chunk
                chunk_embeddings = compute_layerwise(
                    wrapper,
                    chunk_texts,
                    batch_size=batch_size,
                    token_pooling_method=pooling_method
                )

                # Convert to numpy and write directly to HDF5
                chunk_array = np.stack(chunk_embeddings, axis=0)
                writer.write_chunk(chunk_array)

                # Free memory immediately (chunk never accumulates in RAM)
                del chunk_embeddings
                del chunk_array
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"  ✓ {split_name}: Saved successfully (chunked writing, no RAM accumulation)")

        # Free remaining data
        del texts
        del labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"  ✗ {split_name}: Extraction/save failed - {e}")
        # Cleanup on failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


def save_pipeline_manifest(
    output_dir: Path,
    model_specs: TextModelSpecifications,
    tasks: List[str],
    pooling_method: str,
    results: Dict[str, Dict[str, bool]]
):
    """Save manifest file with pipeline run information."""
    manifest = {
        'model_family': model_specs.model_family,
        'model_size': model_specs.model_size,
        'revision': model_specs.revision,
        'pooling_method': pooling_method,
        'tasks': tasks,
        'extraction_results': results,
        'timestamp': datetime.now().isoformat(),
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")


def split_train_for_validation(
    task_name: str,
    output_dir: Path,
    pooling_method: str,
    val_ratio: float = 0.15
) -> bool:
    """
    Split existing train.h5 into new train.h5 (85%) and validation.h5 (15%).

    Args:
        task_name: Name of the task
        output_dir: Base output directory
        pooling_method: Pooling method used
        val_ratio: Fraction of data to use for validation (default: 0.15)

    Returns:
        True if successful, False otherwise
    """
    from sklearn.model_selection import train_test_split
    from .h5_utils import load_embeddings_from_h5, save_embeddings_to_h5

    task_dir = output_dir / task_name
    train_h5 = task_dir / "train.h5"
    val_h5 = task_dir / "validation.h5"

    if not train_h5.exists():
        print(f"  ✗ Cannot split: train.h5 not found")
        return False

    try:
        # Load full training data
        print(f"  → Loading train.h5...")
        embeddings, labels, metadata = load_embeddings_from_h5(train_h5, load_metadata=True)

        N, L, D = embeddings.shape
        print(f"    Loaded: {N} samples, {L} layers, {D} dim")

        # Split indices
        indices = np.arange(N)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=42,
            stratify=labels  # Preserve class distribution
        )

        print(f"    Train: {len(train_indices)} samples ({100*(1-val_ratio):.0f}%)")
        print(f"    Val: {len(val_indices)} samples ({100*val_ratio:.0f}%)")

        # Split data
        train_emb = embeddings[train_indices]
        train_labels = labels[train_indices]
        val_emb = embeddings[val_indices]
        val_labels = labels[val_indices]

        # Update metadata
        train_metadata = metadata.copy()
        train_metadata['split_name'] = 'train'
        train_metadata['extraction_timestamp'] = datetime.now().isoformat()
        train_metadata['note'] = f'Split from original train ({100*(1-val_ratio):.0f}%)'

        val_metadata = metadata.copy()
        val_metadata['split_name'] = 'validation'
        val_metadata['extraction_timestamp'] = datetime.now().isoformat()
        val_metadata['note'] = f'Split from original train ({100*val_ratio:.0f}%)'

        # Save new train.h5 (overwrite)
        print(f"  → Saving new train.h5 ({len(train_indices)} samples)...")
        save_embeddings_to_h5(
            train_emb,
            train_labels,
            train_h5,
            texts=None,  # Don't save texts in split
            metadata=train_metadata
        )

        # Save validation.h5
        print(f"  → Saving validation.h5 ({len(val_indices)} samples)...")
        save_embeddings_to_h5(
            val_emb,
            val_labels,
            val_h5,
            texts=None,
            metadata=val_metadata
        )

        return True

    except Exception as e:
        print(f"  ✗ Error splitting train data: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_pipeline(
    model_family: str,
    model_size: str,
    tasks: List[str],
    output_base_dir: str = "./precomputed_embeddings",
    pooling_method: str = "mean",
    splits: List[str] = None,
    batch_size: int = 256,
    force_recompute: bool = False,
    revision: str = "main",
    save_texts: bool = True,
    chunk_size: int = 5000
):
    """
    Main pipeline: load LLM once, extract embeddings for all tasks/splits.

    Args:
        model_family: e.g., 'Pythia', 'Llama3'
        model_size: e.g., '410m', '8B'
        tasks: List of task names to process
        output_base_dir: Base directory for saving embeddings
        pooling_method: Token pooling method ('mean', 'first_hidden_state', 'last_hidden_state')
        splits: List of splits to process (default: ['train', 'validation', 'test'])
        batch_size: Batch size for embedding extraction
        force_recompute: If True, recompute even if files exist
        revision: Model revision/checkpoint
        save_texts: If True, save original texts for traceability
        chunk_size: Number of samples to process at once per split (to avoid OOM)
    """
    if splits is None:
        splits = ['train', 'validation', 'test']

    # Setup output directory
    output_dir = get_output_dir(output_base_dir, model_family, model_size, pooling_method)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EMBEDDING EXTRACTION PIPELINE")
    print("="*70)
    print(f"Model: {model_family}-{model_size} (revision: {revision})")
    print(f"Pooling: {pooling_method}")
    print(f"Tasks: {len(tasks)} tasks")
    print(f"Splits: {splits}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Chunk size: {chunk_size}")
    print(f"Force recompute: {force_recompute}")
    print(f"Save texts: {save_texts}")
    print("="*70)

    # Load model ONCE
    print(f"\nLoading {model_family}-{model_size}...")
    model_specs = TextModelSpecifications(model_family, model_size, revision, ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(
        model_specs,
        device_map="auto",
        evaluation_layer_idx=-1,  # Extract all layers
        use_memory_efficient_hooks=True  # Use memory-efficient hooks
    )
    print(f"✓ Model loaded. Layers: {wrapper.num_layers}, Device: {wrapper.device}")

    # Adjust batch size for large models to prevent OOM
    original_batch_size = batch_size
    if model_family == "Llama3" and "8B" in model_size:
        # Llama3-8B is huge (~28 GB), use very small batch size
        batch_size = min(batch_size, 4)
        print(f"⚠️  Large model detected! Reducing batch size: {original_batch_size} → {batch_size}")
    elif model_family == "TinyLlama" or (model_family == "Pythia" and "2.8b" in model_size):
        # Medium models, use moderate batch size
        batch_size = min(batch_size, 32)
        if batch_size != original_batch_size:
            print(f"⚠️  Adjusting batch size for model: {original_batch_size} → {batch_size}")

    # Process each task
    results = {}
    total_tasks = len(tasks)

    for task_idx, task_name in enumerate(tasks, 1):
        print(f"\n[{task_idx}/{total_tasks}] Processing: {task_name}")
        print("-"*50)

        results[task_name] = {}

        # Check if validation split exists in the dataset
        validation_exists = True
        if 'validation' in splits:
            try:
                load_task_data(task_name, 'validation')
            except Exception:
                validation_exists = False
                print(f"  ⚠ Validation split not found, will create from 15% of train")

        for split_name in splits:
            # Skip validation if it doesn't exist (will be created from train)
            if split_name == 'validation' and not validation_exists:
                continue

            success = extract_and_save_task_split(
                wrapper=wrapper,
                task_name=task_name,
                split_name=split_name,
                output_dir=output_dir,
                pooling_method=pooling_method,
                batch_size=batch_size,
                force_recompute=force_recompute,
                save_texts=save_texts,
                chunk_size=chunk_size
            )
            results[task_name][split_name] = success

            # If this is train and validation doesn't exist, split train into train+val
            if split_name == 'train' and not validation_exists and 'validation' in splits:
                print(f"  → Creating validation split from 15% of train...")
                success_val = split_train_for_validation(
                    task_name=task_name,
                    output_dir=output_dir,
                    pooling_method=pooling_method,
                    val_ratio=0.15
                )
                results[task_name]['validation'] = success_val
                if success_val:
                    print(f"  ✓ Validation split created successfully")

            # Force memory cleanup after each split
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all operations to complete
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()  # Call twice to be more aggressive

        # Force memory cleanup after each task
        print(f"  → Cleaning up memory after {task_name}...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()  # Call twice to be more aggressive

    # Save manifest
    save_pipeline_manifest(output_dir, model_specs, tasks, pooling_method, results)

    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    success_count = 0
    fail_count = 0

    for task_name, task_results in results.items():
        task_status = "✓" if all(task_results.values()) else "⚠"
        splits_status = ", ".join([f"{s}:{'✓' if ok else '✗'}" for s, ok in task_results.items()])
        print(f"  {task_status} {task_name}: {splits_status}")

        for ok in task_results.values():
            if ok:
                success_count += 1
            else:
                fail_count += 1

    print(f"\nTotal: {success_count} successful, {fail_count} failed")
    print(f"Output directory: {output_dir}")
    print("="*70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Precompute layer-wise embeddings for multiple tasks")

    parser.add_argument("--model_family", type=str, required=True,
                        help="Model family (e.g., Pythia, Llama3)")
    parser.add_argument("--model_size", type=str, required=True,
                        help="Model size (e.g., 410m, 8B)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of task names to process")
    parser.add_argument("--output_dir", type=str, default="./precomputed_embeddings",
                        help="Base directory for saving embeddings")
    parser.add_argument("--pooling_method", type=str, default="mean",
                        choices=["mean", "first_hidden_state", "last_hidden_state"],
                        help="Token pooling method")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"],
                        help="Splits to process")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for embedding extraction")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Recompute embeddings even if files exist")
    parser.add_argument("--revision", type=str, default="main",
                        help="Model revision/checkpoint")
    parser.add_argument("--no_save_texts", action="store_true",
                        help="Don't save original texts (smaller files but less traceability)")
    parser.add_argument("--chunk_size", type=int, default=5000,
                        help="Number of samples to process per chunk (to avoid OOM, default: 5000)")

    args = parser.parse_args()

    run_pipeline(
        model_family=args.model_family,
        model_size=args.model_size,
        tasks=args.tasks,
        output_base_dir=args.output_dir,
        pooling_method=args.pooling_method,
        splits=args.splits,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        revision=args.revision,
        save_texts=not args.no_save_texts,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
