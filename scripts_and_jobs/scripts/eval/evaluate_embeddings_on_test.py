#!/usr/bin/env python3
"""
Flexible evaluation script for different embedding types.
Supports: single layer, mean pooling, GNN-encoded, MLP-encoded embeddings.
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

from experiments.utils.precompute.h5_utils import load_embeddings_from_h5, save_embeddings_to_h5
from .precomputed_evaluator import (
    load_gnn_encoder_from_checkpoint,
    load_mlp_encoder_from_checkpoint,
    encode_with_gnn,
    encode_with_mlp
)
from .mteb_embedding_evaluator import evaluate_embeddings


def extract_embeddings(
    embedding_type: str,
    layerwise_embeddings: np.ndarray,
    layer_idx: int = None,
    model_path: str = None,
    graph_type: str = "fully_connected",
    mlp_input_mode: str = "last",
    batch_size: int = 32,
    device: torch.device = None,
    encoder=None
) -> np.ndarray:
    """
    Extract embeddings based on type.

    Args:
        embedding_type: 'layer', 'mean', 'gin', or 'mlp'
        layerwise_embeddings: (N, L, D) layer-wise embeddings
        layer_idx: Layer index for 'layer' type
        model_path: Path to model checkpoint for 'gin'/'mlp'
        graph_type: Graph type for GNN
        mlp_input_mode: Input mode for MLP
        batch_size: Batch size for encoding
        device: Device
        encoder: Pre-loaded encoder (optional, for efficiency)

    Returns:
        (N, D) final embeddings
    """
    N, L, D = layerwise_embeddings.shape

    if embedding_type == "layer":
        if layer_idx is None:
            raise ValueError("--layer_idx required for embedding_type='layer'")
        print(f"  Extracting layer {layer_idx} embeddings...")
        return layerwise_embeddings[:, layer_idx, :]  # (N, D)

    elif embedding_type == "mean":
        print(f"  Mean-pooling {L} layers...")
        return layerwise_embeddings.mean(axis=1)  # (N, D)

    elif embedding_type == "gin":
        if model_path is None and encoder is None:
            raise ValueError("--model_path required for embedding_type='gin'")

        if encoder is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder, saved_args = load_gnn_encoder_from_checkpoint(model_path, D, device)
            graph_type = saved_args.get('graph_type', 'fully_connected')

        print(f"  Encoding with GNN (graph_type={graph_type})...")
        return encode_with_gnn(encoder, layerwise_embeddings, graph_type, batch_size, device)

    elif embedding_type == "mlp":
        if model_path is None and encoder is None:
            raise ValueError("--model_path required for embedding_type='mlp'")

        if encoder is None:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load to determine input mode
            checkpoint = torch.load(model_path, map_location='cpu')
            saved_args = checkpoint['args']
            mlp_input_mode = saved_args.get('mlp_input', 'last')

            if mlp_input_mode in ('last', 'mean'):
                in_dim = D
            else:  # flatten
                in_dim = L * D

            encoder, saved_args = load_mlp_encoder_from_checkpoint(model_path, in_dim, device)

        print(f"  Encoding with MLP (input_mode={mlp_input_mode})...")
        return encode_with_mlp(encoder, layerwise_embeddings, mlp_input_mode, batch_size, device)

    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")


def evaluate_task(
    embeddings_dir: str,
    task_name: str,
    embedding_type: str,
    layer_idx: int = None,
    model_path: str = None,
    batch_size: int = 32,
    save_embeddings: bool = False,
    output_embeddings_dir: str = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluate a task using specified embedding type.

    Args:
        embeddings_dir: Directory with precomputed layer-wise embeddings
        task_name: Task name
        embedding_type: 'layer', 'mean', 'gin', or 'mlp'
        layer_idx: Layer index (for 'layer' type)
        model_path: Model checkpoint path (for 'gin'/'mlp')
        batch_size: Batch size
        save_embeddings: Whether to save extracted embeddings
        output_embeddings_dir: Where to save embeddings
        device: Device

    Returns:
        Evaluation results dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"EVALUATING: {task_name}")
    print(f"Embedding type: {embedding_type}")
    print(f"{'='*70}\n")

    task_dir = Path(embeddings_dir) / task_name

    # Load precomputed embeddings
    train_h5 = task_dir / "train.h5"
    test_h5 = task_dir / "test.h5"

    if not train_h5.exists():
        raise FileNotFoundError(f"Train embeddings not found: {train_h5}")
    if not test_h5.exists():
        raise FileNotFoundError(f"Test embeddings not found: {test_h5}")

    print("Loading precomputed layer-wise embeddings...")
    train_layerwise, train_labels, train_meta = load_embeddings_from_h5(train_h5, load_metadata=True)
    test_layerwise, test_labels, test_meta = load_embeddings_from_h5(test_h5, load_metadata=True)
    print(f"  Train: {train_layerwise.shape}")
    print(f"  Test: {test_layerwise.shape}")

    # Load encoder once if needed
    encoder = None
    graph_type = "fully_connected"
    mlp_input_mode = "last"

    if embedding_type in ("gin", "mlp"):
        N, L, D = train_layerwise.shape

        if embedding_type == "gin":
            encoder, saved_args = load_gnn_encoder_from_checkpoint(model_path, D, device)
            graph_type = saved_args.get('graph_type', 'fully_connected')
        else:  # mlp
            checkpoint = torch.load(model_path, map_location='cpu')
            saved_args = checkpoint['args']
            mlp_input_mode = saved_args.get('mlp_input', 'last')

            if mlp_input_mode in ('last', 'mean'):
                in_dim = D
            else:
                in_dim = L * D

            encoder, _ = load_mlp_encoder_from_checkpoint(model_path, in_dim, device)

    # Extract embeddings
    print("\nExtracting train embeddings...")
    train_final = extract_embeddings(
        embedding_type=embedding_type,
        layerwise_embeddings=train_layerwise,
        layer_idx=layer_idx,
        model_path=model_path,
        graph_type=graph_type,
        mlp_input_mode=mlp_input_mode,
        batch_size=batch_size,
        device=device,
        encoder=encoder
    )

    print("Extracting test embeddings...")
    test_final = extract_embeddings(
        embedding_type=embedding_type,
        layerwise_embeddings=test_layerwise,
        layer_idx=layer_idx,
        model_path=model_path,
        graph_type=graph_type,
        mlp_input_mode=mlp_input_mode,
        batch_size=batch_size,
        device=device,
        encoder=encoder
    )

    print(f"\nFinal embeddings:")
    print(f"  Train: {train_final.shape}")
    print(f"  Test: {test_final.shape}")

    # Save embeddings if requested
    if save_embeddings:
        if output_embeddings_dir is None:
            raise ValueError("--output_embeddings_dir required when --save_embeddings is set")

        output_task_dir = Path(output_embeddings_dir) / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving embeddings to {output_task_dir}...")

        # Reshape to (N, 1, D) for h5_utils compatibility
        train_final_reshaped = train_final[:, np.newaxis, :]
        test_final_reshaped = test_final[:, np.newaxis, :]

        # Save train
        train_output_h5 = output_task_dir / "train.h5"
        save_embeddings_to_h5(
            embeddings=train_final_reshaped,
            labels=train_labels,
            save_path=train_output_h5,
            metadata={
                'task_name': task_name,
                'split_name': 'train',
                'embedding_type': embedding_type,
                'layer_idx': int(layer_idx) if layer_idx is not None else -1,
                'model_path': str(model_path) if model_path else '',
                'source_embeddings_dir': str(embeddings_dir),
            }
        )

        # Save test
        test_output_h5 = output_task_dir / "test.h5"
        save_embeddings_to_h5(
            embeddings=test_final_reshaped,
            labels=test_labels,
            save_path=test_output_h5,
            metadata={
                'task_name': task_name,
                'split_name': 'test',
                'embedding_type': embedding_type,
                'layer_idx': int(layer_idx) if layer_idx is not None else -1,
                'model_path': str(model_path) if model_path else '',
                'source_embeddings_dir': str(embeddings_dir),
            }
        )

        print(f"  ✓ Saved train: {train_output_h5}")
        print(f"  ✓ Saved test: {test_output_h5}")

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_embeddings(
        task_type="classification",  # TODO: Add STS support
        task_name=task_name,
        train_embeddings=train_final,
        train_labels=train_labels,
        test_embeddings=test_final,
        test_labels=test_labels
    )

    # Add embedding type info to results
    results['embedding_type'] = embedding_type
    results['layer_idx'] = layer_idx
    results['model_path'] = str(model_path) if model_path else None

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(f"  F1 (micro): {results['f1_micro']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings on test set with flexible embedding types"
    )

    # Required args
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed layer-wise embeddings")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--embedding_type", type=str, required=True,
                        choices=["layer", "mean", "gin", "mlp"],
                        help="Type of embeddings to evaluate")

    # Conditional args
    parser.add_argument("--layer_idx", type=int, default=None,
                        help="Layer index (required for embedding_type='layer')")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model checkpoint path (required for embedding_type='gin'/'mlp')")

    # Optional args
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for GNN/MLP encoding")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Save extracted embeddings to disk")
    parser.add_argument("--output_embeddings_dir", type=str, default=None,
                        help="Directory to save embeddings (required if --save_embeddings)")
    parser.add_argument("--output_results", type=str, required=True,
                        help="Path to save evaluation results JSON")

    args = parser.parse_args()

    # Validate args
    if args.embedding_type == "layer" and args.layer_idx is None:
        parser.error("--layer_idx required for embedding_type='layer'")
    if args.embedding_type in ("gin", "mlp") and args.model_path is None:
        parser.error("--model_path required for embedding_type='gin'/'mlp'")
    if args.save_embeddings and args.output_embeddings_dir is None:
        parser.error("--output_embeddings_dir required when --save_embeddings is set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Evaluate
    results = evaluate_task(
        embeddings_dir=args.embeddings_dir,
        task_name=args.task,
        embedding_type=args.embedding_type,
        layer_idx=args.layer_idx,
        model_path=args.model_path,
        batch_size=args.batch_size,
        save_embeddings=args.save_embeddings,
        output_embeddings_dir=args.output_embeddings_dir,
        device=device
    )

    # Save results
    output_path = Path(args.output_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
