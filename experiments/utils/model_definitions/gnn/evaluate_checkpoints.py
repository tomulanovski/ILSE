#!/usr/bin/env python3
"""
Evaluate all saved checkpoints on the test set.
Loads checkpoints from the checkpoints/ directory and evaluates each on test data.
"""
import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
import pandas as pd
import numpy as np

from ..text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from .gnn_datasets import load_task_data, compute_layerwise, SingleGraphDataset, SimpleTensorDataset
from .gnn_models import LayerGINEncoder, SingleClassifier, MLPEncoder, SingleClassifierLinearHead
from .basic_gin_trainer import validate_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints on test set")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoints directory (e.g., ./basic_gin_models/checkpoints)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (must match checkpoint filenames)")
    parser.add_argument("--model_family", type=str, required=True,
                        help="Model family (must match checkpoint filenames)")
    parser.add_argument("--model_size", type=str, required=True,
                        help="Model size (must match checkpoint filenames)")
    parser.add_argument("--encoder", type=str, required=True, choices=["gin", "mlp"],
                        help="Encoder type (must match checkpoint filenames)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save results CSV (optional)")
    return parser.parse_args()


def load_checkpoint_and_build_model(checkpoint_path, num_classes, device):
    """Load checkpoint and reconstruct the model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint["args"]

    # Reconstruct model architecture from saved args
    encoder_type = args_dict["encoder"]

    if encoder_type == "gin":
        # Get dimensions from checkpoint (we'll need layerwise data to know L, D)
        gin_encoder = LayerGINEncoder(
            in_dim=args_dict.get("gin_hidden_dim", 256),  # This will be corrected below
            hidden_dim=args_dict.get("gin_hidden_dim", 256),
            num_layers=args_dict.get("gin_layers", 3),
            dropout=args_dict.get("dropout", 0.1),
            gin_mlp_layers=args_dict.get("gin_mlp_layers", 1),
            node_to_choose=args_dict.get("node_to_choose", "last"),
            graph_type=args_dict["graph_type"]
        )
        model = SingleClassifier(gin_encoder, args_dict.get("gin_hidden_dim", 256), num_classes)
    else:  # mlp
        mlp_input_mode = args_dict.get("mlp_input", "last")
        # in_dim will be set correctly when we have layerwise data
        mlp_encoder = MLPEncoder(
            in_dim=args_dict.get("mlp_hidden_dim", 256),  # Placeholder
            hidden_dim=args_dict.get("mlp_hidden_dim", 256),
            num_layers=args_dict.get("mlp_layers", 2),
            dropout=args_dict.get("dropout", 0.1)
        )
        model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, args_dict


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return

    # Find all checkpoint files matching the pattern
    pattern = f"{args.task}_{args.model_family}_{args.model_size}_{args.encoder}_epoch*.pt"
    checkpoint_files = sorted(checkpoint_dir.glob(pattern))

    if not checkpoint_files:
        print(f"Error: No checkpoint files found matching pattern: {pattern}")
        print(f"Searched in: {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s) to evaluate")
    print(f"Task: {args.task} | Model: {args.model_family}-{args.model_size} | Encoder: {args.encoder}")
    print(f"Device: {device}\n")

    # Load test data
    print("Loading test data...")
    try:
        test_data = load_task_data(args.task, "test")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Load model wrapper for extracting embeddings
    specs = TextModelSpecifications(args.model_family, args.model_size, "main", ignore_checks=True)
    wrapper = TextLayerwiseAutoModelWrapper(specs, device_map="auto", evaluation_layer_idx=-1)

    print("Extracting layer-wise embeddings for test set...")
    test_layerwise = compute_layerwise(wrapper, test_data["text"], batch_size=256, token_pooling_method="mean")
    num_classes = test_data["num_classes"]

    # Load one checkpoint to get architecture details
    sample_checkpoint = torch.load(checkpoint_files[0], map_location='cpu')
    args_dict = sample_checkpoint["args"]
    encoder_type = args_dict["encoder"]

    # Prepare test dataset and loader
    L, D = test_layerwise[0].shape

    if encoder_type == "gin":
        graph_type = args_dict.get("graph_type", "fully_connected")
        test_dataset = SingleGraphDataset(test_layerwise, test_data["labels"], graph_type=graph_type)
        test_loader = PyGDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        eval_mode = "gin"

        # Build model with correct dimensions
        gin_encoder = LayerGINEncoder(
            in_dim=D,
            hidden_dim=args_dict.get("gin_hidden_dim", 256),
            num_layers=args_dict.get("gin_layers", 3),
            dropout=args_dict.get("dropout", 0.1),
            gin_mlp_layers=args_dict.get("gin_mlp_layers", 1),
            node_to_choose=args_dict.get("node_to_choose", "last"),
            graph_type=args_dict["graph_type"]
        )
        base_model = SingleClassifier(gin_encoder, args_dict.get("gin_hidden_dim", 256), num_classes).to(device)
    else:  # mlp
        mlp_input_mode = args_dict.get("mlp_input", "last")
        if mlp_input_mode in ("last", "mean"):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        test_dataset = SimpleTensorDataset(test_layerwise, test_data["labels"], mode=mlp_input_mode)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        eval_mode = "mlp"

        mlp_encoder = MLPEncoder(
            in_dim=in_dim,
            hidden_dim=args_dict.get("mlp_hidden_dim", 256),
            num_layers=args_dict.get("mlp_layers", 2),
            dropout=args_dict.get("dropout", 0.1)
        )
        base_model = SingleClassifierLinearHead(mlp_encoder, mlp_encoder.out_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    # Evaluate each checkpoint
    results = []
    print("\n" + "="*80)
    print(f"{'Checkpoint':<40} {'Epoch':<8} {'Val Acc':<10} {'Test Acc':<10} {'Test Loss':<10}")
    print("="*80)

    for checkpoint_path in checkpoint_files:
        # Extract epoch number from filename
        filename = checkpoint_path.stem
        epoch_str = filename.split("_epoch")[-1]
        try:
            epoch_num = int(epoch_str)
        except ValueError:
            epoch_num = -1

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(checkpoint["model_state_dict"])
        base_model.eval()

        # Evaluate on test set
        test_loss, test_acc = validate_epoch(base_model, test_loader, criterion, device, eval_mode)

        # Get validation accuracy from checkpoint
        val_acc = checkpoint.get("val_acc", -1)

        print(f"{checkpoint_path.name:<40} {epoch_num:<8} {val_acc:<10.4f} {test_acc:<10.4f} {test_loss:<10.4f}")

        results.append({
            "checkpoint": checkpoint_path.name,
            "epoch": epoch_num,
            "train_acc": checkpoint.get("train_acc", -1),
            "train_loss": checkpoint.get("train_loss", -1),
            "val_acc": val_acc,
            "val_loss": checkpoint.get("val_loss", -1),
            "test_acc": test_acc,
            "test_loss": test_loss,
        })

    print("="*80)

    # Create DataFrame and sort by epoch
    df = pd.DataFrame(results)
    df = df.sort_values("epoch")

    # Summary statistics
    print("\nSummary:")
    print(f"  Best test accuracy: {df['test_acc'].max():.4f} (epoch {df.loc[df['test_acc'].idxmax(), 'epoch']:.0f})")
    print(f"  Worst test accuracy: {df['test_acc'].min():.4f} (epoch {df.loc[df['test_acc'].idxmin(), 'epoch']:.0f})")
    print(f"  Mean test accuracy: {df['test_acc'].mean():.4f}")
    print(f"  Std test accuracy: {df['test_acc'].std():.4f}")

    # Check if best validation = best test
    best_val_epoch = df.loc[df['val_acc'].idxmax(), 'epoch']
    best_test_epoch = df.loc[df['test_acc'].idxmax(), 'epoch']
    if best_val_epoch == best_test_epoch:
        print(f"\n✓ Best validation epoch ({best_val_epoch:.0f}) matches best test epoch!")
    else:
        print(f"\n⚠ Best validation epoch ({best_val_epoch:.0f}) differs from best test epoch ({best_test_epoch:.0f})")
        print(f"  Val acc at best val epoch: {df.loc[df['val_acc'].idxmax(), 'val_acc']:.4f}")
        print(f"  Test acc at best val epoch: {df.loc[df['val_acc'].idxmax(), 'test_acc']:.4f}")
        print(f"  Test acc at best test epoch: {df.loc[df['test_acc'].idxmax(), 'test_acc']:.4f}")

    # Save to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
