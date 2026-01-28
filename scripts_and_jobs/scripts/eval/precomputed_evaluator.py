#!/usr/bin/env python3
"""
Evaluate GNN/MLP models using precomputed embeddings.
No LLM loading required - fast evaluation!
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from experiments.utils.precompute.h5_utils import load_embeddings_from_h5, get_h5_metadata
from experiments.utils.model_definitions.gnn.gnn_models import LayerGINEncoder, MLPEncoder, LearnedWeightingEncoder
from experiments.utils.model_definitions.gnn.gnn_datasets import SingleGraphDataset, SimpleTensorDataset, LayerwiseTensorDataset


def load_gnn_encoder_from_checkpoint(checkpoint_path: str, input_dim: int, device: torch.device):
    """
    Load GNN encoder from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        input_dim: Hidden dimension of layer embeddings (D)
        device: Device to load model on

    Returns:
        GNN encoder, saved_args dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint['args']

    # Create encoder
    encoder = LayerGINEncoder(
        in_dim=input_dim,
        hidden_dim=saved_args.get('gin_hidden_dim', 256),
        num_layers=saved_args.get('gin_layers', 3),
        dropout=saved_args.get('dropout', 0.1),
        gin_mlp_layers=saved_args.get('gin_mlp_layers', 1),
        node_to_choose=saved_args.get('node_to_choose', 'last'),
        graph_type=saved_args['graph_type']
    )

    # Load weights - extract from classifier wrapper
    model_state = checkpoint['model_state_dict']
    # Extract encoder weights (saved with 'enc.' prefix in SingleClassifier)
    encoder_state = {k.replace('enc.', ''): v for k, v in model_state.items() if k.startswith('enc.')}
    encoder.load_state_dict(encoder_state)

    return encoder.to(device).eval(), saved_args


def load_mlp_encoder_from_checkpoint(checkpoint_path: str, input_dim: int, device: torch.device):
    """
    Load MLP encoder from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        input_dim: Input dimension (D for last/mean, L*D for flatten)
        device: Device to load model on

    Returns:
        MLP encoder, saved_args dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint['args']

    encoder = MLPEncoder(
        in_dim=input_dim,
        hidden_dim=saved_args.get('mlp_hidden_dim', 256),
        num_layers=saved_args.get('mlp_layers', 2),
        dropout=saved_args.get('dropout', 0.1)
    )

    # Load weights - extract from classifier wrapper
    model_state = checkpoint['model_state_dict']
    # Extract encoder weights (saved with 'enc.' prefix in SingleClassifierLinearHead)
    encoder_state = {k.replace('enc.', ''): v for k, v in model_state.items() if k.startswith('enc.')}
    encoder.load_state_dict(encoder_state)

    return encoder.to(device).eval(), saved_args


def load_weighted_encoder_from_checkpoint(checkpoint_path: str, num_layers: int, layer_dim: int, device: torch.device):
    """
    Load Weighted (learned layer weighting) encoder from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        num_layers: Number of layers in the model (L)
        layer_dim: Dimension of each layer embedding (D)
        device: Device to load model on

    Returns:
        Weighted encoder, saved_args dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint['args']

    encoder = LearnedWeightingEncoder(
        num_layers=num_layers,
        layer_dim=layer_dim
    )

    # Load weights - extract from classifier wrapper
    model_state = checkpoint['model_state_dict']
    # Extract encoder weights (saved with 'enc.' prefix in SingleClassifierLinearHead)
    encoder_state = {k.replace('enc.', ''): v for k, v in model_state.items() if k.startswith('enc.')}
    encoder.load_state_dict(encoder_state)

    return encoder.to(device).eval(), saved_args


def encode_with_gnn(
    encoder: LayerGINEncoder,
    embeddings: np.ndarray,
    graph_type: str = "fully_connected",
    batch_size: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """
    Pass precomputed layer embeddings through GNN encoder.

    Args:
        encoder: Trained GNN encoder
        embeddings: (N, L, D) layer-wise embeddings
        graph_type: Graph construction type
        batch_size: Batch size for processing
        device: Device

    Returns:
        (N, hidden_dim) final embeddings
    """
    if device is None:
        device = next(encoder.parameters()).device

    # Create graph dataset (dummy labels)
    dummy_labels = [0] * len(embeddings)
    layerwise_list = [embeddings[i] for i in range(len(embeddings))]
    dataset = SingleGraphDataset(layerwise_list, dummy_labels, graph_type=graph_type)
    dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Encode
    all_embeddings = []
    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = encoder(batch)  # (B, hidden_dim)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def encode_with_mlp(
    encoder: MLPEncoder,
    embeddings: np.ndarray,
    mlp_input_mode: str = "last",
    batch_size: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """
    Pass precomputed layer embeddings through MLP encoder.

    Args:
        encoder: Trained MLP encoder
        embeddings: (N, L, D) layer-wise embeddings
        mlp_input_mode: 'last', 'mean', or 'flatten'
        batch_size: Batch size
        device: Device

    Returns:
        (N, hidden_dim) final embeddings
    """
    if device is None:
        device = next(encoder.parameters()).device

    # Prepare input based on mode
    N, L, D = embeddings.shape
    if mlp_input_mode == "last":
        X = embeddings[:, -1, :]  # (N, D)
    elif mlp_input_mode == "mean":
        X = embeddings.mean(axis=1)  # (N, D)
    elif mlp_input_mode == "flatten":
        X = embeddings.reshape(N, L * D)  # (N, L*D)
    else:
        raise ValueError(f"Unknown mlp_input_mode: {mlp_input_mode}")

    X = torch.from_numpy(X).float()

    # Encode in batches
    all_embeddings = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            emb = encoder(batch_X)  # (B, hidden_dim)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def encode_with_weighted(
    encoder: LearnedWeightingEncoder,
    embeddings: np.ndarray,
    batch_size: int = 32,
    device: torch.device = None
) -> np.ndarray:
    """
    Pass precomputed layer embeddings through Weighted (learned layer weighting) encoder.

    Args:
        encoder: Trained Weighted encoder
        embeddings: (N, L, D) layer-wise embeddings
        batch_size: Batch size
        device: Device

    Returns:
        (N, D) final embeddings (softmax-weighted combination of layers)
    """
    if device is None:
        device = next(encoder.parameters()).device

    # Create tensor dataset (dummy labels)
    dummy_labels = [0] * len(embeddings)
    layerwise_list = [embeddings[i] for i in range(len(embeddings))]
    dataset = LayerwiseTensorDataset(layerwise_list, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Encode
    all_embeddings = []
    encoder.eval()
    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)  # (B, L, D)
            emb = encoder(batch_X)  # (B, D)
            all_embeddings.append(emb.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def evaluate_task_with_precomputed(
    model_path: str,
    embeddings_dir: str,
    task_name: str,
    model_type: str = "auto",
    batch_size: int = 32,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluate a trained GNN/MLP model on a task using precomputed embeddings.

    Args:
        model_path: Path to trained model checkpoint
        embeddings_dir: Directory containing precomputed embeddings
        task_name: Name of the task
        model_type: 'gin', 'mlp', or 'auto' (detect from filename)
        batch_size: Batch size for encoding
        device: Device

    Returns:
        Dict with metrics (accuracy, f1, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect model type
    if model_type == "auto":
        if "gin" in model_path.lower() or "gcn" in model_path.lower():
            model_type = "gin"
        elif "mlp" in model_path.lower():
            model_type = "mlp"
        else:
            raise ValueError(f"Cannot auto-detect model type from {model_path}")

    # Load embeddings
    task_dir = Path(embeddings_dir) / task_name
    train_h5 = task_dir / "train.h5"
    test_h5 = task_dir / "test.h5"

    if not train_h5.exists():
        raise FileNotFoundError(f"Train embeddings not found: {train_h5}")
    if not test_h5.exists():
        raise FileNotFoundError(f"Test embeddings not found: {test_h5}")

    print(f"Loading precomputed embeddings for {task_name}...")
    train_emb, train_labels, train_meta = load_embeddings_from_h5(train_h5, load_metadata=True)
    test_emb, test_labels, test_meta = load_embeddings_from_h5(test_h5, load_metadata=True)

    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

    N, L, D = train_emb.shape

    # Load model and encode
    if model_type == "gin":
        encoder, saved_args = load_gnn_encoder_from_checkpoint(model_path, D, device)
        graph_type = saved_args.get('graph_type', 'fully_connected')

        print(f"Encoding with GNN (graph_type={graph_type})...")
        train_final = encode_with_gnn(encoder, train_emb, graph_type, batch_size, device)
        test_final = encode_with_gnn(encoder, test_emb, graph_type, batch_size, device)

    else:  # mlp
        mlp_input_mode = "last"  # Default, could be read from saved_args
        # Determine input dimension
        checkpoint = torch.load(model_path, map_location='cpu')
        saved_args = checkpoint['args']
        mlp_input_mode = saved_args.get('mlp_input', 'last')

        if mlp_input_mode in ('last', 'mean'):
            in_dim = D
        else:  # flatten
            in_dim = L * D

        encoder, saved_args = load_mlp_encoder_from_checkpoint(model_path, in_dim, device)

        print(f"Encoding with MLP (input_mode={mlp_input_mode})...")
        train_final = encode_with_mlp(encoder, train_emb, mlp_input_mode, batch_size, device)
        test_final = encode_with_mlp(encoder, test_emb, mlp_input_mode, batch_size, device)

    print(f"Final embeddings: Train {train_final.shape}, Test {test_final.shape}")

    # Train classifier
    print("Training logistic regression classifier...")
    clf = LogisticRegression(max_iter=100, random_state=42, n_jobs=-1)
    clf.fit(train_final, train_labels)

    # Predict and evaluate
    predictions = clf.predict(test_final)

    accuracy = accuracy_score(test_labels, predictions)
    f1_micro = f1_score(test_labels, predictions, average='micro')
    f1_macro = f1_score(test_labels, predictions, average='macro')

    results = {
        'task_name': task_name,
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'train_samples': len(train_labels),
        'test_samples': len(test_labels),
        'model_type': model_type,
        'model_path': str(model_path),
    }

    print(f"\nResults for {task_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  F1 (micro): {f1_micro:.4f}")

    return results


def evaluate_multiple_tasks(
    model_path: str,
    embeddings_dir: str,
    task_names: List[str],
    model_type: str = "auto",
    batch_size: int = 32,
    output_json: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate model on multiple tasks using precomputed embeddings.

    Args:
        model_path: Path to trained model checkpoint
        embeddings_dir: Directory containing precomputed embeddings
        task_names: List of task names
        model_type: 'gin', 'mlp', or 'auto'
        batch_size: Batch size
        output_json: Path to save results JSON (optional)

    Returns:
        Dict mapping task_name to results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*70)
    print("PRECOMPUTED EMBEDDINGS EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Tasks: {task_names}")
    print(f"Device: {device}")
    print("="*70)

    all_results = {}

    for i, task_name in enumerate(task_names, 1):
        print(f"\n[{i}/{len(task_names)}] Evaluating: {task_name}")
        print("-"*50)

        try:
            results = evaluate_task_with_precomputed(
                model_path=model_path,
                embeddings_dir=embeddings_dir,
                task_name=task_name,
                model_type=model_type,
                batch_size=batch_size,
                device=device
            )
            all_results[task_name] = results
        except Exception as e:
            print(f"✗ Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {'error': str(e)}

    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    successful = {k: v for k, v in all_results.items() if 'accuracy' in v}
    if successful:
        print(f"{'Task':<40} {'Accuracy':<12} {'F1 (macro)':<12}")
        print("-"*70)
        for task_name, res in successful.items():
            print(f"{task_name:<40} {res['accuracy']:<12.4f} {res['f1_macro']:<12.4f}")

        # Average
        avg_acc = np.mean([r['accuracy'] for r in successful.values()])
        avg_f1 = np.mean([r['f1_macro'] for r in successful.values()])
        print("-"*70)
        print(f"{'AVERAGE':<40} {avg_acc:<12.4f} {avg_f1:<12.4f}")

    failed = {k: v for k, v in all_results.items() if 'error' in v}
    if failed:
        print(f"\nFailed tasks: {list(failed.keys())}")

    print("="*70)

    # Save results
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN/MLP models using precomputed embeddings")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory containing precomputed embeddings (e.g., precomputed_embeddings/Pythia_410m_mean_pooling)")
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="List of task names to evaluate")
    parser.add_argument("--model_type", type=str, default="auto", choices=["gin", "mlp", "auto"],
                        help="Model type (auto-detect from filename if not specified)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    evaluate_multiple_tasks(
        model_path=args.model_path,
        embeddings_dir=args.embeddings_dir,
        task_names=args.tasks,
        model_type=args.model_type,
        batch_size=args.batch_size,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
