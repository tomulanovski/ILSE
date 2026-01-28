#!/usr/bin/env python3
"""
Evaluate Linear GCN using exact MTEB classification evaluator protocol.
Implements undersampling (8 samples per label) and 10 experiments with averaging.
"""
import argparse
import torch
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from pathlib import Path
import json

from ...precompute.h5_utils import load_embeddings_from_h5
from .gnn_datasets import SingleGraphDataset
from .gnn_models import LayerGINEncoder, SingleClassifier
from torch_geometric.loader import DataLoader as PyGDataLoader


def extract_embeddings(model, layerwise_embeddings, graph_type, batch_size=32, device='cuda'):
    """
    Extract embeddings/logits from trained Linear GCN model.
    
    Args:
        model: Trained model (SingleClassifier with Linear GCN encoder)
        layerwise_embeddings: (N, L, D) layer-wise embeddings
        graph_type: Graph construction type
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        (N, hidden_dim) embeddings for Variant 2, or (N, num_classes) logits for Variant 1
    """
    model.eval()
    
    # Get encoder (remove classifier head)
    encoder = model.enc if hasattr(model, 'enc') else model
    
    # Create graph dataset
    dummy_labels = [0] * len(layerwise_embeddings)
    dataset = SingleGraphDataset(
        layerwise_embeddings,
        dummy_labels,
        graph_type=graph_type,
        add_virtual_node=(graph_type == "virtual_node")
    )
    
    dataloader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract embeddings
    embeddings_list = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = encoder(batch)  # [B, hidden_dim]
            embeddings_list.append(emb.cpu().numpy())
    
    return np.concatenate(embeddings_list, axis=0)


def undersample_data(embeddings, labels, samples_per_label=8, seed=42, idxs=None):
    """
    Undersample data to have `samples_per_label` samples of each label.
    Matches MTEB's _undersample_data implementation exactly.
    
    Args:
        embeddings: (N, D) embeddings array
        labels: (N,) labels array
        samples_per_label: Number of samples per label to keep
        seed: Random seed for reproducibility
        idxs: Optional pre-shuffled indices (for reproducibility across experiments)
    
    Returns:
        Tuple of (undersampled_embeddings, undersampled_labels, shuffled_idxs)
    """
    N = len(labels)
    
    if idxs is None:
        idxs = list(range(N))
    
    # Use RandomState for backward compatibility (matches MTEB)
    rng_state = np.random.RandomState(seed)
    rng_state.shuffle(idxs)
    
    label_counter = defaultdict(int)
    sampled_idxs = []
    
    for i in idxs:
        label = labels[i]
        if label_counter[label] < samples_per_label:
            sampled_idxs.append(i)
            label_counter[label] += 1
    
    sampled_idxs = np.array(sampled_idxs)
    return embeddings[sampled_idxs], labels[sampled_idxs], idxs


def calculate_scores(y_test, y_pred):
    """
    Calculate all MTEB classification metrics.
    Matches MTEB's _calculate_scores implementation.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
    
    Returns:
        Dict with all classification metrics
    """
    is_binary = len(np.unique(y_test)) == 2
    
    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'ap': None,
        'ap_weighted': None,
    }
    
    # Average precision only for binary classification
    if is_binary:
        scores['ap'] = average_precision_score(y_test, y_pred, average='macro')
        scores['ap_weighted'] = average_precision_score(y_test, y_pred, average='weighted')
    
    return scores


def evaluate_task(
    model_path, 
    embeddings_dir, 
    task_name, 
    device='cuda',
    samples_per_label=8,
    n_experiments=10,
    seed=42
):
    """
    Evaluate a task using exact MTEB classification protocol:
    1. Loading trained Linear GCN model
    2. Extracting embeddings from train/test sets
    3. Running n_experiments with undersampled training data (samples_per_label per class)
    4. Training LogisticRegression on extracted embeddings
    5. Evaluating on test set and averaging results
    
    Args:
        model_path: Path to trained model checkpoint
        embeddings_dir: Directory with precomputed embeddings
        task_name: Task name
        device: Device to use
        samples_per_label: Number of samples per label for training (MTEB default: 8)
        n_experiments: Number of experiments to run (MTEB default: 10)
        seed: Base random seed
    
    Returns:
        Dict with evaluation results (averaged across experiments)
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Linear GCN on {task_name} (MTEB Protocol)")
    print(f"Model: {model_path}")
    print(f"Protocol: {n_experiments} experiments, {samples_per_label} samples per label")
    print(f"{'='*80}\n")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    saved_args = checkpoint['args']
    
    # Extract configuration (with backward compatibility)
    graph_type = saved_args['graph_type']
    hidden_dim = saved_args['gin_hidden_dim']
    dropout = saved_args['dropout']
    node_to_choose = saved_args['node_to_choose']
    variant = saved_args.get('variant', 2)  # Default to variant 2 for backward compatibility
    use_variant1 = (variant == 1)
    
    print(f"Configuration:")
    print(f"  Variant: {variant} ({'direct num_classes' if use_variant1 else 'hidden_dim + classifier'})")
    print(f"  Graph type: {graph_type}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Pooling: {node_to_choose}")
    
    # Load precomputed embeddings from separate split files
    task_dir = f"{embeddings_dir}/{task_name}"
    train_h5 = f"{task_dir}/train.h5"
    test_h5 = f"{task_dir}/test.h5"
    
    print(f"\nLoading embeddings from: {task_dir}")
    
    train_emb, train_labels = load_embeddings_from_h5(train_h5, "train")
    test_emb, test_labels = load_embeddings_from_h5(test_h5, "test")
    
    N_train, L, D = train_emb.shape
    N_test = len(test_labels)
    num_classes = len(np.unique(train_labels))
    
    print(f"Train: {N_train} samples, {L} layers, {D} dims")
    print(f"Test: {N_test} samples")
    print(f"Classes: {num_classes}")
    
    # Reconstruct model
    print("\nReconstructing model...")
    encoder = LayerGINEncoder(
        in_dim=D,
        hidden_dim=hidden_dim,
        num_layers=1,  # Linear GCN is always 1 layer
        dropout=dropout,
        gin_mlp_layers=0,  # GCN mode
        node_to_choose=node_to_choose,
        graph_type=graph_type,
        use_linear=True,  # Linear GCN (no ReLU)
        output_dim=num_classes if use_variant1 else None  # Variant 1: direct num_classes output
    )
    
    model = SingleClassifier(encoder, hidden_dim, num_classes, use_variant1=use_variant1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Extract embeddings/logits ONCE (before experiments)
    print("\nExtracting train embeddings/logits...")
    train_final = extract_embeddings(model, train_emb, graph_type, device=device)
    print(f"Train embeddings/logits: {train_final.shape}")
    
    print("Extracting test embeddings/logits...")
    test_final = extract_embeddings(model, test_emb, graph_type, device=device)
    print(f"Test embeddings/logits: {test_final.shape}")
    
    # Determine evaluation strategy based on variant
    if use_variant1:
        print(f"\n{'='*80}")
        print(f"Variant 1: Using logits directly (no LogisticRegression training)")
        print(f"Logits shape: {train_final.shape[1]} (num_classes)")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print(f"Variant 2: Training LogisticRegression on embeddings")
        print(f"Embeddings shape: {train_final.shape[1]} (hidden_dim)")
        print(f"{'='*80}\n")
    
    # Run multiple experiments with undersampling
    print(f"\n{'='*80}")
    print(f"Running {n_experiments} experiments with MTEB protocol...")
    print(f"{'='*80}\n")
    
    all_scores = []
    idxs = None  # Will be reused across experiments for reproducibility
    
    for i in range(n_experiments):
        print(f"Experiment {i+1}/{n_experiments}...")
        
        # Undersample training data (matches MTEB's _undersample_data)
        # MTEB uses same seed for all experiments, shuffles idxs once, then samples
        train_emb_subset, train_labels_subset, idxs = undersample_data(
            train_final,
            train_labels,
            samples_per_label=samples_per_label,
            seed=seed,  # Same seed for all experiments (matches MTEB)
            idxs=idxs  # Reuse shuffled indices for reproducibility
        )
        
        print(f"  Undersampled train: {len(train_emb_subset)} samples "
              f"(target: {samples_per_label} per label)")
        
        if use_variant1:
            # Variant 1: Encoder outputs logits directly, use argmax
            # No training needed - logits are already class predictions
            print("  Variant 1: Using logits directly (argmax)")
            predictions = np.argmax(test_final, axis=1)
        else:
            # Variant 2: Train LogisticRegression on embeddings
            print("  Variant 2: Training LogisticRegression on embeddings")
            clf = LogisticRegression(max_iter=100, random_state=seed, n_jobs=-1)
            clf.fit(train_emb_subset, train_labels_subset)
            predictions = clf.predict(test_final)
        
        # Calculate scores
        scores_exp = calculate_scores(test_labels, predictions)
        all_scores.append(scores_exp)
        
        print(f"  Accuracy: {scores_exp['accuracy']:.4f}, "
              f"F1 (macro): {scores_exp['f1']:.4f}")
    
    # Average scores across experiments (matches MTEB's averaging logic)
    print(f"\n{'='*80}")
    print("Averaging results across experiments...")
    print(f"{'='*80}\n")
    
    avg_scores = {}
    for key in all_scores[0].keys():
        values = [s[key] for s in all_scores if s[key] is not None]
        if values:
            avg_scores[key] = float(np.mean(values))
        else:
            avg_scores[key] = None
    
    # Store per-experiment scores for analysis
    results = {
        'task_name': task_name,
        **avg_scores,  # Main scores (averaged)
        'scores_per_experiment': all_scores,  # Individual experiment scores
        'n_experiments': n_experiments,
        'samples_per_label': samples_per_label,
        'train_samples_original': int(N_train),
        'train_samples_undersampled': len(train_emb_subset),
        'test_samples': int(N_test),
        'num_classes': int(num_classes),
        'variant': variant,
        'graph_type': graph_type,
        'hidden_dim': hidden_dim,
        'pooling': node_to_choose,
        'model_path': str(model_path),
        'evaluation_protocol': 'MTEB_exact'
    }
    
    print("Final Results (averaged):")
    print(f"  Accuracy: {avg_scores['accuracy']:.4f}")
    print(f"  F1 (macro): {avg_scores['f1']:.4f}")
    print(f"  F1 (weighted): {avg_scores['f1_weighted']:.4f}")
    print(f"  Precision (macro): {avg_scores['precision']:.4f}")
    print(f"  Recall (macro): {avg_scores['recall']:.4f}")
    if avg_scores['ap'] is not None:
        print(f"  AP (macro): {avg_scores['ap']:.4f}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Linear GCN with exact MTEB classification protocol"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                        help="Directory with precomputed embeddings")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name")
    parser.add_argument("--output_dir", type=str, default="./linear_gcn_results",
                        help="Directory to save results")
    parser.add_argument("--samples_per_label", type=int, default=8,
                        help="Number of samples per label for training (MTEB default: 8)")
    parser.add_argument("--n_experiments", type=int, default=10,
                        help="Number of experiments to run (MTEB default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluate with MTEB protocol
    results = evaluate_task(
        args.model_path, 
        args.embeddings_dir, 
        args.task, 
        device=device,
        samples_per_label=args.samples_per_label,
        n_experiments=args.n_experiments,
        seed=args.seed
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Include variant and config in filename to avoid collisions
    graph_type = results['graph_type']
    pooling = results['pooling']
    variant = results['variant']
    output_path = output_dir / f"{args.task}_{graph_type}_{pooling}_v{variant}_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

